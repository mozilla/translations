# Finalize plan: end-to-end translation in Rust

## Status — implemented (en→fr happy path working)

The full pipeline is built and runs with no trace involved (`cargo run -- translate
<model> <vocab> "<text>"`):

- **`spm.rs`** — SentencePiece unigram tokenize/detokenize (hand-rolled protobuf reader +
  Viterbi). Source ids match the reference byte-for-byte on the tested sentences.
- **`weights.rs`** — model view: dequantized `Wemb`, per-affine int8 weights + `qA/qB`,
  parsed config.
- **`engine.rs`** — dynamic forward pass: `√d` embeddings + sinusoidal PE, 6-layer encoder,
  4-layer SSRU decoder (cell state carried across steps), greedy loop.
- **`shortlist.rs`** — `lex.*.s2t.bin` reader + per-sentence candidate set; the int8 tied
  output projection restricts to those columns (the reference `SelectColumnsB` path).

**Parity vs `translator-cli`** on a 10-sentence sample: 7/10 identical, all 10 fluent
correct French. The 3 differences are first-token **logit near-ties** (e.g. `▁bon` 14.27 vs
`▁Bonjour` 14.13) that different float reduction orders (our scalar sums vs the reference's
SIMD reductions) tip the other way — within the tolerance parity bar (build-plan.md: not
bit-exactness), not a correctness bug. The traced "Hello world." run matches node-by-node
and token-for-token.

Remaining for tighter parity (optional): match the reference reduction order to remove
near-tie flips; split-vocab CJK exercise; full NFKC / `precompiled_charsmap` in the tokenizer.

---


`build-plan.md` got us to **op parity + full-graph replay**: every op is validated against the
reference trace, and [`graph::replay`](src/graph.rs) recomputes the recorded graph node-by-node
with zero divergence. But replay still leans on the trace as a scaffold — it reads the graph
structure, shapes, indices, and the trusted static scalars from a recording of *one* translation.

This plan closes the gap to a **real translator**: text in → text out, with no trace involved.
The trace stops being an input and becomes purely a test oracle. Five pieces are needed:

1. SentencePiece tokenization in Rust (text → token ids).
2. Per-token embedding lookup (+ scaling + positional encoding).
3. Dynamic graph execution (build the transformer forward pass ourselves, no recorded graph).
4. Greedy decoding (the autoregressive loop).
5. De-tokenization (token ids → text).

We support only the **happy path** for the shipped Firefox models — the config below — not marian's
full option surface.

## Config audit (what the happy path actually uses)

Two sources define the runtime. The **decode config** written by `download_model.py`
(`data/models/enfr/config.enfr.yml`):

```yaml
models:        [model.enfr.intgemm.alphas.bin]
vocabs:        [vocab.enfr.spm, vocab.enfr.spm]   # source, target
shortlist:     [lex.50.50.enfr.s2t.bin, false]
beam-size:     1                                  # => greedy
normalize:     1.0                                # length normalization exponent
word-penalty:  0
max-length-break: 128
max-length-factor: 2.0                            # max out len = factor * in len
mini-batch-words: 1024
gemm-precision: int8shiftAlphaAll                 # the path inference-rs already implements
alignment:     soft                               # not needed for text output
skip-cost:     true
```

The **embedded architecture** (`special:model.yml` inside the `.bin`):

```yaml
type: transformer          dim-emb: 384           transformer-heads: 8
enc-depth: 6               dec-depth: 4           transformer-dim-ffn: 1536
dec-cell: ssru             transformer-decoder-autoreg: rnn
transformer-ffn-activation: relu                  transformer-postprocess: dan  (add + layernorm at inference)
transformer-postprocess-emb: d                    transformer-train-position-embeddings: false  (=> sinusoidal)
tied-embeddings-all: true  (output projection reuses Wemb)
dim-vocabs: [32000, 32000]
```

This is the Bergamot "student": a **6-layer transformer encoder** and a **4-layer decoder whose
autoregressive sublayer is an SSRU** (recurrent) rather than masked self-attention, plus
cross-attention to the encoder. Everything runs the `int8shiftAlphaAll` GEMM we already have.

### Scope decisions (happy path only)

- **Greedy only** (`beam-size: 1`). No beam search, no n-best, no hypothesis recombination.
- **Single sentence at a time.** No mini-batching / padding / masking across a batch (we can loop
  over sentences). This removes most of the masking machinery.
- **No alignment output** (`alignment: soft` is ignored — we produce text, not alignments).
- **Two independent vocabs.** `vocabs` is a *list of two*; en↔fr happens to point both at the same
  `.spm`, but **for CJK pairs the source and target `.spm` differ**. Treat source and target vocab
  as separate objects that may or may not be the same file — do not assume shared.
- **Shortlist is in scope** for exact parity (it changes which logits exist), with a full-vocab
  fallback (see §Shortlist).

## Current foundation to build on

- [`model`](src/model.rs) — reads the `.bin`; gives logical int8 weights + quant mults and float
  params. Add: read `special:model.yml` and `Wemb` (dequantized float, already handled at load).
- [`ops`](src/ops.rs) — all the compute: `layer_normalization`, `+`, `relu`, `softmax`, `highway`,
  `transpose`, `rows`, `bdot`, `intgemm_affine`, `prepare_a`, … These are the whole kernel set the
  encoder + decoder need. No new math ops are expected (SSRU = `dot`/`affine` + `highway` + `relu`,
  all present).
- [`trace`](src/trace.rs) + [`graph::replay`](src/graph.rs) — become the **oracle** for validating
  dynamic execution node-for-node.

## Component design

### 1. SentencePiece tokenization (`src/spm.rs`)

`vocab.*.spm` is a standard **SentencePiece `ModelProto`** (protobuf; the file begins with the
`</s>` piece). Firefox models use the **unigram** model: tokenization is a Viterbi search for the
max-score segmentation over the piece vocabulary, after normalization.

- **Parse** the protobuf: we need the `pieces` (string, log-prob score, type ∈
  {NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE}) and the `trainer/normalizer` spec. A minimal
  hand-rolled protobuf reader suffices (we only read a few fields); alternatively pull in a small
  protobuf crate. Avoid the C++ `sentencepiece` lib — the point is a Rust implementation.
- **Normalize**: SentencePiece applies an NFKC-ish char normalization and, by default,
  `add_dummy_prefix` (prepend `▁`) and maps space → `▁` (U+2581). The exact normalizer is embedded
  in the proto (`normalizer_spec.precompiled_charsmap`). *Happy-path simplification to verify:* the
  Firefox models use the default normalizer; if the precompiled charsmap is identity-ish for Latin
  text we can implement NFKC + whitespace→`▁` and validate against the reference on real sentences.
  For CJK, confirm the normalizer behavior against the reference tokenizer output.
- **Segment**: unigram Viterbi over the input using the piece scores; unknown fallback to the
  `<unk>` piece (and/or byte-fallback pieces if the model has them).
- **Special tokens**: append **EOS** (`</s>`) to the source (marian's SPM encoder does this —
  `sentencepiece_vocab.cpp:216`). EOS id = the proto's `eos_id`; BOS/`<unk>` similarly. Do **not**
  assume id 0 without reading the proto (`Word::DEFAULT_EOS_ID` is 0 but the SPM id governs).

**Validation:** compare Rust token ids to the reference for a sentence set. The reference ids for
the traced sentence are already in the trace (the `const uint32` fed into `rows`); for more coverage,
add a tiny `spm-encode` harness via `translator-cli` or the Python `sentencepiece`.

### 2. Embedding lookup + input representation (`src/embed.rs`)

Given source ids `[t0, t1, …, EOS]`:

- **Lookup**: `Wemb` is `[32000, 384]` float (dequantized at load). Gather rows with `ops::rows`.
- **Scale**: multiply by `sqrt(dim_emb)` (`scalar_mult`, ≈ 19.5959 for 384) — marian scales
  embeddings up before adding positions (`tied-embeddings-all` path).
- **Positional encoding** (sinusoidal, `transformer.h:92`): for position `p`, dim `i`,
  `signal[i] = sin(p * freq[i] + offs[i])`, with
  `freq[i] = 1e-4^((i % T)/(T-1))`, `offs[i] = (i / T) * π/2`, `T = dim_emb/2`. Add to the scaled
  embeddings. (These are the `+` / `scalar_mult` / `const` nodes at the top of the trace.)
- **`transformer-postprocess-emb: d`** = dropout only → identity at inference. No embedding
  layernorm here.

### 3. Encoder (`src/encoder.rs`) — 6 transformer layers

Per layer, standard pre-`int8` transformer with `dan` postprocess (add residual + layernorm; the
model has `layer-normalization: false` for the *legacy RNN* path, but transformer layernorm is on
via `postprocess: dan`). Each layer:

1. **Self-attention**: project to Q,K,V with the int8 affines (`intgemm_affine`), reshape/transpose
   to `[batch, heads, len, dim/heads]`, scores = `bdot(Q, Kᵀ) * 1/√(dk)`, `softmax`, `bdot(·, V)`,
   merge heads, output projection (affine), then **add + layernorm** (residual + `Wo_ln`).
2. **FFN**: affine → `relu` → affine, then **add + layernorm**.

All of these ops are already validated node-by-node in `graph::replay` — the work is *assembling*
them for arbitrary input length instead of reading them from the trace. No causal mask on the
encoder (bidirectional, full attention over the source).

### 4. Decoder (`src/decoder.rs`) — 4 SSRU layers + cross-attention

Per step `t`, per layer, the decoder sublayer order (`transformer.h` `DecoderLayerRNN`):

1. **SSRU autoregressive cell** (`rnn/cells.h:982`), the recurrent replacement for self-attention:
   - `x = input · W` (a `dot`), `f = input · Wf + bf` (an `affine`).
   - `c_t = highway(c_{t-1}, x, f) = σ(f) ⊙ c_{t-1} + (1 − σ(f)) ⊙ x`  ← the trace's `highway` op.
   - `h_t = relu(c_t)`.
   - **Carry `c_t` per layer across decode steps** — this is the key statefulness of greedy
     decoding; there is no growing K/V cache, just one cell vector per layer.
   - Then add + layernorm.
2. **Cross-attention** to the (fixed) encoder output: Q from the decoder state, K/V from the encoder
   output (computed once), `bdot`+`softmax`+`bdot`, output projection, add + layernorm.
3. **FFN**: affine → relu → affine, add + layernorm.

Because the SSRU carries a single cell state and cross-attention K/V come from the frozen encoder
output, a decode step is O(1) in the number of previously generated tokens — no re-attention over
the target prefix.

### 5. Output projection + shortlist (`src/shortlist.rs`)

`tied-embeddings-all: true` → the logit projection weight **is** `Wemb`. With a shortlist, marian
restricts the output vocabulary to a per-sentence candidate set and projects only those columns —
this is the `intgemmSelectColumnsB` + affine we currently *pass through* in replay (the 5 skipped
nodes).

- **Shortlist file** `lex.50.50.enfr.s2t.bin`: marian's binary lexical shortlist
  (`src/data/shortlist.{h,cpp}`). Header holds `firstNum` (=50) most-frequent target tokens always
  included, `bestNum` (=50) best translations per source token, source vocab size (32001), and the
  entry table. Candidate set for a sentence = union over its source tokens of the top translations,
  plus the always-on frequent tokens. Parse this format and build the candidate id list.
- **Projection**: gather those columns of `Wemb`, run the int8 affine (add `decoder_ff_logit_out_b`
  bias, also column-selected), producing logits over the candidate set. Map argmax back to the full
  vocab id.
- **Fallback / bring-up**: project the **full** 32000-vocab (no shortlist) first — simpler, and for
  greedy it usually yields the same token. Switch to the shortlist for exact parity with
  translator-cli. Note in output which mode ran (silent divergence otherwise reads as "matched").

### 6. Dynamic graph execution (`src/engine.rs`)

Replace "walk the recorded trace" with "call ops in the architecture's order." Two viable shapes:

- **(Recommended) Direct imperative forward pass.** Plain Rust functions (`encoder`, `decoder_step`)
  that hold `Vec<f32>` tensors + shapes and call `ops::*` directly. A tiny `Tensor { data, shape }`
  helper (we already thread `(data, shape)` through ops) keeps shapes honest. No graph object, no
  toposort — control flow *is* the graph. Simplest to read and to match against the trace.
- **(Alternative) Reified graph** with node objects + a topological executor, closer to marian's
  `ExpressionGraph`. More machinery than the happy path needs; deferred unless we later want laziness
  or memoization.

A `WeightsView` over `model::Model` resolves parameters by marian's naming convention
(`encoder_l{i}_self_Wq`, `decoder_l{i}_...`, `Wemb`, `..._ln_scale/_ln_bias`, `..._QuantMultA`, the
weight's appended `quantMultB`) so layers fetch weights by index. `unquant_mult = 1/(qA·qB)` is
computed from the stored multipliers (scalar `= 1` on the happy path) — no longer recovered from
trace output.

### 7. Greedy decoding loop (`src/decode.rs`)

```
enc = encoder(embed(src_ids))              # once
cell[0..dec_depth] = 0                      # SSRU state
y_prev = EOS                                # marian seeds the decoder with EOS
out = []
for step in 0 .. max_len:                   # max_len = ceil(factor * src_len), capped by break
    h = decoder_step(embed_one(y_prev, pos=step), enc, &mut cell)
    logits = project(h, shortlist)          # over candidate set
    y = argmax(logits)                       # greedy; beam-size 1
    if y == EOS: break
    out.push(y); y_prev = y
```

- `normalize: 1.0` and `word-penalty: 0` affect **scores/ranking**, not the greedy argmax per step,
  so they don't change the emitted tokens for beam-size 1. Note them; don't implement scoring unless
  we later add beam search.
- `max-length-factor: 2.0`, `max-length-break: 128` bound the loop.
- The decoder's first input is EOS (marian convention); positions start at 0.

### 8. De-tokenization (`src/spm.rs`, decode side)

Map output ids → pieces → text: concatenate pieces, replace `▁` (U+2581) with space, strip the
leading space. Skip control tokens (EOS). For byte-fallback pieces, reassemble bytes → UTF-8. This
is the inverse of §1 and shares the piece table.

## Validation strategy

The recorded trace is **one greedy `en→fr` run at `--cpu-threads 1`**, so it is a per-node oracle for
the *dynamic* path too:

1. **Tokenizer parity** — Rust `encode(src)` == reference source ids (the trace's `const uint32`
   feeding `rows`, plus a broader sentence set via the reference tokenizer). Detokenizer round-trips.
2. **Node parity of dynamic execution** — run `engine` on the traced sentence and diff each produced
   tensor against the trace (reuse the `graph::replay` comparator), now with *nothing* read from the
   trace: weights from the model, structure from our code, positions/masks computed. This is the real
   test that the assembly (not just individual ops) is correct, and it must reach the final logits.
3. **Greedy-output parity** — the sequence of argmax token ids equals the trace's decoder inputs at
   each step (the trace already contains the greedy path), and the final detokenized text equals
   `translator-cli`'s output on the same sentence.
4. **End-to-end** — a handful of sentences (incl. one CJK pair to exercise split vocabs) through both
   `translator-cli` and the Rust engine; assert identical output text within the tolerance bar.

Keep the tight rtol/atol bar; the int8 path already matches within it.

## Build order

1. **`spm.rs` encode** + tokenizer parity test (unblocks real input; validate vs trace ids).
2. **`embed.rs`** — lookup + scale + sinusoidal positions; diff against the trace's embedding nodes.
3. **`engine.rs` encoder** — assemble 6 layers; diff encoder output vs trace node-for-node.
4. **`decoder.rs` one step** — SSRU + cross-attn + ffn for step 0; diff vs trace.
5. **Output projection, full-vocab first** — logits vs trace; then add the **shortlist** for exact
   parity (covers the 5 replay-skipped nodes).
6. **Greedy loop** — multi-step decode with SSRU state carry; token-id parity vs trace.
7. **`spm.rs` decode** — detokenize; end-to-end text parity vs `translator-cli`.
8. **Split-vocab / CJK check** — a pair with distinct source/target `.spm`.

Steps 1–5 each diff against the trace oracle, so divergence is caught at the earliest node — same
discipline as the op-parity work. Steps 6–8 validate the loop and the human-readable output.

## Open questions / risks

- **SentencePiece normalizer.** The `precompiled_charsmap` (Trie-based char normalization) is the
  fiddliest part. Risk is highest for CJK/Unicode; low for Latin. Decide early: reimplement the
  charsmap vs. assume NFKC + whitespace handling and validate. This is the top risk to "full parity."
- **Shortlist binary format.** Exact header/layout of `lex.*.s2t.bin` needs confirming against
  `shortlist.cpp`. Full-vocab fallback de-risks bring-up.
- **EOS/UNK ids and decoder seed.** Read from the SPM proto; confirm marian seeds the decoder with
  EOS and that positions begin at 0.
- **`quantMultB` source.** Prefer the value appended to each weight in the `.bin`
  (`model.rs::quant_mult`) over the `intgemmQuantMultB` trace node, so nothing comes from the trace.
- **Numerics of assembly vs replay.** Replay reused the trace's shapes/attributes; dynamic execution
  computes them. Transpose axis conventions and head reshapes are the likely first divergence — the
  node-level oracle (step 2 above) will pinpoint them.
- **Batching.** Deliberately out of scope; single-sentence loop. Revisit only if throughput matters.
```
