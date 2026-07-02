# Batched (minibatch) inference — plan

Add minibatch decoding to `inference-rs` so it translates a **block of sentences together**,
matching the production path, and build a faithful native perf comparison against the
marian-fork on top of it. This is both a correctness feature (production batches) and the
main perf lever (matrix×matrix reuse instead of matrix×vector per sentence).

The validation approach mirrors the original op-parity work
([01-build-plan.md](./01-build-plan.md)): the native marian-fork build is the oracle, via
its recorded internal-value trace. The one new idea is that **the oracle extends to
batching for free** — a batched marian forward records batched node tensors in the same
trace format — so we get cheat-proof validation of the batched ops without new oracle
machinery.

## 1. Why batching, and the production model it matches

Production translates a document as a sequence of **blocks** (style/DOM elements, usually a
paragraph). Per block: the block's `innerHTML` is submitted, the Wasm engine parses the
HTML, sentence-splits, and **translates the block's sentences as a minibatch** (bergamot's
`mini-batch-words`, so a paragraph's handful of sentences batch together). So the production
unit of work is *a paragraph's sentences translated in one batch* — not one sentence, not a
whole corpus.

Our engine is strictly batch=1 today (matrix×vector: each weight matrix is streamed from
memory for a single sentence). To match production — and to be comparable with marian, which
always batches — the engine must run N sentences per GEMM.

### Scope of this plan

- **In:** batched encode + decode, batched greedy with per-row EOS, cheat-proof parity vs a
  batched marian trace, and a block-level perf benchmark.
- **Out (pre-processed or deferred):** HTML parsing/tag-reinsertion and sentence splitting
  ([15-sentence-splitting.md](./issues/15-sentence-splitting.md)) — cheap relative to NN
  inference and separate features; for the perf number we feed pre-split blocks. A
  cache-blocked int8 GEMM is a **follow-on** (see §6): batching only creates the
  *opportunity* for matrix×matrix efficiency; our current naive scalar GEMM won't capture it
  without blocking, and marian uses intgemm SIMD.

## 2. Engine changes (the op surface that grows a batch dimension)

Single-sentence today → `[batch, …]` batched:

- **Embedding / encode.** Pad the block's sentences to the batch's max source length; carry a
  **source padding mask**. Encoder self-attention and the affines run over `[batch, seq, dim]`
  (GEMM `M = batch·seq`).
- **decode_step.** All active sentences step together: input embed `[batch, dim]`; SSRU cell
  state `[batch, dim]` per layer; cross-attention to each sentence's own encoder context with
  the **source mask** so padded source positions score −∞; FFN and projection `[batch, dim]`
  → logits `[batch, vocab]`; **per-row argmax**.
- **Greedy loop / EOS.** Rows finish at different steps. marian keeps the full batch and
  masks finished rows; we match that (keep-and-mask) for trace comparison, and can add
  active-set compaction later as a perf optimization (behind the same parity).
- **Length cap & ordering.** Sort sentences by length within a block (marian does, to
  minimize padding waste) — but see §3 on reproducing marian's exact layout.

New numeric surface vs batch=1: **attention masking** (self + cross), **padding**, and the
**batched affine** (`M > 1`). Each is a discrete thing to validate.

## 3. Cheat-proof validation

Same oracle, same three granularities as the original build, extended to batched tensors.
The anchor is always **native marian-fork internal values**, never our own outputs.

### The batched reference trace (the oracle)

`MARIAN_TRACE=<path>` already records *every* node's freshly-computed tensor in execution
order, keyed by id → `{type, name, shape, dtype, bytes}` — and it is **generic over shape**,
so a batched forward simply records batched tensors (with marian's sort order, padding, and
mask tensors). No recorder change is needed.

To produce a controlled batched trace: feed marian a small block of N short sentences with
`mini-batch-words` large enough that they land in **one** minibatch (default 1024 already
does), `--cpu-threads 1`, `MARIAN_TRACE` set. The trace then contains the batched encode,
the masks, and each decode step for the whole batch.

### (a) Op-level parity (unit) — the primary gate

For each batched CPU op, pull its **exact input tensors + output** from the batched trace,
feed the inputs to our batched op, assert equal within tolerance. This is **layout-agnostic
and non-tautological**: we consume marian's exact batched tensors (its padding/mask/sort
included) and must reproduce marian's exact output — the reference is marian, not us.

New fixtures beyond the batch=1 set:
- **batched affine** (`intgemmAffine` with `M = batch·seq` and `M = batch`),
- **masked softmax** (attention scores with a padding mask → the recorded masked weights),
- **layernorm / elementwise** over `[batch, …]` (broadcast correctness),
- **mask construction / additive mask** nodes, if they appear as distinct ops.

Extends `tests/ops_parity.rs`; goldens are trace-derived tensors committed under
`corpora/`/fixtures with sha256s.

### (b) Graph-level parity (integration) — the bisector

`replay` the batched trace node-by-node and report the first divergence
([graph.rs](./src/graph.rs), `compare::Tolerance`). **Action item:** extend replay's op
coverage to any *new* batched op types (masking/padding) so they are recomputed and compared
rather than passed through — otherwise batching bugs could hide in an unmodeled node. A clean
batched replay (zero divergence) is the integration proof, exactly as it was for batch=1.

### (c) End-to-end parity

Our batched greedy output vs marian's output for the same block. Compare **per sentence**
(map each result back to its source sentence), because our internal batch order/compaction
may differ from marian's — the per-sentence *tokens* must match (to the same near-tie
ceiling as [06-numeric-reduction-parity.md](./issues/06-numeric-reduction-parity.md)), not
the internal layout.

### Cheat-proof properties (explicit)

- **Oracle is external:** native marian-fork gemmology build internal values
  ([02-gemm-backends.md](./02-gemm-backends.md)), recorded once; we never assert against our
  own computation as ground truth.
- **Whole-tensor / node-by-node equality**, tight rtol/atol (reduction-order tolerant), no
  bit-exactness.
- **No tautological tests.** In particular, "translate a sentence at batch=1 vs inside a
  batch of N" is a useful differential check, but on its own it is *self-referential*. It is
  only cheat-proof when the batch=N result is also gated against the **marian** oracle — so
  the batch-invariance test asserts *batch-N == marian per-sentence* (the real gate) and
  *batch-N == our batch-1* as a cheap regression on top.
- **Batch-invariance** is the one genuinely-new invariant batching introduces, and it catches
  the characteristic batching bug — **mask leakage** (a sentence's output changing because of
  its batch-mates / padding). A sentence's decoded tokens must be independent of who it is
  batched with; padded positions must not influence real ones.
- **Held-out committed fixtures** (trace-derived golden tensors + block corpora) with
  hashes/seeds.

### On patching marian's assertions

Not required for this plan. bergamot asserts `mini-batch-words > max-length-break` (128),
which only blocks forcing batch=**1**; block-level batching uses ≥256 (or the default 1024)
and works unpatched. We would only relax that assertion if we later want a true one-off
(batch=1) marian reference without the wasm build — call that out as an explicit,
green-lit-if-needed change, not a dependency here.

## 4. Perf comparison (the point of the exercise)

- **Block corpus (no splitter needed).** The committed NLLB corpus is already **one sentence
  per line** (the sampler extracted single sentences), so a synthetic block is a *group* of
  consecutive sentences — we never sentence-split running text here. (Real splitting/HTML is
  deferred, [15-sentence-splitting.md](./issues/15-sentence-splitting.md).) Only the grouping
  must be deterministic: draw each block's sentence count with a small documented **LCG**
  seeded by a constant (a pure integer recurrence — reproducible across languages/Python
  versions, auditable in a few lines), from a bounded paragraph-like range (≈1–8, skewed
  small). The harness regroups the existing `nllb-en-fr.txt` from the seed at runtime, so no
  new artifact is committed and every run / both engines see byte-identical blocks; report
  the resulting block-size and length distribution alongside results. (Alternative if a frozen
  fixture is preferred: emit a blank-line-delimited block file + sha256.)
- **Both engines, block by block, batched, model loaded once, 1 thread, shortlist off.**
  inference-rs: batched greedy over each block. translator-cli: feed a block's sentences with
  `mini-batch-words` sized to a paragraph so it batches that block.
- **Metrics** (via the existing harness, [10-perf-harness.md](./issues/10-perf-harness.md)):
  per-block latency, sentences/s, tok/s, and TTFT (first token of the block). median + IQR
  over warmup+runs.
- **Residual asymmetry to state honestly:** translator-cli over stdin batches *across* block
  boundaries; to respect blocks we either invoke per-block (model-reload cost) or accept
  mild over-batching. Neither needs marian source changes; document whichever we pick.

## 5. Sequencing / milestones

1. **Batched encode** + op parity vs a batched trace (embedding, masked self-attn, affine
   `M=batch·seq`, layernorm).
2. **Batched decode_step** (SSRU cell `[batch,dim]`, masked cross-attn, ffn, projection) +
   op parity.
3. **Batched greedy** with per-row EOS/masking; **end-to-end per-sentence parity** vs marian.
4. **Batch-invariance** test (batch-N == marian per-sentence == our batch-1).
5. **Block corpus + block benchmark** wired into `task inference-rs:perf`.
6. **Follow-on (separate):** cache-blocked int8 GEMM to *realize* the batching speedup
   (measure before/after), then HTML handling for full-block cost.

## 6. Risks / open questions

- **GEMM quality.** Batching alone may not beat marian: our int8 GEMM is a naive scalar loop
  that re-reads the weight per output row, so it won't get matrix×matrix reuse without
  blocking. Expect the competitive number to need the §6 GEMM follow-on; the benchmark will
  quantify how much.
- **Padding waste.** Batching mixed-length sentences pads to the max; sort within a block to
  minimize (as marian does). More padding = more wasted compute, so block composition
  affects the number — report block size/length distribution alongside results.
- **EOS strategy.** keep-and-mask (matches marian, simplest for parity) vs active-set
  compaction (faster, more complex). Start with keep-and-mask under parity; compact later.
- **Trace size.** Batched tensors are larger; keep the trace fixture to a small block of
  short sentences so the committed goldens stay small.
- **Determinism.** marian sorts sentences by length inside a maxi-batch; reproduce layout by
  consuming the trace's input tensors for op parity (don't reconstruct marian's sort), and
  compare end-to-end per sentence rather than by batch position.
