# Decoder throughput & allocation pass: 0.36× → 0.96× marian

This document records an optimization pass on the inference-rs decoder and its results, written for
review. It covers **what** was changed, **why** (grounded in profiling), the **design decisions and
trade-offs**, how **correctness** was preserved, and what was deliberately **not** done. It follows
[08-perf-analysis.md](./08-perf-analysis.md) (the kernel-swap profile) and
[09-final-comparison.md](./09-final-comparison.md) (the three-way benchmark), and closes the levers
scoped in [issue 22](./issues/22-gemm-batching-gap.md) (throughput) and
[issue 21](./issues/21-activation-scratch-pool.md) (allocation churn).

## TL;DR

On the canonical benchmark (en→ru **base** model, Firefox's Frankenstein block corpus, single
thread, shortlist off, Apple M-series), measured through the project's own harness
`scripts/final_comparison.py`:

| engine | words/s | translate (s) | settled RSS (MiB) | peak RSS (MiB) |
|---|---:|---:|---:|---:|
| **inference-rs** (before) | 467 | 20.4 | 251 | 256 |
| **inference-rs** (after) | **1274** | **7.5** | **249** | **249** |
| marian `block-bench` (native) | 1327 | 7.2 | 298 | 298 |
| Firefox Wasm (Full-Page) | 419 | 22.9 | 355 | 355 |

- **Throughput: 0.36× → 0.96× native marian** (2.7×), and **1.11× → 3.04× the shipping Firefox Wasm
  path**. inference-rs is now within ~4% of native marian.
- **Memory unchanged at 249 MiB settled** — still the lightest of the three (16% under marian, 30%
  under Firefox's inference process). See [§5](#5-decision-why-we-did-not-build-the-scratch-pool).
- **Every change is token-identical** — 13,192 output tokens unchanged across the whole pass;
  verified against the marian oracle transitively (see [§6](#6-correctness-methodology)).
- **Allocation churn 22.66 GB → 8.15 GB, allocations 2.63M → 732K** (dhat).

Commits (oldest→newest): `df600c01`, `dabfc419`, `ef93017c`, `c4894f46`.

## 1. Measurement infrastructure (how the numbers were produced)

All results are grounded in existing project tooling so a reviewer can reproduce them:

- **Definitive three-way throughput + memory:** `scripts/final_comparison.py en ru`. Rebuilds the
  release binary (default fast config `lean-embed,gemmology`), runs inference-rs and marian
  `block-bench` on the identical block corpus, samples each process's RSS every 20 ms, and reports
  `words/s = source_words / Σ per-block compute`, `settled` (median RSS over the run's second half),
  and `peak`. Median of 4 timed runs + 1 warmup. Firefox's column is baked in from its perftest.
- **Per-commit deltas (throughput):** a small repeatable driver over the release binary's
  `translate --blocks --timing` JSON (`[block]` spans → Σ encode+decode seconds → words/s), median
  of 4. Used for fast iteration between the heavier three-way runs; its absolute numbers track the
  harness within run-to-run noise.
- **Allocation churn:** `--features dhat-heap` build + `DHAT_OUT=…`, reporting dhat's total bytes /
  blocks (churn), t-gmax (peak live heap), and the per-call-site histogram used to prioritize.
- **Peak RSS spot checks:** `/usr/bin/time -l`.

The benchmark model is the **base** architecture (dim-emb 512, ffn 2048, enc-depth 6, dec-depth 2
SSRU, 8 heads, tied embeddings, vocab 32000), i.e. the larger of the shipped models — the harder
case for both throughput and memory.

## 2. Baseline and where the time went

Before this pass: **467 words/s, 0.36× marian**, with decode dominating (≈15.4 s of the ≈20 s
compute; encode ≈3.9 s). [08](./08-perf-analysis.md) had already established that after the
gemmology kernel swap both engines spend ~73–81% of self-time in the **identical** i8mm GEMM kernel
— so the gap was never kernel speed, it was **how much GEMM work we issue**. This pass profiled the
decoder with dhat to find that work.

The dhat per-call-site histogram (baseline) showed two things:

1. **Two ~7 GB churn sites in `PreparedB::matmul_into`** — the cross-attention **K and V
   projections**, each recomputed over the whole encoder context every decode step.
2. **`softmax` at 1.56M of 2.63M allocations** — a per-`(batch, head, query)` `Vec`.

The first is redundant *compute* (the payoff below); the second is redundant *allocation*.

## 3. Optimization 1 — cache cross-attention K/V (`df600c01`)

**Observation.** In the batched decoder, cross-attention projects K and V from the encoder context
(`kv_in = ctx.data`, `batch·seq` rows). The context is fixed for the entire decode, so `K = Wk·ctx`
and `V = Wv·ctx` are **constant across all decode steps** — yet they were recomputed every step. For
the base model (dec-depth 2, hundreds of steps per block) this is the single largest GEMM in the
decoder, done redundantly tens of times over.

**Change.** Split `multihead_batched` into `project_kv` (computes K/V) + `attend_batched` (Q + the
scaled-dot-product + output projection). Project K/V **once per block** in `Engine::cross_attn_kv`
(indexed by decoder layer) and reuse across steps; the per-step path only projects Q. The encoder's
self-attention still projects K/V per layer (its `kv_in` changes each layer — not cacheable), so the
split is a no-op there.

**Design decision — memoization, not approximation.** This changes *when* K/V are computed, not
*what* is computed: bit-for-bit the same values. So it is safe by construction, independent of
numerics. We also moved the one-time projection *inside* the decode-timing window in the timed path
(`translate_batch_timed`), so words/s is not flattered by moving work out of the measured region — it
genuinely replaces the per-step K/V that used to be counted as decode.

**Result (per-commit driver):** decode **15.4 s → 4.1 s**, **467 → 1177 words/s** (0.36× → 0.89×
marian via the harness). dhat churn **22.66 GB → 8.87 GB** (the two ~7 GB sites gone). Token-identical.

## 4. Optimization 2 — retire finished sentences from the decoder body (`c4894f46`)

**Observation.** A block is decoded in lockstep for `cap = max sentence length` steps, stopping only
when *all* sentences finish. But `decode_step_batch` ran every affine at `m = batch` for all `cap`
steps — recomputing rows whose sentence had already emitted EOS. Only the output projection
(`step_select`) gathered the still-active rows; the decoder body (SSRU, cross-attention Q/O, FFN) did
not. We **measured the waste directly** by instrumenting the loop: **44.2%** of decoder-body
row-steps were spent on already-finished sentences.

**Change.** Compute the active row set each step (`active_rows`: rows not done and within their
length cap) and run the decoder body over just those rows at `m = active.len()`:
- `decode_step_batch(active, …)` compacts the activation `x` to `[active, dim]`;
- the SSRU cell state stays full-batch and is **gathered → highway/ReLU → scattered** back per active
  row (indexed by original batch id, so finished rows' state is simply frozen);
- cross-attention uses a new `attend_cross`, which is `attend_batched` with one query per active row
  but indexes the **full-batch cached K/V by original id** `active[i]` (masked to that sentence's
  source length) — so we neither recompute nor re-pack K/V for the shrinking active set;
- `step_select` becomes `select_active` over the already-compacted tops.

**Design decisions / trade-offs.**
- *Full-batch K/V + index map, not compacted K/V.* Compacting the cached K/V to the active set each
  step would re-copy `active·seq·dim` floats per step — reintroducing the very traffic we removed in
  §3. Passing an `active` index map into `attend_cross` avoids that at the cost of one small
  gather/scatter of the cell state per layer (`active·dim`, negligible).
- *Monotonic active set.* Sentences never un-finish, so the active set only shrinks; a row past its
  length cap is simply never in `active` again — no explicit "done" bookkeeping is needed for
  termination (the loop stops when `active` is empty), which keeps `active_rows` a pure function.
- *This is the riskiest change* (gather/scatter + cross-KV indexing must line up), so it was landed
  last, behind the strongest correctness gate we have (see §6), and only after the 44% waste
  measurement justified the risk.

**Result (per-commit driver):** decode **4.1 s → 3.4 s**, **1177 → 1300 words/s**. Token-identical.

## 4b. Allocation-churn eliminations (`dabfc419`, `ef93017c`)

Issue 21's count-dominant sites, all bit-identical:
- **`dabfc419` softmax in place** — `ops::softmax_in_place` softmaxes the reused `scores` buffer
  directly instead of returning a fresh `Vec` per `(batch, head, query)`. Removed ~1.56M allocations.
- **`ef93017c` embed into destination + cached layer-norm params** — `embed_into` /
  `*_embed_row_into` / `dequant_row_into` write the embedding row + `√d` scale + positional encoding
  straight into the caller's activation slice (no per-token `Vec`, no copy); layer-norm scale/bias
  are decoded once at load (`Weights::layer_norm`, ~72 KB) instead of per `postnorm` call.

**Result:** allocations **2.63M → 732K**, churn to **8.15 GB**. Speed effect is within noise (these
are byte-trivial or count-only), which is expected — see §5.

## 5. The memory follow-ups: jemalloc is the lever, the scratch pool is churn-only

Two memory follow-ups were sequenced and measured (commits `9a35c04e`, `15068183`). The result is
clean-cut: **the allocator, not our allocation pattern, gates settled RSS.**

**jemalloc (opt-in `jemalloc` feature) — the real memory win.** Swapping in a page-returning
allocator (`tikv-jemallocator`), settled RSS on the same benchmark:

| allocator | settled MiB | peak MiB | words/s |
|---|--:|--:|--:|
| libmalloc (default) | 248.7 | 248.7 | 1283 |
| jemalloc (default decay) | **145.9** | 165.3 | 1271 |
| jemalloc `dirty/muzzy_decay_ms:0` | **124.7** | 146.4 | 1221 |

**−103 MiB at ~0% throughput cost**, or **−124 MiB at ~5%**. 124.7 MiB ≈ 86 (dhat t-gmax live Rust
heap) + 41 (gemmology prepared-B) — jemalloc reaches the **live-memory floor**, directly confirming
[issue 19](./issues/19-settled-rss-allocator.md)'s hypothesis that the retention was libmalloc's, not
ours. This is the memory lever; a candidate to make default (or ship with a tuned `MALLOC_CONF`).

**The scratch pool (issue 21) — built, and it is churn-only.** A capacity-keyed `FxHashMap`
free-list (`src/pool.rs`) with `Drop`-returning `Buf` leases, cleared per block; `Weights::affine`
and the forward draw activation buffers from it (a ring/pool, *not* a bump allocator — see issue 21
for why bump is the worse fit). Bit-identical. Measured:

| metric | before | after |
|---|--:|--:|
| dhat churn | 8.15 GB | **3.92 GB** |
| dhat t-gmax | 86 MB | 86 MB |
| **settled RSS** | 249 MiB | **249 MiB** |
| words/s | 1300 | ~1295 |

So the pool **halves churn but moves neither settled RSS nor throughput** — exactly as predicted: it
reduces allocator *traffic*, but the retained *footprint* is what libmalloc holds (and jemalloc
returns). Kept because it is metric-neutral, but it is not a memory lever. One caveat worth its own
line: an *unbounded* pool (no per-block clear) instead **raised t-gmax to 226 MB** by hoarding
buffers across the widely-varying activation sizes (encoder `batch·seq`; decoder `m` shrinks as
sentences retire) — a concrete "pool everything" failure mode; the per-block `Pool::clear()` fixes it.

**Bottom line:** for memory, adopt jemalloc; the pool is optional cleanliness.
This is the key judgement call in the pass: **we optimized to the metrics that move (redundant GEMM),
not to the churn number for its own sake.**

## 6. Correctness methodology

The engine is validated against the marian C++ reference by a layered gate; every commit in this pass
was required to pass **all** of it, and additionally to leave the benchmark output byte-for-byte
unchanged:

1. **`tests/gemm_parity.rs`** — the gemmology i8mm kernel is bit-close to the scalar `intgemm_affine`,
   which is itself validated against the marian oracle. Kernel numerics are pinned.
2. **`tests/ops_parity.rs`, `tests/int8_parity.rs`, `tests/real_trace.rs`** — each op matches tensors
   recorded from the reference engine.
3. **`tests/batched_decode.rs` batch-invariance** — `greedy_batch` must produce, per sentence,
   exactly what single-sentence `greedy` produces. **This is the critical guard for §3 and §4:** the
   test block deliberately mixes sentence lengths, so sentences finish at different steps — a
   K/V-caching or retirement indexing bug shows up as a per-sentence divergence against the
   oracle-validated single path, not against our own batched output (cheat-proof).
4. **`tests/translate.rs::matches_reference_translations`** — end-to-end strings match known-good
   reference translations; `mmap_matches_owned` pins the mmap path.
5. **Output hash** — the full 103-block Frankenstein translation hashed to the same SHA
   (`2fbf9ee…`) before and after every commit, and the decoded token count stayed 13,192.

Because §3 and §4 are memoization / dead-work elimination (not numeric changes), passing (3)+(5) is
strong evidence of correctness, and both were run on every commit. Full suite: 44 unit + all
integration tests green; `cargo fmt --check` clean.

## 7. Results in detail

Per-stage progression (throughput via the per-commit driver; churn via dhat; both on en→ru base /
Frankenstein):

| stage | commit | words/s | decode (s) | churn | allocations |
|---|---|---:|---:|---:|---:|
| baseline | `df600c01^` | 492 | 15.4 | 22.66 GB | 2,631,507 |
| + K/V cache | `df600c01` | 1178 | 4.1 | 8.87 GB | 2,537,436 |
| + softmax in place | `dabfc419` | — | — | ~8.5 GB | ~1.0M |
| + embed/LN elimination | `ef93017c` | — | — | 8.15 GB | 732,175 |
| + finished-sentence retirement | `c4894f46` | 1300 | 3.4 | 8.15 GB | 732K + small gather/scatter |

Definitive three-way (harness `final_comparison.py`, final state) is the TL;DR table above:
**1274 words/s, 0.96× marian, 3.04× Firefox, 249 MiB settled.**

**Interpretation of the remaining ~4% vs marian.** Both engines are GEMM-bound on the same kernel;
the residual is small-`m` arithmetic intensity (the per-block batch averages ~4 sentences, so the
output projection and decoder affines run at small `m`) plus per-call overhead (activation quantize +
aligned copy). This is *interpretation*, not yet isolated — attributing it needs the seconds-level
kernel self-time comparison noted in issue 22 §1/§3. Closing it (cross-block batching by word budget,
per-call trimming) trades against the strict per-paragraph production shape, so it is deprioritized at
0.96×.

## 8. Reproduction

```bash
# Definitive three-way (rebuilds release, samples RSS):
python3 inference-rs/scripts/final_comparison.py en ru

# Allocation churn + per-call-site histogram:
cargo build --release --features dhat-heap --manifest-path inference-rs/Cargo.toml
DHAT_OUT="$PWD/inference-rs/artifacts/dhat.json" \
  inference-rs/target/release/inference-rs translate \
  data/models/enru/model.enru.intgemm.alphas.bin data/models/enru/vocab.enru.spm \
  data/models/enru/vocab.enru.spm --blocks inference-rs/corpora/frankenstein-en.blocks.txt >/dev/null

# Correctness gate:
cargo test --features instrumentation --manifest-path inference-rs/Cargo.toml
```

Model/corpus provenance and the config template are in [09](./09-final-comparison.md#reproducing).

## 9. Code map

| change | primary symbols | files |
|---|---|---|
| K/V caching | `cross_attn_kv`, `project_kv`, `attend_batched` | `src/engine.rs` |
| finished-sentence retirement | `active_rows`, `decode_step_batch(active,…)`, `attend_cross`, `select_active` | `src/engine.rs` |
| softmax in place | `ops::softmax_in_place` | `src/ops.rs`, `src/engine.rs` |
| embed into dest | `embed_into`, `Weights::{src,trg}_embed_row_into`, `dequant_row_into` | `src/engine.rs`, `src/weights.rs` |
| layer-norm cache | `Weights::layer_norm` | `src/weights.rs`, `src/engine.rs` |
| jemalloc (opt-in `jemalloc` feature) | `#[global_allocator]` | `Cargo.toml`, `src/main.rs` |
| activation scratch pool | `Pool`, `Buf`; `Weights::affine`→`Buf`, forward returns `Buf` | `src/pool.rs`, `src/weights.rs`, `src/engine.rs` |

## 10. Remaining follow-ups

- **Small-`m` / per-call overhead (issue 22 §1/§3)** — the residual ~4%; measure kernel self-time in
  seconds before optimizing; cross-block batching is the lever but changes the benchmark's work shape.
- **Adopt jemalloc more widely (§5)** — the −100 MiB settled-RSS win. Decide default-vs-opt-in after a
  cross-platform check (Linux/Wasm) and pick a `MALLOC_CONF` decay that balances the ~0–5% throughput
  trade. The scratch pool does not stack on top (jemalloc already hits the live floor).
- **Single-sentence decode path** — `decode_step`/`greedy` (used only by the one-line `translate`)
  still recompute cross-attention K/V per step. The production path is batched; caching there too is
  a small, safe follow-up for consistency.
