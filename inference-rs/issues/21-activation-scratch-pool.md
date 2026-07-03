# Reuse activation buffers via an engine-owned scratch pool (kill the per-op churn)

**Open, scoped follow-on.** The eager forward allocates a fresh `Vec` for essentially every
intermediate; steady-state translation churns ~22 GB over millions of allocations (dhat, post
C1+C2+B). This issue records *why* that happens architecturally and the concrete fix. Continues
the "reduce activation copies" lever from [../08-perf-analysis.md](../08-perf-analysis.md) and the
allocator analysis in [19-settled-rss-allocator.md](./19-settled-rss-allocator.md); the memory
breakdown is in [../09-final-comparison.md](../09-final-comparison.md) ("How the memory inflates").

## Why we churn: eager forward, no arena

inference-rs has **no expression graph and no tensor arena**. `engine.rs` is an imperative,
eager forward — the Rust call stack *is* the traversal — and each op in `ops.rs` computes and
returns a fresh `Vec<f32>` that drops at end of scope. So we pay one allocator `alloc` + `free`
per op invocation.

Contrast marian, which pairs a lazy reified tape (`ExpressionGraph::nodesForward_`) with a
**single retained arena** (`TensorAllocator` → `Allocator`, a coalescing `Gap` free-list, 128 MB
`GROW` chunks; 1 MB on Wasm). During inference it frees each intermediate back into the arena as
soon as its consumer has run (`children().clear()` + `~Node → graph()->free`), so the arena grows
once to the *peak simultaneously-live* set and recycles slots across every step, sentence, and
block — steady-state does ~zero OS allocation.

| marian | inference-rs |
|---|---|
| lazy tape (reified DAG) | none — imperative call stack |
| arena + `Gap` free-list, retained | system allocator, one alloc/free per op |
| node frees back to arena when consumers done | `Drop` frees `Vec` at scope end |
| arena retained at high-water | libmalloc magazines retain freed pages |

The consequence is the ~120 MiB "allocator working set" line in the memory breakdown: libmalloc
holds freed pages resident (absent pressure), so RSS floats up to the churn high-water and stays.
Note the *live* footprint is already small (single-digit MiB) via RAII — the problem is churn and
the retention it drives, not peak live memory. **Lazy-vs-eager is orthogonal to arena-vs-malloc;
we can keep the eager forward and still get the arena, because our lifetimes are already lexical
(Drop fires at exactly the right moment — no tape needed to track them).**

## Where the 22 GB lives — two shape families dominate the bytes

Byte-dominant activation buffers, recurring every layer and every decode step:

- **`[rows × d]`** (rows = `batch·seq` encoder, `batch` per decode step; d=512): the Wq/Wk/Wv/Wo
  affine outputs (`weights.rs:524`), `joined` (`engine.rs:679`,`:753`), every `postnorm`'s
  residual `sum` (`engine.rs:819`) and `layer_normalization` out (`ops.rs:58`), SSRU `highway` /
  `c.clone()` (`engine.rs:454`,`:498`) / `relu`, and `embed` out (`engine.rs:837`,`:481`).
- **`[rows × ffn]`** (ffn=2048 = 4×d): the FFN `W1` output and its `relu` (`engine.rs:796`,
  `ops.rs:78`).

Per encoder layer ≈ 8×`[rows×d]` + 2×`[rows×ffn]`; the decoder repeats ~the same per layer **per
output token**, which is where the gigabytes accumulate. The full-vocab logits `[active×vocab]`
are already the reused `logits_scratch` (`engine.rs:40`).

Count-dominant but byte-trivial (won't move the GB, but drive alloc *count*/CPU): `softmax`
allocates a fresh `[kv_len]` `Vec` **per (batch, head, query)** (`engine.rs:771`,`:691`), and
`postnorm` re-decodes `_ln_scale`/`_ln_bias` from bytes into a new `Vec` every call
(`engine.rs:822–824`).

## Fix

Two size/frequency classes of allocation, handled two different ways. The split is not just
cosmetic: it falls out of how a bump allocator reclaims memory (see below).

### Step 0 — measure the split first

Before building anything, read the **dhat per-call-site histogram** from the run we already do
(`--features dhat-heap`, DHAT_OUT; the same run that gave 22 GB / millions of allocations). The
per-record count-and-size breakdown decides how much effort each half is worth. Working
prediction to confirm/refute: `softmax` dominates the *count*, and the `[rows×d]`/`[rows×ffn]`
pair dominates the *bytes*.

### Big, coarse-grained activations → bump/pool (attacks the bytes)

The per-layer/per-step `[rows×d]` and `[rows×ffn]` buffers. Reintroduce, by hand, what marian
gets from `TensorAllocator`: draw them from reused buffers instead of malloc/free.

- **Bump (cheapest).** One slab + an offset; `take` bumps the offset, and a reset at each barrier
  (end of encoder layer / end of decode step) frees everything since the last barrier in one
  instruction. Requires clean reset points and that no scratch outlive its barrier — so the
  encoder context, SSRU `cells`, and logits stay outside the slab. Sized to the (small, bounded)
  peak simultaneously-live set. **Needs no map.**
- **Shape-keyed pool (robust).** An engine-owned pool bucketed by capacity; `take(len)` pops a
  buffer of that class (or allocates on a cold miss) and hands back a `Lease` whose `Drop` returns
  it to the pool. `d` and `ffn` are constants and `rows` takes few values, so after the first
  block the pool is warm → steady-state zero allocation. marian's `Gap` free-list minus
  coalescing; no barriers needed because Drop timing is exact. Key it with **`FxHashMap<usize,
  Vec<Vec<f32>>>`** (`rustc-hash`) — the key is a small integer hit on every `take`, so the
  default `SipHash` is pure overhead. (We have only a few live size classes, so a tiny `Vec` of
  buckets would also do; `FxHashMap` is the general default, not worth hand-rolling first.)

Given the clean per-layer/per-step structure, bump-with-reset is the tempting default; the pool
is the fallback for buffers crossing a barrier. This extends the existing `*_into` reuse
(`prepare_a_into`, `matmul_into`, `logits_scratch`) that already took churn 29 GB → 22 GB.

### Small, high-frequency inner-loop buffers → eliminate, else stack (attacks the count)

**Why these can't go in the bump.** A bump slab only reclaims at its barrier. `softmax` is
allocated inside the `batch × head × query` triple loop (`engine.rs:771`), thousands per decode
step; in a step-granularity slab they would accumulate un-freed until reset, forcing the slab to
hold `batch·heads·q_len·kv_len` at once — defeating the point. High-frequency allocations need
per-iteration lifetime, which is exactly why they split off from the coarse pool. (That
frequency split lines up with the size split: the inner-loop buffers are the small ones.)

**Prefer "don't allocate" over "stack-allocate"** — elimination also removes repeated work:

- **`softmax` `[kv_len]` (the count hog).** Computed and immediately consumed in-loop. Give it a
  single engine-held `[kv_len]` scratch, or overwrite the already-reused `scores` in place. Zero
  alloc, no barrier concern.
- **Embedding rows `[d]`.** `dequant_row` returns a `Vec` the caller then `copy_from_slice`s into
  the batch buffer (`engine.rs:403`). Dequantize *directly into the destination slice* — the
  intermediate `Vec` disappears.
- **Layernorm `gamma`/`beta` `[d]`.** Constant params, but `postnorm` re-decodes them from bytes
  into a fresh `Vec` every call (`engine.rs:822–824`). Cache the decoded params at load (as the
  prepared affine bias already is) — kills the alloc *and* the repeated byte-decode.

**Where a genuinely-transient small buffer remains**, use `SmallVec<[f32; N]>` — inline on the
stack up to `N`, spilling to heap beyond — sized so the common case stays inline (e.g. `N = 128`
for `kv_len`). This is the fallback, not the first move; most of the inner-loop allocations above
should be eliminated outright.

## Expected payoff and caveat

Primarily a **memory-cleanliness** win: it attacks the churn that inflates the allocator working
set (issue 19's ~120 MiB), and pairs with `--mmap` / a page-returning allocator. It is **not** a
big throughput lever — allocation is low single-digit % of runtime (08 profile is ~73–81% GEMM),
so expect a small speed bump from reduced allocator/cache pressure, not a step change. The
throughput gap is [22-gemm-batching-gap.md](./22-gemm-batching-gap.md). Correctness gate: token-
identical output + bit-exact `gemm_parity` (a pure buffer-reuse change must not move a logit).
Pairs with the Rust gemmology port ([18](./18-gemmology-rust-port.md)) — one owned representation,
scratch included.

## Update — count-dominant allocations eliminated (commits `df600c01`, `dabfc419`, `ef93017c`)

Step 0 (measure) confirmed the prediction: dhat showed `softmax` at 1.56M of 2.63M allocations
(count), and `matmul_into`/`ffn`/`postnorm`/`layernorm` activation buffers dominating bytes — plus
a redundant cross-attention K/V recompute worth two ~7 GB call-sites (fixed under
[issue 22](./22-gemm-batching-gap.md), commit `df600c01`).

Done, all bit-identical (output hash unchanged; full suite green):
- **softmax in place** (`ops::softmax_in_place` over the reused `scores`) — removes the per-`(batch,
  head, query)` `Vec`.
- **embed into destination** (`embed_into` / `*_embed_row_into` / `dequant_row_into`) — writes the
  row + `√d` + PE straight into the activation buffer; no per-token `Vec`, no copy.
- **layer-norm params cached at load** (`Weights::layer_norm`, ~72 KB) — no per-`postnorm` decode.

| metric (en→ru base, Frankenstein) | before | after |
|---|--:|--:|
| dhat total churn | 22.66 GB | **8.15 GB** |
| dhat allocations | 2,631,507 | **732,175** |
| settled RSS (MiB) | 251 | 249 |

**Settled RSS did not move**, exactly as [issue 19](./19-settled-rss-allocator.md) predicted: it is
gated by libmalloc's retained working set (t-gmax ~86 MiB + ~41 MiB gemmology prepared-B), not by
churn. So the count/byte wins are cleanliness + a hair of allocator CPU, not a settled-RSS lever.

**Remaining (deprioritized): the big-activation scratch pool.** The residual 8.15 GB is now purely
per-op `[rows×d]`/`[rows×ffn]` buffers (`matmul_into` affine outputs, FFN hidden, `postnorm` sum,
`layernorm` out). The bump/pool design above would collapse it, but the measurement shows it would
move neither settled RSS (allocator-gated) nor throughput meaningfully (732K allocs ≈ tens of ms of
a 8 s run). Left as a cleanliness follow-up; revisit if paired with a page-returning allocator or
the Rust gemmology port ([18](./18-gemmology-rust-port.md)), where an owned scratch arena is natural.
