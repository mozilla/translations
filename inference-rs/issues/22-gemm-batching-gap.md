# Close the GEMM-batching gap vs marian (fewer, larger, less-wasted matmuls)

**Open, scoped follow-on.** After the gemmology kernel swap, inference-rs is GEMM-bound on the
*same* i8mm kernel as marian yet still runs at **0.50× marian on en→fr** and **0.36× on en→ru
base** (~2× and ~2.8× slower). This issue records what that gap is and the concrete levers.
Sharpens the "remaining ~2× vs marian" section of [../08-perf-analysis.md](../08-perf-analysis.md)
and the throughput discussion in [../09-final-comparison.md](../09-final-comparison.md).

## It is not kernel speed — it is issued work × shape

Post-swap profiles put both engines with the large majority of self-time inside the **identical**
routine (`gemmology::Shift::Multiply<i8mm<neon64>>`): ~73–81% for us, ~80% for marian. Same
compiled kernel, same per-FLOP speed. So the gap is that **we issue more total kernel work, in a
less efficient shape** — not that our matmul is slower.

**Why shape matters.** For `C[m,n] = A[m,k]·W[k,n]`, the weight `W` is streamed from memory. Large
`m` → each streamed weight byte is reused across many rows → compute-bound, near peak. Small `m`
(e.g. decoding one token) → the whole weight is streamed to produce a handful of rows →
memory-bound, far below peak (low arithmetic intensity). Batching raises `m`; the win is getting
the same answer with **fewer, larger** matmuls.

## Where our `m` is small / our work is wasted

1. **Small `m` in the decoder + projection (inherent to block-shaped work).** The decoder is
   autoregressive — one pass per output token — so `m` = active sentences in the block, and our
   blocks average ~3.3 sentences (en→fr). The output projection (`k=512`, `n=32000`; 61% of
   self-time pre-swap) runs at that tiny `m` **every step**, streaming the ~16 MB packed `Wemb` to
   produce a few rows of logits. 08 already notes batched projection "pays off more on larger
   blocks" — the same statement inverted.

2. **We don't retire finished sentences from the decoder body (our own inefficiency).** In
   `greedy_batch` (`engine.rs:546`) the loop runs `0..cap` where `cap` = the *longest* sentence's
   length, stopping only when *all* are done. `decode_step_batch` (`engine.rs:472`) runs every
   affine at `m = batch` — the SSRU, cross-attention, and FFN recompute **all** rows, including
   sentences that already emitted EOS, until the longest finishes. Only the projection
   (`step_select`) gathers active rows. A block with one 40-token and three 10-token sentences
   costs 4×40 = 160 decoder row-steps vs the necessary 40+10+10+10 = 70 (>2× waste). **marian
   shrinks its batch as sentences complete, so it doesn't pay this.** This directly inflates the
   kernel work we issue — consistent with "same kernel, run longer." Most actionable lever.

3. **Per-call overhead (smaller).** Each affine quantizes its activation (`prepare_a`) and the
   shim memcpys A into an aligned buffer before the multiply; that fixed cost is amortized over
   few rows when `m` is small.

**Why the base model is worse (0.36× vs 0.50×) — interpretation, not isolated.** The base model
is larger, so the projection that dominates is a bigger matmul run at the same small `m`; and
en→ru decodes are longer and more length-variable, enlarging the #2 waste. Both push the same
direction, matching the measured ratio.

## Measured vs hypothesis

- **Measured:** the throughput ratios; both GEMM-bound on the identical kernel post-swap; batched
  projection helped only modestly at small block sizes.
- **Not yet isolated:** the split between small-`m` intensity, finished-sentence waste, and
  per-call overhead. 08 frames the remainder as "per-call overhead rather than a single hot spot"
  and lists candidate levers.

## Plan

1. **Measure the decomposition first.** Compare total kernel self-time in *seconds* (not %)
   between the two profiles — tells us if we issue more FLOPs or run them less efficiently.
   Histogram our GEMM `(m,k,n)` shapes (how small is `m`, really). Instrument the ratio of
   `batch·cap` to Σ(true sentence lengths) to size the #2 waste directly.
2. **Fix #2 (retire finished sentences).** Shrink the active set in the decoder body, not just the
   projection — compact `prev`/`cells`/context to active rows each step (or skip done rows), so
   affines run at `m = active`, matching marian. Small change, likely a real win, directly
   comparable to marian's behavior. Re-profile; expect `gemmology::Shift::Multiply` seconds to
   drop and the words/s ratio to move.
3. **Then reassess #1/#3.** Larger effective batches (cross-block batching by word budget, like
   marian's `mini-batch-words`) and trimming per-call overhead ([21](./21-activation-scratch-pool.md)
   removes the activation-copy part). Cross-block batching departs from the strict per-paragraph
   production shape, so weigh it against the block-bench fairness basis.

Correctness gate throughout: token-identical output + batch-invariance + oracle parity (a pure
perf change must not move a logit), per the tests already guarding the batched path.

## Update — cross-attention K/V caching (commit `df600c01`): 0.36× → 0.89× marian

Step 1 (measure first) found a redundant GEMM bigger than any of the three levers above: the
batched decoder recomputed the **cross-attention K and V projections over the whole encoder
context (`batch·seq` rows) on every decode step**, though the context is fixed for the decode.
dhat showed two ~7 GB churn call-sites; the profile's decode time was dominated by it.

Fix (`Engine::cross_attn_kv`, splitting `multihead_batched` into `project_kv` + `attend_batched`):
project K/V once per block and reuse across steps. Pure memoization — **token-identical**
(13,192 tokens unchanged; batch-invariance + reference parity green). Measured on the
`final_comparison.py` harness (en→ru base, Frankenstein):

| metric | before | after |
|---|--:|--:|
| decode (s) | 15.4 | **4.1** |
| words/s | 467 | **1177** |
| vs marian | 0.36× | **0.89×** |
| dhat churn | 22.66 GB | 8.15 GB (with issue 21 elims) |

This was effectively lever "issue fewer/less GEMM work" in its most extreme form (recomputing a
constant). **The gap is now ~11%.** Remaining, lower-priority levers from the analysis above:
- **§2 finished-sentence retirement** — still open. After K/V caching the decoder body (SSRU/FFN/Q
  affines) still runs at full batch width for `cap` steps; retiring completed rows would shrink `m`.
  Marginal now that the gap is 11%, moderate risk (gather/scatter + cross-KV indexing), so it wants
  the length-varied parity test noted below before landing.
- **§1 small-`m` / §3 per-call overhead** — the residual ~11%; needs the seconds-level kernel
  self-time comparison to attribute.
