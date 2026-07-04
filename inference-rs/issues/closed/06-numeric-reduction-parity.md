# Numeric Reduction-Order Parity (timeboxed) — closed, residual accepted

The near-tie greedy flips (e.g. `▁bon` vs `▁Bonjour`, a ~0.14 logit gap on a ~14 scale)
come from f32 reduction-order drift, not a bug. The int8 GEMM accumulation is integer and
therefore *exact*; the drift lives only in the float reductions around it: layernorm
mean/variance, softmax, and the attention `bdot`s (QKᵀ and scores·V).

## Approach

Replicate the reference's f32 accumulation order rather than summing sequentially. marian's
layernorm uses `#pragma omp simd reduction`, softmax is `float32x4/x8`, and `bdot` goes
through the CPU sgemm — all lane-parallel partial sums, then a combine. Match that
lane-width-partial-then-combine pattern for the reference build (NEON / gemmology, since
the oracle trace is ARM).

## Guardrails — why this is explicitly timeboxed

- f64 accumulation makes us *more accurate* but not *closer*: parity needs the reference's
  f32 rounding, not better rounding. Easy trap.
- Perfect greedy parity on every input may be unattainable without effectively vendoring
  the SIMD reductions, which couples inference-rs to one compiler/arch and undercuts the
  clean-reimplementation goal.
- So drive this by the [02-parity-harness.md](./02-parity-harness.md) flip-rate, stop at
  diminishing returns, and accept a high-but-imperfect match rate. Do **not** let parity
  turn the engine into a transliteration of the reference kernels.

## Depends on

[02-parity-harness.md](./02-parity-harness.md) — need the flip-rate metric and the per-node
bisector to target the right reductions and to know when to stop.

## Stop criterion — good enough, don't over-invest

This is a "quantify and nudge up" task, not a march to 100% (see the goal in
[02-parity-harness.md](./02-parity-harness.md)). Keep it cheap:

1. Bisect the current mismatch set → the reduction op that most often produces the first
   divergence.
2. Make that op match the reference's f32 lane order. Re-run `task rs:parity`.
3. **Stop** at the first of: a fix that doesn't move the match-rate (revert it and stop), a
   full pass under ~1 pp improvement, or ~3 fixes landed. Whatever's left is the accepted
   residual — record it and, if any is an actual bug rather than a near-tie, file a
   follow-up instead of grinding here.

## Example fix (layernorm mean)

```rust
// sequential (current) — one rounding order
let mean = row.iter().sum::<f32>() / n;

// lane-matched: 4 partial sums (NEON f32x4) then combine, mirroring the
// reference's `#pragma omp simd reduction` so f32 rounding lines up
let mut acc = [0f32; 4];
for c in row.chunks(4) { for i in 0..c.len() { acc[i] += c[i]; } }
let mean = (acc[0] + acc[1] + acc[2] + acc[3]) / n;
```

The exact lane width/combine order is arch-specific (NEON here); the fix is only "correct"
if the parity harness flip-rate moves — which is why it's measured, not asserted in
isolation (a unit test that checks our lane-sum against our lane-sum would be tautological).

## Acceptance criteria

- Corpus greedy match-rate is measured before/after and moves up (or we learn it's already
  near the near-tie ceiling) — improvement, not perfection.
- Residual mismatches are classified; genuine node bugs become follow-up issues.
- No f64 accumulation introduced for parity's sake (see guardrails).

Tolerances and the near-tie definition are inherited from
[02-parity-harness.md](./02-parity-harness.md).

## Findings (measured) — accepted residual, closed

Ran the process on en-fr, shortlist-off, over 60 diverse NLLB sentences:

- **Per-node parity is already exact.** `replay` on recorded traces shows *zero*
  divergent nodes at `rtol=1e-3` — every op, given the reference's own inputs, matches. So
  step 1 (bisect → the reduction op producing the first divergence) has no target: there is
  no single node exceeding tolerance. The greedy gap is sub-tolerance drift compounding
  through the nonlinearities (ReLU zeroing, layernorm mean/var, int8 re-quantization
  boundaries), not one bad reduction.
- **f64 diagnostic: precision is not the lever.** Accumulating the layernorm mean/variance
  and softmax denominator in f64 (env-gated probe, since reverted) changed 4/60 outputs but
  the match-rate vs `translator-cli` was **32/60 → 32/60** — it gained one sentence and lost
  another. Exactly the guardrail: more-accurate rounding is not *closer* rounding. Only
  replicating the reference's exact f32 NEON lane order would help, and that is the
  kernel-transliteration this issue explicitly rules out.

Per the stop criterion ("a fix that doesn't move the match-rate, revert it and stop"), the
f64 probe was reverted and this is closed. The residual (~32/60 on this deliberately diverse
corpus; much higher on clean text) is the accepted near-tie ceiling. No node bug was found —
`replay`'s zero-divergence result classifies the entire residual as reduction-order
near-ties, so there is no follow-up bug to file.
