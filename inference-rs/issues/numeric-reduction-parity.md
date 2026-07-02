# Numeric Reduction-Order Parity (timeboxed)

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
- So drive this by the [parity-harness.md](./parity-harness.md) flip-rate, stop at
  diminishing returns, and accept a high-but-imperfect match rate. Do **not** let parity
  turn the engine into a transliteration of the reference kernels.

## Depends on

[parity-harness.md](./parity-harness.md) — need the flip-rate metric and the per-node
bisector to target the right reductions and to know when to stop.
