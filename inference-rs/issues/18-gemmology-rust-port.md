# Port gemmology to Rust (drop the C++ i8mm shim)

**Not scheduled** — a deliberate follow-on, recorded so the design is captured. This is a
portability / ergonomics investment, **not** a memory or speed fix (the memory wins are handled
separately by the C1+C2+B pass; see [../09-final-comparison.md](../09-final-comparison.md)).

## What exists today

The shifted int8 affine (`int8shiftAlphaAll`) runs through the vendored C++ **gemmology** i8mm
kernel via a thin FFI shim (the `gemmology` cargo feature, on by default):

- `src/gemmology_shim.cpp` — C-ABI over `gemmology::PrepareBQuantizedTransposed` +
  `Shift::Multiply` with `UnquantizeAndAddBiasAndWrite`; owns aligned scratch.
- `build.rs` — compiles it with `cc`, `-march=armv8.4-a+i8mm`, against the in-tree
  `gemmology`/`xsimd` submodules (aarch64 only).
- `src/gemm.rs` — `PreparedB` (weight packed once) + `matmul`.

It is validated bit-for-bit against the scalar `ops::intgemm_affine` (which is itself
oracle-validated) in `tests/gemm_parity.rs`, and it delivered the perf win in
[../08-perf-analysis.md](../08-perf-analysis.md).

## Why port it to pure Rust

1. **Drop the C++ dependency.** No `cc`, no C++17 toolchain, no `gemmology`/`xsimd` submodules to
   init. The crate becomes buildable with just `cargo` — closer to the memory-safe-Rust premise of
   the project, and simpler CI / packaging ([14-rust-only-package.md](./14-rust-only-package.md)).
2. **The packed weights become first-class Rust.** Today the prepared-B buffers are C++
   `aligned_alloc` and therefore invisible to dhat (we had to add a manual byte counter to account
   for ~41 MiB — see 09). A Rust-owned packed weight is dhat-visible and can be held directly by
   the weight store, so the "retain the packed representation, drop the raw" design (the **B** part
   of the memory pass) is expressed naturally in one place instead of split across an FFI boundary.
3. **One language for the hot path.** Easier to profile, inline, and evolve (e.g. batch-amortized
   GEMM shapes, the remaining ~3× gap to marian in 08/09).

## Scope of the port

`gemmology.h` is ~54 KB of templated C++ SIMD. We only use a narrow slice for the shifted path:

- `PrepareBQuantizedTransposed(input, output, k, n)` — repack transposed int8 `[n, k]` into the
  register-blocked layout (8-column groups × `k/16` × 8 × 16). Pure data movement; straightforward.
- `Shift::Multiply(A_u8, B_packed, m, k, n, callback)` — the i8mm dot-product accumulation, then
  `UnquantizeAndAddBiasAndWrite` (`f32(acc) * unquant + bias`).
- The reductions it leans on: `maddw` (= `vusdotq_s32`, unsigned×signed dot), `Pack0123`,
  `PermuteSummer`.

Rust has all of this in `std::arch::aarch64` behind `#[target_feature(enable = "dotprod,i8mm")]`:
`vusdotq_s32`, `vld1q_s8`/`vld1q_u8`, `vaddvq_s32`, the `vzip`/`vuzp`/`vpaddq` family for the lane
reductions. The kernel is unsafe intrinsics in a small, well-contained module; everything else
stays safe Rust.

## Validation (already in place)

The port is de-risked by the existing gate: `tests/gemm_parity.rs` asserts the kernel is
**bit-identical** to the scalar `ops::intgemm_affine` across shapes (including the non-multiple-of-8
padding path and both transformer inner dims). Swap the backend, keep the test green, and the
end-to-end `translate` / batch-invariance / oracle-parity suites confirm token-identical output.
So the risk is *effort*, not *correctness*.

## Layout / alignment notes to carry over

- `k % 16 == 0` (NEON int8 register width); output columns padded to a multiple of 8; operands
  16-byte aligned (Rust: allocate `Vec` with an aligned wrapper or over-allocate + align, or use
  `std::alloc` with a 64-byte `Layout`).
- A stays plain row-major shifted-`u8`; only B is repacked.
- The `+127` shift lives in `ops::prepare_a` already; the port only needs the packed-B multiply +
  unquant/bias callback.

## Recommendation

Do it **after** the C1+C2+B memory pass, as a standalone change: introduce a Rust `i8mm` module
behind the same `gemm::PreparedB` API, keep `tests/gemm_parity.rs` green, then delete
`gemmology_shim.cpp` / `build.rs` / the `cc` build-dep and fold the `gemmology` feature into the
default Rust path. At that point the packed weights are Rust-owned and the dhat picture is complete
without the manual counter.
