# x86 SIMD backend for the int8 affine (wire up gemmology's existing kernels)

**Partly landed.** The AVX2 baseline is wired: `build.rs` compiles the shim on x86_64 with
`-mavx2`, the shim is parameterized over `xsimd` arches (see below), and CI proves it runs on a
real x86 runner (`FXTRANSLATE_REQUIRE_SIMD=1`, backend reported as `avx2`). **Remaining:** the VNNI
paths (`avxvnni` / `avx512vnni`) behind a runtime CPUID dispatcher — item 3 below. Touches the
`fxtranslate` engine crate (the shim + `build.rs`).

**CI testability constraint:** stock GitHub x86 runners top out at AVX2 (no AVX-512, no VNNI —
probed `avx2 sse sse2 sse4_1 sse4_2 sse4a`), so a VNNI kernel can be built and dispatched but *not
executed* on them. Validating VNNI numerics needs an Intel SDE-emulated leg (what intgemm/marian
use; VNNI is exact so it can assert full-range parity) or a VNNI-capable larger runner. Until then
the VNNI rows in [../gemm-backends.md](../gemm-backends.md) are marked unvalidated by design.

## Landed (AVX2 baseline + cheat-proof gate)

- The shim (`gemmology_shim.cpp`) no longer hardcodes `xsimd::i8mm<neon64>`: `build.rs` selects the
  arch via a `-D` define (`FXT_GEMM_I8MM` / `FXT_GEMM_AVX2`) and the NEON-shaped `kRegElems = 16`
  is now `xsimd::batch<int8_t, Arch>::size` (16/32/64), matching gemmology's own `RegisterElems`.
  `kColStride = 8` is arch-independent.
- `build.rs` compiles the shim on aarch64 (i8mm) and x86_64 (avx2); other targets still fall back
  to scalar. `FXTRANSLATE_REQUIRE_SIMD` turns any fallback (unsupported arch, missing headers, a
  failed C++ build, `portable`) into a build-time panic — no silent scalar degrade.
- `gemm::backend()` returns the compiled kernel's `xsimd::Arch::name()` (`i8mm+neon64` / `avx2` /
  `scalar`). `tests/gemm_parity.rs` asserts it isn't `scalar` when SIMD is required and that at
  least one shape exercised it — validation sourced from the compiled C++, not fakeable in Rust.
- CI (`.github/workflows/inference-rs.yml`) runs the test job as an `arm-i8mm` + `x86-avx2` matrix
  with `FXTRANSLATE_REQUIRE_SIMD=1`, so both kernels are proven live on every push.

## The gap

gemmology (`crates/fxtranslate/vendor/gemmology/gemmology.h`) is a multi-arch int8 GEMM library
built on xsimd, with kernels for:

- **x86**: `avx512bw`, `avx512vnni`, `avxvnni`, `avx2`, `ssse3`, `sse2`
- **ARM**: `neon`, `neon64`, `i8mm<neon64>` (the fast dot-product path)

`avx512vnni` / `avxvnni` are the x86 analogues of ARM i8mm's `usdot` — the same int8
dot-product acceleration marian/intgemm dispatch to on x86.

But this port instantiates exactly one architecture. The shim hardcodes it
(`crates/fxtranslate/src/gemmology_shim.cpp`):

```cpp
using Arch = xsimd::i8mm<xsimd::neon64>;   // the only kernel compiled
```

and `build.rs` only compiles the shim on aarch64. On x86 the engine falls back to the scalar
`ops::intgemm_affine` (correct, ~an order of magnitude slower on the GEMM). So x86 users get a
working-but-slow engine even though the fast kernels exist in-tree.

## What "wiring it up" involves

Not a new kernel — but not a one-liner either. The work:

1. **Generalize the shim beyond one `Arch`.** The packing constants are NEON-shaped:
   `kColStride = 8`, `kRegElems = 16` (`gemmology_shim.cpp`). AVX2 (32 int8/reg) and AVX-512
   (64) differ, so the padding/stride logic must be parameterized per arch (or derived from
   gemmology) rather than hardcoded to 16.
2. **Pick the x86 kernel(s) + build flags.** In `build.rs`, compile the shim with the right
   `-m` flags per target: a portable baseline (e.g. `avx2`) and/or the VNNI paths
   (`-mavxvnni` / `-mavx512vnni`). xsimd selects the kernel from the compile-time arch.
3. **Runtime dispatch (the real decision).** gemmology's README is explicit: *"It's up to the
   user to handle the dynamic dispatch."* xsimd picks the best arch at *compile* time, so a
   shipped x86 binary that must run on varied CPUs needs either:
   - a fixed baseline (compile for AVX2, accept no VNNI), or
   - multiple instantiations + a CPUID dispatcher (SSE2 → AVX2 → AVX-VNNI → AVX-512-VNNI),
     which is what intgemm/marian do. This is the "runs fast everywhere on x86" answer and the
     bulk of the effort.
4. **Parity + build gating.** An x86 `gemm_parity`-style test (SIMD == scalar == oracle,
   transitively), and extend `build.rs`'s arch handling + the `gemmology_simd` cfg to x86.

## Interaction with the current build

The engine now builds the SIMD kernel *opportunistically*: `build.rs` compiles the shim and emits
`--cfg gemmology_simd` only when it can (aarch64 today; `portable` off; a C++ compiler present),
and otherwise falls back to the scalar kernel without failing the build. Wiring x86 means adding an
x86 branch to that arch handling — no changes needed in `weights.rs` or downstream, since the
`gemm` module already presents a uniform API with a scalar stub when `gemmology_simd` is unset.

## Note

A pure-Rust port of the kernel (e.g. `std::simd`) would remove the C++ toolchain requirement and
cover x86/ARM/wasm from one codebase, but was ruled out for the ongoing maintenance burden of
hand-carrying a SIMD GEMM. Wiring up gemmology's existing C++ x86 kernels is the path.
