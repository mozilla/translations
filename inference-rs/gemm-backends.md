# int8 GEMM backends

The hot path in inference-rs is one operation: the int8 affine (`int8shiftAlphaAll`) — a
matrix multiply of quantized activations by quantized weights. Every backend computes the
**same** dot product; they differ only in *what integer the running total lives in*, and that
one difference decides whether a backend is bit-exact or a (small) approximation.

This doc is the high-level map: which backend runs where, and where the numbers can diverge.
It is not exhaustive — for the arithmetic see `crates/fxtranslate/tests/gemm_parity.rs`, and for
the wiring see `crates/fxtranslate/build.rs` and `src/gemmology_shim.cpp`.

## The one thing that matters: accumulator width

The dot product is `sum(a[i] * b[i])` over the inner dimension, where `a` is a `uint8`
activation (0–255, the `+127`-shifted form) and `b` is an `int8` weight (−128–127).

- **Exact backends** accumulate straight into a **32-bit** integer. A 32-bit total can't
  overflow at these sizes, so the result is bit-identical to the plain scalar loop.
- **Saturating backends** have no "u8×s8 into i32" instruction, so they first sum *two*
  products into a **16-bit** lane (`maddubs`) and then widen to 32-bit. A 16-bit lane clamps
  at ±32767: `2 × 255 × 127 = 64516` doesn't fit, so it saturates and the answer drifts.

Real quantized activations are small and clustered near zero, so the 16-bit lane rarely
overflows and saturating backends agree with the exact ones in practice. The divergence only
shows up on large/adversarial values — which is why `gemm_parity.rs` tests parity with weights
bounded to ±63 (guaranteed no saturation) and *separately* characterizes the full-range case.

## Backends

`gemm::backend()` reports which one is live (from the compiled shim's `xsimd::Arch::name()`).

| Backend | ISA / key instruction | Accumulator | Exact vs. scalar? | Also used by | inference-rs status |
|---|---|---|---|---|---|
| **scalar** | portable Rust (`ops::intgemm_affine`) | i32 | exact — the reference | — | always present (fallback) |
| **i8mm+neon64** | AArch64 `usdot` | i32 | **exact** | marian on ARM | **wired** — default on aarch64 |
| **avx2** | x86 `vpmaddubsw` → `vpmaddwd` | i16 → i32 | approx — i16 saturates | marian x86 baseline, WASM | **wired** — default on x86_64 |
| **avxvnni** | x86 `vpdpbusd` | i32 | **exact** | modern x86 (Alder Lake+) | not wired — see below |
| **avx512vnni** | x86-512 `vpdpbusd` | i32 | **exact** | server x86 | not wired — see below |
| **ssse3 / sse2** | x86 `pmaddubsw` | i16 → i32 | approx — i16 saturates | older x86 | in gemmology, not wired |
| **wasm** | WASM SIMD i16 madd | i16 → i32 | approx — i16 saturates | Firefox in-browser engine | reference engine, not this crate |

"Exact" = bit-identical to the scalar int32 reference, and therefore to the marian oracle the
scalar kernel is validated against. "Approx" = identical *until* the int16 lane saturates.

VNNI (`vpdpbusd`) is the x86 analogue of ARM's `usdot`: it accumulates into i32 directly, so it
is both faster than AVX2 *and* exact. Wiring it up (behind a runtime CPUID dispatch) is the
remaining x86 work — see [issues/x86-gemmology-backend.md](./issues/x86-gemmology-backend.md).

## Why Firefox (WASM) and inference-rs can disagree

Firefox's in-browser engine runs the WASM int8 kernel, which uses the **saturating** i16 path —
the same family as x86 AVX2/SSE. inference-rs on Apple Silicon runs **i8mm**, which is exact. So
on inputs where the i16 lane saturates, the WASM reference and inference-rs-on-ARM produce
slightly different logits, and occasionally a different argmax (hence a different token). This
is the most likely source of observed reference-translation differences between the two — not a
bug in either, but the accumulator-width split above. On an x86 build, inference-rs uses AVX2
and lands on the *same* saturating side as WASM.

## Portable options (Cargo features)

`fast` is on by default and always builds; `build.rs` picks the SIMD kernel for the target and
falls back to scalar where none is wired.

| feature | effect |
|---|---|
| `fast` (= `gemmology` + `lean-embed`) | Default. SIMD int8 kernel where wired (i8mm / AVX2), else scalar. |
| `gemmology` | Request the SIMD kernel. `build.rs` compiles the shim for aarch64/x86_64 and emits `--cfg gemmology_simd`; otherwise the scalar stub is used. |
| `portable` | Force the scalar kernel — no SIMD, no C++ toolchain — even with `fast` on. For reproducible/audited builds and targets without a C++ compiler. |
| `lean-embed` | Drop resident f32 embedding tables (memory win); pure Rust, orthogonal to the GEMM backend. |

`build.rs` selects the arch (i8mm on aarch64, AVX2 on x86_64) and can be pinned:

- `FXTRANSLATE_REQUIRE_SIMD=1` — fail the build instead of falling back to scalar. CI sets this
  on the arm-i8mm and x86-avx2 jobs, so a green build proves the SIMD kernel is actually live.
- `GEMMOLOGY_DIR` / `XSIMD_INCLUDE_DIR` — override the vendored header include paths.
