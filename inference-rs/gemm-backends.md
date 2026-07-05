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
| **wasm (Firefox)** | escapes to a *native* kernel | per that kernel's arch | same as native on that CPU | Firefox in-browser engine | reference engine, not this crate |

"Exact" = bit-identical to the scalar int32 reference, and therefore to the marian oracle the
scalar kernel is validated against. "Approx" = identical *until* the int16 lane saturates.

Firefox is not a distinct WASM-SIMD kernel: its translations WASM module imports the int8 GEMM
from a native, sandbox-external module (`WebAssembly.mozIntGemm`; see the import in
`inference/marian-fork/src/tensors/cpu/wasm_intgemm_interface.h` and `inference/wasm/import-gemm-module.js`),
so it runs native CPU-dispatched code — the same behavior as a native build on that CPU. The
in-WASM kernel is only a fallback.

**marian's x86 path is runtime-dispatched, and mostly saturating.** The forked marian's int8
CPU inference uses **intgemm** on x86 (not gemmology, not FBGEMM), which CPUID-dispatches
`AVX-512-VNNI → AVX-512BW → AVX2 → SSSE3` (`inference/marian-fork/src/3rd_party/intgemm/intgemm/intgemm.h`).
Only the **AVX-512-VNNI** tier (`vpdpbusd`) is exact; **AVX2, SSSE3, and even AVX-512BW all
saturate** via `maddubs`. On ARM, marian uses gemmology (i8mm / neon64), which is exact. intgemm
and gemmology emit the *same* instructions per arch, so inference-rs matches production
arch-for-arch: our AVX2 == intgemm AVX2 (saturating), our i8mm == marian ARM gemmology (exact).
The practical divergence axis is therefore the **CPU's ISA**, not the engine — two Firefox users
on a VNNI server vs. an AVX2 laptop already differ the same way.

VNNI (`vpdpbusd`) is the x86 analogue of ARM's `usdot`: it accumulates into i32 directly, so it
is both faster than AVX2 *and* exact. Wiring it up (behind a runtime CPUID dispatch) is the
remaining x86 work.

## Validation status

What CI actually proves, and what it can't with the hardware available. The GitHub runners are
`ubuntu-24.04-arm` (i8mm) and stock `ubuntu-latest` x86 — and the x86 runner tops out at **AVX2**
(probed flags: `avx2 sse sse2 sse4_1 sse4_2 sse4a`; no AVX-512, no VNNI). So a backend is only
CI-validated if some runner CPU can actually execute it. `tests/gemm_parity.rs` is the harness;
`FXTRANSLATE_REQUIRE_SIMD` / `FXTRANSLATE_REQUIRE_SCALAR` make a wrong backend a hard failure.

| Backend | Wired | CI-validated | How, or why not |
|---|---|---|---|
| **scalar** | ✓ fallback | ✓ | `test-portable` leg: `fast,portable`, `REQUIRE_SCALAR` asserts the build fell back to scalar; tests pass with no C++ toolchain. |
| **i8mm+neon64** | ✓ | ✓ **exact, full-range** | `test (arm-i8mm)`: `REQUIRE_SIMD`, bit-parity vs. scalar on adversarial full-range inputs (`usdot` → i32, no saturation). |
| **avx2** | ✓ | ◐ **in-regime; saturation characterized** | `test (x86-avx2)`: `REQUIRE_SIMD`, exact on bounded (±63) inputs; the full-range divergence is *measured and reported*, not asserted (it's an accepted approximation). |
| **avxvnni** | ✗ | ✗ **unvalidated** | No stock runner exposes VNNI. Would be exact → could assert full-range parity, but only under Intel SDE emulation or a VNNI-capable larger runner. |
| **avx512vnni** | ✗ | ✗ **unvalidated** | No AVX-512 on stock runners. Same path to validation as `avxvnni`. |
| **ssse3 / sse2** | ✗ | ✗ **unvalidated** | Kernel exists in gemmology and *is* executable on the stock runner (SSE present), but not wired. Would saturate like AVX2. Testable here if we ever ship it. |
| **wasm** | n/a | ✗ **unvalidated here** | A separate engine/toolchain (Firefox), not built by this crate. Validated in the Firefox tree, not here. |

Legend: ✓ proven · ◐ proven within its valid numeric range, known-divergent outside it · ✗ not exercised.

**Gaps and how to close them.** The exact x86 path (VNNI) is the notable unvalidated one, purely
because no available runner can run it. An **Intel SDE**-emulated leg (the tool intgemm/marian use)
would run the VNNI-compiled tests on any x86 and — since VNNI is exact — assert full-range parity,
the strongest gate here; a VNNI-capable larger runner would do it on real silicon. SSE would only
be validated if we choose to ship it. Until then those rows stay ✗ by design, not by oversight.

## Why Firefox and inference-rs can disagree — it's the CPU, not the engine

Firefox and inference-rs run the *same family* of kernel (native, CPU-dispatched int8 GEMM), so
on the **same CPU** they agree (modulo reduction order). They diverge when they run on CPUs with
different accumulator behavior:

- Firefox on a non-VNNI x86 (most laptops) saturates; inference-rs on x86 (AVX2) saturates too →
  **they match**.
- Firefox-native and inference-rs both on ARM i8mm are exact → **they match**.
- Firefox on a saturating x86 vs. inference-rs on an exact Apple-Silicon Mac → **they differ** —
  but so do two Firefox users on a VNNI server vs. an AVX2 laptop. The split is inherent to the
  quantization scheme, not to either engine.

So observed reference-translation differences trace to the **accumulator-width split by CPU ISA**
(exact `usdot`/`vpdpbusd` vs. saturating `maddubs`), not a bug and not something inference-rs
introduces — it mirrors production arch-for-arch.

## Is the shipped model biased toward saturation, or toward x86?

Short answer: **no meaningful bias.** Two facts from the training pipeline (`pipeline/quantize/`,
`pipeline/train/`) settle it:

- **Fine-tuning never runs the int8 kernel.** `student.finetune.yml`'s `quantize-bits: 8` uses
  marian's `ModelQuantizer` (`marian-dev/src/optimizers/quantizer.cpp`): it rounds *weights* to the
  int8 grid and reverts them to float — ordinary FP math on GPU, weights only, **no integer
  accumulation, no saturation**. The model is tuned to tolerate weight *rounding*, not to depend on
  saturation. So an exact kernel is not "wrong" — there is no saturation-trained reference to
  deviate from.
- **Calibration is a magnitude statistic with headroom.** The activation multipliers come from
  `marian-decoder --dump-quantmult` on the devset: `QuantMultA = 127 / (mean(maxAbs) + 1.1·std)`
  (`pipeline/quantize/extract_stats.py`). That measures how large activations get (plus ~1.1σ of
  headroom), not how the accumulator behaves — essentially saturation-independent.

The one real x86 flavor: calibration runs on an x86 GCP CI worker, so the alphas are gathered on
x86 int8 numerics. But it's magnitude-based with headroom, and the model ships as ISA-neutral
`intgemm8` (dispatched per-CPU at inference), so it privileges neither saturation nor a specific
ISA. Net: exact kernels (ARM i8mm, VNNI, our scalar) are faithful — arguably closer to the float
model than the saturating x86 default most users actually run.

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
