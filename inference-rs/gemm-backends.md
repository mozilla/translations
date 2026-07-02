# CPU GEMM backends: native Apple arm64 vs WASM

Reference for what actually executes the matrix multiplies (and the surrounding
SIMD) when `inference/marian-fork` runs a Firefox `int8shiftAlphaAll` model. The
native arm64 build and the WASM build select **different backends at compile
time**, so they are not numerically interchangeable. This documents where they
agree and — more importantly — where they diverge, so a native build can be used
for development while knowing which behavior it does *not* reproduce.

## Terminology

- **Kernel** — one optimized routine for a single operation on a single
  instruction set (e.g. an AVX-512 int8 matmul, a NEON i8mm matmul).
- **GEMM library** — a library bundling such kernels for matrix multiply:
  intgemm, Ruy, FBGEMM. intgemm/FBGEMM are quantized-only (int8/int16); Ruy does
  both float and int8.
- **BLAS implementation** — a GEMM library exposing the standard BLAS API for
  *float* SGEMM: MKL, Apple Accelerate, OpenBLAS. Marian selects one via the
  `BLAS_VENDOR` cmake variable.

Umbrella term used below: **GEMM backend**.

## Dispatch is compile-time, not runtime

There is no runtime backend selection. The int8 path is chosen by preprocessor
guards in `graph/expression_operators.cpp` (`dot`, `affine`) and
`layers/generic.cpp`:

```cpp
#ifdef ARM
#include "tensors/cpu/ruy_interface.h"        // native ARM  -> Ruy
#else
#include "tensors/cpu/intgemm_interface.h"    // else:
#endif                                         //   #if defined(WASM)       -> wasm intgemm
                                               //   #elif defined(USE_INTGEMM) -> x86 intgemm
```

Float SGEMM is chosen by include precedence in `tensors/cpu/prod_blas.h`:
`MKL_FOUND` → `BLAS_FOUND` → `USE_ONNX_SGEMM` → `USE_RUY_SGEMM`.

`intgemm`/`FBGEMM` are x86 SIMD libraries (SSE/AVX); they have no ARM kernels,
which is why the ARM build cannot use them and substitutes Ruy + Accelerate.

## Backend breakdown

| Compute layer | Native Apple arm64 | WASM |
|---|---|---|
| **int8 matmul** (the quantized model weights) | **Ruy** — genuine signed-int8 GEMM on NEON (dotprod/i8mm where available). `cpu::integer::affineOrDotRUI`, `ruy_interface.h`. | **intgemm**, compiled to WASM SIMD. Built-in `int8*Fallback` kernels plus the optimized **MozIntGEMM** module linked at runtime from JS. `wasm_intgemm_interface.h`, `wasm_intgemm_fallback.cpp`. |
| **`int8shiftAlphaAll` algorithm** (+127 unsigned-domain shift, precomputed-alpha correction term) | **Not executed.** On ARM the code only branches on `isInt8()`; all int8 variants collapse to the same standard Ruy int8 GEMM. The shift/alpha logic exists only in the intgemm (`#else`) branch. | **Executed.** This is the algorithm's native home (`intgemm_interface.h`, gated by `backend.h` flags `setInt8/setShifted/setShiftedAll` + precomputed alpha). |
| **float SGEMM** (attention projections, non-int8 layers) | **Apple Accelerate** (`BLAS_FOUND=1`; `USE_RUY_SGEMM` off on Apple). | **onnx-sgemm** (`USE_ONNX_SGEMM`, the onnxjs WASM GEMM). |
| **elementwise / activations** (`float32x4` in `functional/operators.h`: layernorm, softmax, exp/log/sin, add/mul/…) | **NEON**, via the sse2neon shim mapping `_mm_*` to NEON + `neon_mathfun` for transcendentals. | **WASM SIMD** — the x86 SSE path (`sse_mathfun.h` + `<immintrin.h>`) lowered by emscripten (`-msimd128 -mssse3`). |
| **compile flags** | `-DARM -DFMA -DSSE -march=native`, Ruy + `USE_APPLE_ACCELERATE`, `USE_WASM_COMPATIBLE_SOURCE`. FBGEMM/MKL/intgemm off. | `-DCOMPILE_WASM=on -DWORMHOLE=off`, emscripten (`emcmake`), `USE_WASM_COMPATIBLE_SOURCE` on. |

## Consequences for parity

- **Shared, and safe to trust from a native build:** the float and elementwise
  layers — layernorm, softmax, attention scores, activation functions, and the
  float projections — are the same logic on both. Differences are only
  reduction-order / rounding noise.
- **Divergent, and NOT reproducible on native arm64:** the int8 matmul. Native
  ARM runs a *different algorithm* (Ruy standard int8) than WASM (intgemm
  shifted + precomputed-alpha). Same weights, real int8 arithmetic, different
  numerics.

To match WASM int8 behavior precisely you must run the **intgemm** path — i.e.
the WASM build itself, or the x86 build (which shares intgemm with WASM). A
native arm64 binary is correct for float-path parity and end-to-end iteration,
but is not a bit-level reference for `int8shiftAlphaAll`.

## Unifying the int8 path with gemmology (native arm64)

The divergence above exists only because intgemm is x86-only, which forced native
ARM onto Ruy (a different int8 algorithm). [gemmology](https://github.com/mozilla/gemmology)
is a rewrite of intgemm on top of xsimd that **preserves intgemm's
`int8shiftAlphaAll` algorithm and API** but adds real ARM NEON kernels (dotprod +
i8mm). Building the native arm64 engine against gemmology therefore runs the
*same* int8 algorithm as x86/WASM, so a native binary matches the intgemm numerics
(within reduction-order tolerance) instead of diverging.

gemmology is the **default** int8 backend for the native ARM build — a plain
build enables it, no flag required:

```
task inference-build      # or: inference/scripts/build.py
```

`build.py` passes `-DUSE_GEMMOLOGY=on` automatically on ARM. The gemmology + xsimd
submodules are checked out by the cmake configure step (`git submodule update --init
--checkout --recursive` in `inference/CMakeLists.txt`), the same as every other
`3rd_party` submodule — nothing gemmology-specific. (To fall back to the old Ruy int8
path you'd have to drop the `-DUSE_GEMMOLOGY=on` flag in `build.py` by hand.)

How it is wired (all gated behind the `USE_GEMMOLOGY` cmake option, ARM-only):

- `tensors/cpu/gemmology_intgemm_shim.h` re-exposes the intgemm API
  (`Int8`/`Int8Shift`/`callbacks`/`MaxAbsolute`) on top of `gemmology::`, so the
  existing intgemm node-op wiring (`intgemmPrepareA`, `intgemmAffine`, …) compiles
  and runs unchanged — emitting the same node types the golden-trace validation
  expects.
- `integer_common.h` includes the shim instead of the x86 intgemm header;
  `backend.h` honors `shifted`/`shiftedAll` on arm64; `expression_operators.cpp` /
  `layers/generic.cpp` take the intgemm path instead of Ruy; and
  `prepareAndTransposeB` repacks weights via `PrepareBQuantizedTransposed` (rather
  than the plain Ruy memcpy) into gemmology's tiled layout.
- gemmology + xsimd are header-only git submodules under `src/3rd_party/{gemmology,xsimd}`
  (gemmology `mozilla/gemmology`@`38ec34f1`, xsimd `xtensor-stack/xsimd`@`14.2.0`); no
  compiled lib. Run `git submodule update --init` to fetch them. The build only consumes
  gemmology's two headers and xsimd's `include/`, and is bumped to C++17 for these TUs.

WASM stays on intgemm/MozIntGEMM (gemmology ships no WASM SIMD kernels), and x86 stays on
intgemm. The **native arm64 gemmology build is the golden-trace oracle** for inference-rs:
it runs intgemm's `int8shiftAlphaAll` numerics on the host with no Docker, so it can be
diffed against the WASM/x86 intgemm path if a cross-check is ever needed.

## Key source locations

| Concern | File |
|---|---|
| int8 path dispatch | `graph/expression_operators.cpp` (`dot`, `affine`), `layers/generic.cpp` |
| Ruy int8 (ARM) | `tensors/cpu/ruy_interface.h` |
| intgemm int8 + `int8shiftAlphaAll` | `tensors/cpu/intgemm_interface.h` |
| WASM int8 / MozIntGEMM linkage | `tensors/cpu/wasm_intgemm_interface.h`, `wasm_intgemm_fallback.cpp` |
| int8 variant flags | `tensors/cpu/backend.h`, `common/config_parser.cpp`, `common/aliases.cpp` |
| float SGEMM selection | `tensors/cpu/prod_blas.h` |
| `float32x4` SIMD ops | `functional/operators.h`, `common/types.h` |
| backend/arch selection | `inference/marian-fork/CMakeLists.txt`, `inference/CMakeLists.txt` |
