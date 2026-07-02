# Follow-ups: gemmology int8 backend

The gemmology backend (see [gemm-backends.md](./gemm-backends.md)) is the **default int8
path on ARM** (`USE_GEMMOLOGY`), with C++17 raised only for that build. Two follow-ups would
round it out.

## 1. Make the whole build C++17

### Current state

The marian-fork build is pinned to C++11 and only bumped to C++17 when gemmology is on:

- `inference/marian-fork/CMakeLists.txt:14` — `set(CMAKE_CXX_STANDARD 11)`
- `inference/marian-fork/CMakeLists.txt` (~lines 367 and 383) — `CMAKE_CXX_FLAGS` hardcodes
  `-std=c++11` (WASM branch and native branch respectively)
- The gemmology-only override sits right after `endif(MSVC)` (~line 403):
  ```cmake
  if(USE_GEMMOLOGY)
    set(CMAKE_CXX_STANDARD 17)
    string(REPLACE "-std=c++11" "-std=c++17" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif(USE_GEMMOLOGY)
  ```

This string-replace hack works but is ugly and only covers the gemmology config. gemmology
and xsimd *require* C++17, so as gemmology becomes the default (or x86 gains it, see below)
we should just standardize the whole engine on C++17.

### Proposed change

- Change `set(CMAKE_CXX_STANDARD 11)` → `17` at line 14.
- Replace the two hardcoded `-std=c++11` occurrences (native + WASM `CMAKE_CXX_FLAGS`) with
  `-std=c++17`.
- Delete the `if(USE_GEMMOLOGY)` C++17 override block — no longer needed.

### Validation / risk

- **Native arm64 is already proven C++17-clean**: the gemmology build compiled the *entire*
  engine at `-std=c++17 -Werror` with zero errors, so marian's own sources are fine at C++17
  on clang. The remaining unknowns are the other two toolchains.
- **x86 (Docker, gcc/clang)** — the native arm64 gemmology build is now the golden-trace
  oracle, so x86 is no longer required for inference-rs iteration. But it's still built for
  other uses, so build it at C++17 and confirm it still compiles (FBGEMM + intgemm + the rest).
- **WASM (emscripten)** — the WASM `CMAKE_CXX_FLAGS` branch also hardcodes `-std=c++11`; bump
  and confirm the emscripten build + `-Werror` still pass.
- Watch for C++17-removed features surfacing under `-Werror`: `std::random_shuffle`,
  dynamic exception specifications (`throw()`), `std::auto_ptr`, trigraphs, `std::unary_function`.
  None showed up in the native build, but 3rd-party/x86-only code may differ.
- Vendored 3rd-party libs pin their own standard via `target_compile_features`, so raising
  the top-level standard shouldn't regress them (it only raises the floor).

## 2. Add gemmology support for x86

gemmology is not ARM-specific — it ships x86 kernels for SSE2 / SSSE3 / AVX2 / AVX-512 /
AVXVNNI / AVX512VNNI (selected via `xsimd::default_arch` at compile time, same as the NEON
path). The shim (`tensors/cpu/gemmology_intgemm_shim.h`) is already architecture-agnostic, so
enabling x86 is mostly build wiring.

### Why do it

- One int8 code path across ARM + x86 (and eventually WASM) instead of intgemm-on-x86 /
  gemmology-on-ARM.
- Lets us **cross-validate** gemmology against intgemm on the *same* x86 machine: run the
  golden-trace dumper with both backends and diff the `intgemmAffine` outputs. That directly
  measures the reduction-order/rounding gap between gemmology and reference intgemm, and
  confirms the ARM gemmology build (the golden-trace oracle) is trustworthy against intgemm.
- Note: this is a confidence check on gemmology's numerics, not a change of oracle — the
  native arm64 gemmology build stays the golden-trace source for inference-rs.

### Current state (ARM-only wiring to generalize)

- `inference/marian-fork/CMakeLists.txt` — `option(USE_GEMMOLOGY ...)` and the xsimd
  `include_directories(...)` live *inside* the `if(... MATCHES "arm")` branch (~lines 68-108).
- `inference/scripts/build.py` — `run_cmake` appends `-DUSE_GEMMOLOGY=on` unconditionally in
  the non-x86 (`else`/ARM) branch, so gemmology is the default on ARM; the x86 branch never
  sets it.
- The int8 dispatch guards use `#if defined(ARM) && !defined(USE_GEMMOLOGY)`. On x86, `ARM`
  is undefined so those already take the intgemm path — **no source change needed** there.
  Likewise `prepareAndTransposeB` and `backend.h` already behave correctly on x86.

### Proposed change

- Move the `USE_GEMMOLOGY` option declaration and the xsimd `include_directories(...)` out of
  the ARM branch so both arches see them (guard the include dir on `if(USE_GEMMOLOGY)`).
- In the x86 CMake path, when `USE_GEMMOLOGY` is on: skip building/linking the intgemm static
  lib (the `AND NOT USE_GEMMOLOGY` guards in `CMakeLists.txt` and `src/3rd_party/CMakeLists.txt`
  already handle this generically — just confirm they trigger on x86). `USE_INTGEMM` stays on
  so the intgemm *code paths* compile against the shim.
- `build.py` — append `-DUSE_GEMMOLOGY=on` in the x86 branch too (currently only the ARM
  branch sets it).
- Confirm `-march=native` (already set for the native branch) is present so xsimd selects the
  right x86 arch; for the Docker reference, pick an explicit `BUILD_ARCH` (e.g. the AVX2 or
  AVX-512 target the reference is built for) so the kernel choice is reproducible.
- FBGEMM (`isPacked` path) is independent of the intgemm int8 path, so it can stay enabled;
  no interaction expected.

### Validation

- Build x86 with `-DUSE_GEMMOLOGY=on`; translate a sample and confirm coherent output.
- Golden-trace diff: same model + input through the x86 intgemm build vs the x86 gemmology
  build; assert `intgemmAffine` float outputs match within the inference-rs rtol/atol bar.

## (Stretch) unify WASM too

gemmology ships **no WASM SIMD kernels** today (xsimd has a `wasm` arch, but gemmology's
kernels are hand-written per-arch for x86/NEON only). Fully unifying WASM onto gemmology would
require either gemmology gaining wasm-simd128 kernels or falling back to its generic path — out
of scope for the two follow-ups above, but the natural end state once x86 is done.
