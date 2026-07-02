#pragma once
/*
 * gemmology -> intgemm compatibility shim.
 *
 * gemmology (https://github.com/mozilla/gemmology) is a rewrite of intgemm on top of
 * xsimd. It preserves intgemm's `int8shiftAlphaAll` algorithm (unsigned-A / +127 shift +
 * precomputed-alpha bias correction) but, unlike intgemm, has real ARM NEON kernels
 * (dotprod + i8mm). This header re-exposes the small slice of the intgemm API that
 * `intgemm_interface.h`, `integer_common.h` and `expression_graph_packable.h` use, mapping
 * each call onto its gemmology equivalent. That lets the existing intgemm node-op wiring
 * compile and run unchanged on ARM, so a native arm64 build runs the *same* int8 algorithm
 * as the x86/WASM intgemm path (see inference-rs/gemm-backends.md).
 *
 * Only int8 is supported (which is all `int8shiftAlphaAll` needs). Int16 and the
 * non-shifted int8 Multiply have no gemmology equivalent; they are provided as ABORT stubs
 * so the templates still compile. They are never reached for shifted models, exactly as on
 * the WASM path where those code paths also ABORT.
 */

#include "3rd_party/gemmology/gemmology.h"
#include "common/logging.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace intgemm {

// intgemm callbacks are structurally identical to gemmology's; alias them so
// `intgemm::callbacks::Unquantize*` resolves to the gemmology implementation.
namespace callbacks = ::gemmology::callbacks;

// intgemm::MaxAbsolute — gemmology does not ship one.
inline float MaxAbsolute(const float *begin, const float *end) {
  float result = 0.0f;
  for(const float *p = begin; p < end; ++p)
    result = std::max(result, std::fabs(*p));
  return result;
}

// intgemm::MeanStd / GetVectorMeanStd — only used behind the --dump-quantmult debug path.
struct MeanStd {
  float mean;
  float stddev;
};

inline MeanStd GetVectorMeanStd(const float *begin, const float *end, bool absolute = false) {
  const std::size_t n = static_cast<std::size_t>(end - begin);
  double sum = 0.0, sumsq = 0.0;
  for(const float *p = begin; p < end; ++p) {
    const double v = absolute ? std::fabs(static_cast<double>(*p)) : static_cast<double>(*p);
    sum += v;
    sumsq += v * v;
  }
  MeanStd result{0.0f, 0.0f};
  if(n != 0) {
    const double mean = sum / static_cast<double>(n);
    result.mean = static_cast<float>(mean);
    result.stddev = static_cast<float>(std::sqrt(std::max(0.0, sumsq / static_cast<double>(n) - mean * mean)));
  }
  return result;
}

// Standard (signed) int8 routines. Output pointers are int8_t, matching intgemm.
struct Int8 {
  static void PrepareA(const float *input, int8_t *output, float quant_mult,
                       std::size_t rows, std::size_t cols) {
    gemmology::PrepareA(input, output, quant_mult, rows, cols);
  }

  static void PrepareB(const float *input, int8_t *output, float quant_mult,
                       std::size_t rows, std::size_t cols) {
    gemmology::PrepareB(input, output, quant_mult, rows, cols);
  }

  static void PrepareBTransposed(const float *input, int8_t *output, float quant_mult,
                                 std::size_t cols, std::size_t rows) {
    gemmology::PrepareBTransposed(input, output, quant_mult, cols, rows);
  }

  static void PrepareBQuantizedTransposed(const int8_t *input, int8_t *output,
                                          std::size_t cols, std::size_t rows) {
    gemmology::PrepareBQuantizedTransposed(input, output, cols, rows);
  }

  template <typename IntegerTy>
  static void SelectColumnsB(const int8_t *input, int8_t *output, std::size_t rows,
                             const IntegerTy *cols_begin, const IntegerTy *cols_end) {
    gemmology::SelectColumnsB(input, output, rows, cols_begin, cols_end);
  }

  // Non-shifted multiply: no gemmology equivalent. Compiled (the shifted routing means it
  // is never reached), mirrors the WASM fallback which also refuses this path.
  template <typename Callback>
  static void Multiply(const int8_t * /*A*/, const int8_t * /*B*/, std::size_t /*rows_A*/,
                       std::size_t /*width*/, std::size_t /*cols_B*/, Callback /*callback*/) {
    ABORT("Non-shifted Int8::Multiply is not supported by the gemmology backend; "
          "use int8shiftAlphaAll (shifted) models.");
  }
};

// Shifted int8 routines — the int8shiftAlphaAll algorithm. gemmology's shifted A is
// uint8_t; the marian node ops hand us int8_t* buffers, so reinterpret across.
struct Int8Shift {
  static void PrepareA(const float *input, int8_t *output, float quant_mult,
                       std::size_t rows, std::size_t cols) {
    gemmology::Shift::PrepareA(input, reinterpret_cast<uint8_t *>(output), quant_mult, rows, cols);
  }

  template <typename Callback>
  static void PrepareBias(const int8_t *B, std::size_t width, std::size_t cols_B, Callback callback) {
    gemmology::Shift::PrepareBias(B, width, cols_B, callback);
  }

  template <typename Callback>
  static void Multiply(const int8_t *A, const int8_t *B, std::size_t rows_A,
                       std::size_t width, std::size_t cols_B, Callback callback) {
    gemmology::Shift::Multiply(reinterpret_cast<const uint8_t *>(A), B, rows_A, width, cols_B, callback);
  }
};

// int16 is not supported by gemmology. Provide ABORT stubs so the int16 templates in the
// intgemm node ops still compile; they are never instantiated for int8shiftAlphaAll models.
struct Int16 {
  static void PrepareA(const float *, int16_t *, float, std::size_t, std::size_t) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
  static void PrepareB(const float *, int16_t *, float, std::size_t, std::size_t) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
  static void PrepareBTransposed(const float *, int16_t *, float, std::size_t, std::size_t) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
  static void PrepareBQuantizedTransposed(const int16_t *, int16_t *, std::size_t, std::size_t) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
  template <typename IntegerTy>
  static void SelectColumnsB(const int16_t *, int16_t *, std::size_t, const IntegerTy *, const IntegerTy *) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
  template <typename Callback>
  static void Multiply(const int16_t *, const int16_t *, std::size_t, std::size_t, std::size_t, Callback) {
    ABORT("Int16 is not supported by the gemmology backend.");
  }
};

}  // namespace intgemm
