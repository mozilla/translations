#include "integer_common.h"

#ifdef __SSE__
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace integer {
// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;

#ifdef __AVX512F__
    // Multiples of 16 add together.
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#elif __SSE__
    // Multiples of 4 add together.
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int n4 = (n / 4) * 4;
    using __m128 = float32x4_t;
    for(; i < n4; i += 4) {
      __m128 ai = vld1q_f32(x + j * n + i);
      __m128 bi = vld1q_f32(bias + i);
      __m128 yi = vaddq_f32(ai, bi);
      vst1q_f32(y + j * n + i, yi);
    }

#else
    // StandardCPP No SIMD case.
    for(i = 0;  i < n; i++) {
        y[j * n + i] = x[j * n + i] + bias[i];
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }
}

} //integer
} //cpu
} //marian
