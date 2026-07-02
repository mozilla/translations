// Thin C-ABI shim over the vendored gemmology i8mm kernel (the same kernel the
// marian-fork uses on ARM). It exposes exactly the two operations inference-rs
// needs for the shifted int8 affine (`int8shiftAlphaAll`):
//
//   1. gemmology_prepare_b  — one-time transform of a logical transposed int8
//      weight [n, k] (row-major, w[n*k + kk]) into gemmology's register-blocked
//      layout. Returns an opaque handle owning aligned memory.
//   2. gemmology_multiply   — out[m,n] = unquant * (A[m,k] . W[k,n]) + bias[n],
//      where A is the shifted uint8 activation and `bias` is the *prepared* bias
//      (the caller folds in the +127 shift correction, matching marian).
//
// Numerics are identical to the scalar `ops::intgemm_affine`: the integer
// accumulation is exact, and the callback does `float(acc)*unquant + bias`, the
// same formula. `tests/gemm_parity.rs` asserts bit-for-bit-close equality
// against that scalar kernel, which is itself validated against the marian
// oracle — so this path inherits that validation transitively.
//
// gemmology requires: k % 16 == 0 (NEON int8 register width) and rows padded to
// a multiple of 8 columns, with 16-byte-aligned operands (it uses aligned SIMD
// loads/stores). This shim owns all aligned, zero-padded scratch buffers so the
// Rust side can pass plain, unaligned slices. If k is not a multiple of 16,
// gemmology_prepare_b returns null and the caller falls back to the scalar path.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gemmology.h"

namespace {

// Match the marian-fork ARM path exactly: i8mm dot-product over 128-bit NEON.
using Arch = xsimd::i8mm<xsimd::neon64>;

// gemmology packs 8 output columns per group; the NEON int8 register holds 16.
constexpr size_t kColStride = 8;
constexpr size_t kRegElems = 16;

inline size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Aligned allocation sized for SIMD stores. std::aligned_alloc requires the
// size to be a multiple of the alignment, so round it up; never return null for
// a zero request.
inline void *aligned(size_t bytes) {
  size_t n = round_up(bytes ? bytes : 1, 64);
  return std::aligned_alloc(64, n);
}

struct PreparedB {
  int8_t *data;  // register-blocked, 16-byte aligned; logically [n_pad, k]
  size_t n;      // logical output columns
  size_t n_pad;  // padded up to a multiple of kColStride (8)
  size_t k;      // inner dimension (a multiple of kRegElems, 16)
};

}  // namespace

extern "C" {

// Prepare a logical transposed int8 weight [n, k] (w[col*k + kk]) into
// gemmology's layout. Returns null if k is not a multiple of 16.
void *gemmology_prepare_b(const int8_t *b_transposed, size_t n, size_t k) {
  if (k % kRegElems != 0) return nullptr;
  const size_t n_pad = round_up(n, kColStride);

  // PrepareBQuantizedTransposed does aligned loads and needs rows % 8 == 0, so
  // stage the weight in an aligned, zero-padded [n_pad, k] buffer first.
  int8_t *src = static_cast<int8_t *>(aligned(n_pad * k));
  std::memset(src, 0, round_up(n_pad * k, 64));
  std::memcpy(src, b_transposed, n * k);

  int8_t *packed = static_cast<int8_t *>(aligned(n_pad * k));
  gemmology::PrepareBQuantizedTransposed<Arch>(src, packed, /*cols=*/k,
                                               /*rows=*/n_pad);
  std::free(src);

  return new PreparedB{packed, n, n_pad, k};
}

void gemmology_free_b(void *handle) {
  if (!handle) return;
  PreparedB *h = static_cast<PreparedB *>(handle);
  std::free(h->data);
  delete h;
}

// out[m, n] = unquant * (A[m,k] . W[k,n]) + bias[n]. `a` is shifted uint8 [m,k]
// (row-major), `bias` is the prepared bias of length n, `out` is [m, n]
// row-major. All caller buffers may be unaligned; the shim stages aligned
// copies as gemmology requires aligned SIMD access.
void gemmology_multiply(const void *handle, const uint8_t *a, size_t m,
                        float unquant, const float *bias, float *out) {
  const PreparedB *h = static_cast<const PreparedB *>(handle);
  const size_t k = h->k, n = h->n, n_pad = h->n_pad;

  // Aligned A copy [m, k]; k % 16 == 0 keeps every row 16-byte aligned.
  uint8_t *a_al = static_cast<uint8_t *>(aligned(m * k));
  std::memcpy(a_al, a, m * k);

  // Aligned, zero-padded bias [n_pad] (AddBias uses aligned loads).
  float *bias_al = static_cast<float *>(aligned(n_pad * sizeof(float)));
  std::memset(bias_al, 0, round_up(n_pad * sizeof(float), 64));
  std::memcpy(bias_al, bias, n * sizeof(float));

  // Aligned output scratch [m, n_pad]; Write uses aligned stores at
  // row*n_pad + col (col steps by 8), which stays 16-byte aligned.
  float *out_al = static_cast<float *>(aligned(m * n_pad * sizeof(float)));

  gemmology::Shift::Multiply<Arch>(
      a_al, h->data, m, k, n_pad,
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(unquant, bias_al,
                                                         out_al));

  // Copy the valid [m, :n] region back into the caller's tight [m, n] buffer.
  for (size_t r = 0; r < m; ++r)
    std::memcpy(out + r * n, out_al + r * n_pad, n * sizeof(float));

  std::free(a_al);
  std::free(bias_al);
  std::free(out_al);
}

}  // extern "C"
