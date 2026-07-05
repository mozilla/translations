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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gemmology.h"

namespace {

// Match the marian-fork ARM path exactly: i8mm dot-product over 128-bit NEON.
using Arch = xsimd::i8mm<xsimd::neon64>;

// Retained bytes of prepared-B weight buffers (the persistent C++ allocations
// invisible to dhat, which only sees the Rust heap). Reported for memory
// accounting.
std::atomic<size_t> g_prepared_bytes{0};

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

  g_prepared_bytes.fetch_add(n_pad * k, std::memory_order_relaxed);
  return new PreparedB{packed, n, n_pad, k};
}

void gemmology_free_b(void *handle) {
  if (!handle) return;
  PreparedB *h = static_cast<PreparedB *>(handle);
  g_prepared_bytes.fetch_sub(h->n_pad * h->k, std::memory_order_relaxed);
  std::free(h->data);
  delete h;
}

// Total retained bytes of prepared-B weight buffers (persistent C++ allocations).
size_t gemmology_prepared_bytes() {
  return g_prepared_bytes.load(std::memory_order_relaxed);
}

// Read logical row `id` (0 <= id < n) of the packed weight back into `out`
// (length k), inverting PrepareBQuantizedTransposed. That pack is a pure
// register-blocked reshuffle (no arithmetic, no interleave): the register holding
// row `id`, channel-block `cb` sits at index
//   reg = ((id / kColStride) * (k / kRegElems) + cb) * kColStride + (id % kColStride)
// and each register is kRegElems contiguous channels — so a row is k/kRegElems
// contiguous kRegElems-byte copies. Lets the caller drop the raw int8 copy and
// serve embedding lookups out of this one packed buffer.
void gemmology_read_row(const void *handle, size_t id, int8_t *out) {
  const PreparedB *h = static_cast<const PreparedB *>(handle);
  const size_t kblocks = h->k / kRegElems;
  const size_t row_block = id / kColStride;
  const size_t row_in = id % kColStride;
  for (size_t cb = 0; cb < kblocks; ++cb) {
    const size_t reg = (row_block * kblocks + cb) * kColStride + row_in;
    std::memcpy(out + cb * kRegElems, h->data + reg * kRegElems, kRegElems);
  }
}

// out[m, n] = unquant * (A[m,k] . W[k,n]) + bias[n]. `a` is shifted uint8 [m,k]
// (row-major), `bias` is the prepared bias of length n, `out` is [m, n]
// row-major. All caller buffers may be unaligned; the shim stages aligned
// copies as gemmology requires aligned SIMD access.
void gemmology_multiply(void *handle, const uint8_t *a, size_t m, float unquant,
                        const float *bias, float *out) {
  const PreparedB *h = static_cast<const PreparedB *>(handle);
  const size_t k = h->k, n = h->n, n_pad = h->n_pad;

  // Aligned scratch, allocated per call and freed at the end. (A persistent
  // per-weight or shared cache was tried and *raised* settled RSS — the retained
  // peak-sized buffers sit on top of the allocator's working set — so we keep
  // the transient form.) gemmology needs 16-byte-aligned A/bias/out.
  uint8_t *a_al = static_cast<uint8_t *>(aligned(m * k));
  std::memcpy(a_al, a, m * k);

  float *bias_al = static_cast<float *>(aligned(n_pad * sizeof(float)));
  std::memcpy(bias_al, bias, n * sizeof(float));
  for (size_t j = n; j < n_pad; ++j) bias_al[j] = 0.0f;  // pad tail (<8 cols)

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
