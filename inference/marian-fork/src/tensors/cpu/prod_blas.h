#pragma once
#if MKL_FOUND
    #include <mkl.h>
#elif BLAS_FOUND
    #include <cblas.h>
#elif USE_ONNX_SGEMM
    #include "3rd_party/onnxjs/src/wasm-ops/gemm.h"
#elif USE_RUY_SGEMM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
    #include "ruy/ruy.h"
    #include "ruy/system_aligned_alloc.h"
#pragma GCC pop
#endif

#if USE_RUY_SGEMM
// AlignedVector allocates aligned memory and cleans up after itself. RAII
// wrapper similar to intgemm's AlignedVector.
template <class T>
class AlignedVector {
public:
  AlignedVector(size_t num_elem)
      : size_(num_elem),
        storage_(reinterpret_cast<T *>(ruy::detail::SystemAlignedAlloc(sizeof(T) * num_elem))) {}

  T *begin() { return storage_; }
  T *data() { return storage_; }
  size_t size() const { return size_; }
  size_t memSize() const { return sizeof(T) * size_; }

  // Forbid copy
  AlignedVector(const AlignedVector &) = delete;
  AlignedVector &operator=(const AlignedVector &) = delete;

  ~AlignedVector() { ruy::detail::SystemAlignedFree(reinterpret_cast<void *>(storage_)); }

private:
  size_t size_;
  T *storage_;
};


inline void GemmRuy(const bool transA,
                    const bool transB,
                    const int M,
                    const int N,
                    const int K,
                    const float alpha,
                    const float *A,
                    const int lda,
                    const float *B,
                    const int ldb,
                    const float beta,
                    float *C,
                    const int ldc) {
  ruy::Context context;

  // If we need to transpose, we can swap dimensions in layout claim the matrix
  // is just column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
  lhs.set_data(A);

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
  rhs.set_data(B);

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());

  if(beta == 0) {
    // For beta = 0, we want to avoid the additional allocation. This is a
    // large amount of our inference use-cases. sgemm is called with `beta` for
    // accumulating gradients in backpropogation, which is 0.0 during
    // inference.

    dst.set_data(C);
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    if(alpha != 1.0) {
        // Write out C as C = alpha * [op(A) * op(B)] + beta * C
        // Can we expect the compiler to autovectorize this?
        // TODO: Come back and explicitly use SIMD.
        const size_t size    = M * N;
        const float *opA_opB = C;  // Alias.
        for(size_t i = 0; i < size; i++) {
          C[i] = alpha * opA_opB[i];
        }
    }

  } else {
    // @jerinphilip has not yet been able to find a ruy primitive that does in
    // place addition to obtain full gemm.
    //
    // Safe bet is to make an additional allocation to store the result of
    // multiply  and use the existing values in C.
    //
    // See also: https://github.com/google/ruy/issues/307

    AlignedVector<float> intermediate(M * N);
    dst.set_data(intermediate.data());
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    // Write out C as C = alpha * [op(A) * op(B)] + beta * C
    // Can we expect the compiler to autovectorize this?
    // TODO: Come back and explicitly use SIMD.
    const size_t size    = M * N;
    const float *opA_opB = intermediate.data();
    for(size_t i = 0; i < size; i++) {
      C[i] = alpha * opA_opB[i] + beta * C[i];
    }
  }
}

#endif // RUY_SGEMM


inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc) {
#if BLAS_FOUND
        cblas_sgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    rows_a,
                    rows_b,
                    width,
                    alpha,
                    a,
                    lda,
                    b,
                    ldb,
                    beta,
                    c,
                    ldc);
#elif USE_ONNX_SGEMM
        gemm_f32_imp(transA, transB, rows_a, rows_b, width, alpha, a, b, beta, c);
#elif USE_RUY_SGEMM
        GemmRuy(transA, 
                transB,
                rows_a,
                rows_b,
                width,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc);
#else 
    transA; transB; rows_a; rows_b; width; alpha; a; lda; b; ldb; beta; c; ldc; // make compiler happy
    ABORT("Marian must be compiled with a BLAS library");
#endif
}
