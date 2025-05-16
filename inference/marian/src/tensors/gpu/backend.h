#pragma once

#include "common/config.h"
#include "tensors/backend.h"  // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"
#include "common/logging.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

namespace marian {
namespace gpu {

// @TODO: in the future this should pobably become a fully fledged CudaInfo class with many attributes
struct CudaCompute {
  int major;
  int minor;
};

class Backend : public marian::Backend {
private:
  void setCudaComputeCapability() {
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.major, cudaDevAttrComputeCapabilityMajor, (int)deviceId_.no));
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.minor, cudaDevAttrComputeCapabilityMinor, (int)deviceId_.no));
  }

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setCudaComputeCapability();
  }

  ~Backend() {
    setDevice();
    if(cusparseHandle_) {
      cusparseDestroy(cusparseHandle_);
      cusparseHandle_ = 0;
    }
    if(cublasHandle_) {
      cublasDestroy(cublasHandle_);
      cublasHandle_ = 0;
    }
  }

  void configureDevice(Ptr<Options const> options) override {
    setClip(options->get<float>("clip-gemm"));
  }


  void setDevice() override { CUDA_CHECK(cudaSetDevice((int)deviceId_.no)); }

  void synchronize() override { CUDA_CHECK(cudaStreamSynchronize(0)); }

  cublasHandle_t getCublasHandle() {
    if(!cublasHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cublasCreate(&cublasHandle_);
    }
    return cublasHandle_;
  }

  cusparseHandle_t getCusparseHandle() {
    if(!cusparseHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cusparseCreate(&cusparseHandle_);
    }
    return cusparseHandle_;
  }

  CudaCompute getCudaComputeCapability() { return compute_; }

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  void setInt16(bool optimize) override {
    LOG_ONCE(info, "setOptimized() not supported for GPU_{}", optimize);
  }

  bool isInt16() override {
    return false;
  }

  void setInt8(bool optimize) override {
    LOG_ONCE(info, "setOptimized8() not supported for GPU_{}", optimize);
  }

  bool isInt8() override {
    return false;
  }

  void setShifted(bool shifted) override {
    LOG_ONCE(info, "setShifted() not supported for GPU_{}", shifted);
  }

  bool isShifted() override {
    return false;
  }

  void setShiftedAll(bool shiftedAll) override {
    LOG_ONCE(info, "setShiftedAll() not supported for GPU_{}", shiftedAll);
  }

  bool isShiftedAll() override {
    return false;
  }

  void setDumpQuantMult(bool dump) override {
    LOG_ONCE(info, "setDumpQuantMult() not supported for GPU_{}", dump);
  }

  bool DumpQuantMult() override {
    return false;
  }

  void setPrecomputedAlpha(bool alpha) override {
    LOG_ONCE(info, "setPrecomputedAlpha() not supported for GPU_{}", alpha);
  }
  bool isPrecomputedAlpha() override {
    return false;
  }

  void setLegacyBatchedGemm(bool legacyBatch) override {
    LOG_ONCE(info, "setLegacyBatchedGemm() not supported for GPU_{}", legacyBatch);;
  }
  bool isLegacyBatchedGemm() override {
    return false;
  }

private:
  cublasHandle_t cublasHandle_{0};     // make sure it's 0, so it can be initalized lazily
  cusparseHandle_t cusparseHandle_{0}; // as above
  CudaCompute compute_;
};
}  // namespace gpu
}  // namespace marian
