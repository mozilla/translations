#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
protected:
  bool int16_{false};
  bool int8_{false};
  bool shifted_{false};
  bool shiftedAll_{false};
  bool dumpMatrices_{false};
  bool alpha_{false};
  bool legacyBatch_{false};

  void setGemmPrecision(Ptr<const Options> options) {
    std::string gemmPrecision = options->get<std::string>("gemm-precision");
    if (options->get<bool>("dump-quantmult")) {
      setInt8(true);
      setShifted(true);
      setShiftedAll(true);
      setDumpQuantMult(true);
      //float32, int16, int8, int8shift, int8shiftAlpha, int8shiftAll, int8shiftAlphaAll
    } else if (gemmPrecision == "float32") {
      return; // This is the default precisoin.
    } else if (gemmPrecision == "int16") {
      setInt16(true);
    } else if (gemmPrecision == "int8") {
      setInt8(true);
    } else if (gemmPrecision == "int8Alpha") {
      setInt8(true);
      setPrecomputedAlpha(true);
    } else if (gemmPrecision == "int8shift") {
      setInt8(true);
      setShifted(true);
    } else if (gemmPrecision == "int8shiftAlpha") {
      setInt8(true);
      setShifted(true);
      setPrecomputedAlpha(true);
    } else if (gemmPrecision == "int8shiftAll") {
      setInt8(true);
      setShifted(true);
      setShiftedAll(true);
    } else if (gemmPrecision == "int8shiftAlphaAll") {
      setInt8(true);
      setShifted(true);
      setShiftedAll(true);
      setPrecomputedAlpha(true);
    } else {
      ABORT("Unsupported GEMM precision type: {}", gemmPrecision);
    }
  }

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}

  void setDevice() override {}

  // Set the gemm precision
  void configureDevice(Ptr<const Options> options) override {
    setClip(options->get<float>("clip-gemm"));
    setGemmPrecision(options);
    setLegacyBatchedGemm(options->get<bool>("use-legacy-batching"));
  }

  void synchronize() override {}

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setInt16(bool optimize) override { int16_ = optimize; }
  bool isInt16() override { return int16_; }

  void setInt8(bool optimize) override { int8_ = optimize; }
  bool isInt8() override { return int8_; }

  void setShifted(bool shifted) override { 
#if (defined(__arm__) || defined(__aarch64__))
      LOG(info, "gemm-precision: *shifted* is not available on ARM; Setting to false.");
      shifted_ = false;
#else
      shifted_ = shifted; 
#endif
  }
  bool isShifted() override { return shifted_; }

  void setShiftedAll(bool shiftedAll) override {
#if (defined(__arm__) || defined(__aarch64__))
      LOG(info, "gemm-precision: *shifted* is not available on ARM; Setting to false.");
      shiftedAll_ = false;
      shifted_ = false;
#else
    shiftedAll_ = shiftedAll;
    if (shiftedAll_) {
      shifted_ = true;
    }
#endif
  }

  bool isShiftedAll() override {
    return shiftedAll_;
  }

  void setDumpQuantMult(bool dump) override {
    dumpMatrices_ = dump;
  }

  bool DumpQuantMult() override {
    return dumpMatrices_;
  }

  void setPrecomputedAlpha(bool alpha) override {
    alpha_ = alpha;
  }
  bool isPrecomputedAlpha() override {
    return alpha_;
  }

  void setLegacyBatchedGemm(bool legacyBatch) override {
    legacyBatch_ = legacyBatch;
  }
  bool isLegacyBatchedGemm() override {
    return legacyBatch_;
  }

};
}  // namespace cpu
}  // namespace marian
