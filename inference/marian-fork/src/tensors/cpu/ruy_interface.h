/*
 * This file follows intgemm and is a means of retrofitting ruy into the intgemm based wiring in
 * `intgemm_interface.h`. ruy is an inference backend used in tensorflow and android deployment and
 * has an optimized ARM backend for the multiply operations required. Optimized code for quantize,
 * unquantize, transpose are added separately to connect the multiply library to marian.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "integer_common.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#include "ruy/platform.h"
#include "ruy/system_aligned_alloc.h"
#pragma GCC diagnostic pop

#if RUY_PLATFORM_NEON
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace integer {

using Index = unsigned int;

#if RUY_PLATFORM_NEON

/*
 * Optimized path using ARM NEON SIMD intrinsics. Currently only supports int8_t.
 */
inline void quantize(const float *input, int8_t *output, float scale, Index rows, Index width) {
  const float32x4_t *Input = reinterpret_cast<const float32x4_t *>(input);
  const float32x4_t *InputEnd = reinterpret_cast<const float32x4_t *>(input + rows * width);

  int8x8_t *Output = reinterpret_cast<int8x8_t *>(output);
  while(Input != InputEnd) {
    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_lo = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_lo = vcvtnq_s32_f32(scaledFloat_lo);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);   // VQMOVN.S32 d0,q0
    int16x4_t s16x4_lo = vqmovn_s32(scaledInt_lo);

    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_hi = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_hi = vcvtnq_s32_f32(scaledFloat_hi);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);
    // VQMOVN.S32 d0,q0
    int16x4_t s16x4_hi = vqmovn_s32(scaledInt_hi);

    // Combine two ints.
    // int16x8_t   vcombine_s16(int16x4_t low, int16x4_t high);
    int16x8_t s16x8 = vcombine_s16(s16x4_lo, s16x4_hi);

    // Vector saturating narrow integer
    int8x8_t s8x8 = vqmovn_s16(s16x8);

    *Output = s8x8;
    ++Output;
  };
}

inline void _transpose_16x16(const int8_t *src,
                             Index i,
                             Index j,
                             Index rows,
                             Index cols,
                             int8_t *dst) {
  // Implemented following the algorithm described in
  // https://stackoverflow.com/a/29587984/4565794
  //
  // permute n 32-bit rows
  // permute n 64-bit rows
  // ...
  // permute n simd_width/2-bit rows

  // clang-format off
    
    // Permute 8 8-bit rows.
    // Load int8x16x2 from memory into SIMD registers, transpose as 2x2 matrices.

    Index srcRowBegin = i*cols + j;
    int8x16x2_t r0 = vtrnq_s8(vld1q_s8(&src[ 0*cols + srcRowBegin]), vld1q_s8(&src[ 1*cols + srcRowBegin]));
    int8x16x2_t r1 = vtrnq_s8(vld1q_s8(&src[ 2*cols + srcRowBegin]), vld1q_s8(&src[ 3*cols + srcRowBegin]));
    int8x16x2_t r2 = vtrnq_s8(vld1q_s8(&src[ 4*cols + srcRowBegin]), vld1q_s8(&src[ 5*cols + srcRowBegin]));
    int8x16x2_t r3 = vtrnq_s8(vld1q_s8(&src[ 6*cols + srcRowBegin]), vld1q_s8(&src[ 7*cols + srcRowBegin]));
    int8x16x2_t r4 = vtrnq_s8(vld1q_s8(&src[ 8*cols + srcRowBegin]), vld1q_s8(&src[ 9*cols + srcRowBegin]));
    int8x16x2_t r5 = vtrnq_s8(vld1q_s8(&src[10*cols + srcRowBegin]), vld1q_s8(&src[11*cols + srcRowBegin]));
    int8x16x2_t r6 = vtrnq_s8(vld1q_s8(&src[12*cols + srcRowBegin]), vld1q_s8(&src[13*cols + srcRowBegin]));
    int8x16x2_t r7 = vtrnq_s8(vld1q_s8(&src[14*cols + srcRowBegin]), vld1q_s8(&src[15*cols + srcRowBegin]));


    // Permute 8 16-bit rows.
    // Next step is to treat the entries as int16x8x2 (via cast) and do
    // transpose for int16, which will now leave intra-2 pairs intact while
    // transposing inter 2-pairs into the right places.
    int16x8x2_t t0 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[0]), vreinterpretq_s16_s8(r1.val[0]));
    int16x8x2_t t1 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[0]), vreinterpretq_s16_s8(r3.val[0]));
    int16x8x2_t t2 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[0]), vreinterpretq_s16_s8(r5.val[0]));
    int16x8x2_t t3 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[0]), vreinterpretq_s16_s8(r7.val[0]));
    int16x8x2_t t4 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[1]), vreinterpretq_s16_s8(r1.val[1]));
    int16x8x2_t t5 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[1]), vreinterpretq_s16_s8(r3.val[1]));
    int16x8x2_t t6 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[1]), vreinterpretq_s16_s8(r5.val[1]));
    int16x8x2_t t7 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[1]), vreinterpretq_s16_s8(r7.val[1]));

    // Permute 8 32-bit rows.
    int32x4x2_t x0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t x1 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[0]), vreinterpretq_s32_s16(t5.val[0]));
    int32x4x2_t x2 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
    int32x4x2_t x3 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[1]), vreinterpretq_s32_s16(t5.val[1]));

    int32x4x2_t x4 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
    int32x4x2_t x5 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[0]), vreinterpretq_s32_s16(t7.val[0]));
    int32x4x2_t x6 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
    int32x4x2_t x7 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[1]), vreinterpretq_s32_s16(t7.val[1]));

    // There is no permute 8 64-bit rows available. 
    // Instead we follow extracting low and high and placing them into the right places.
    Index dstRowBegin = j*rows + i;
    vst1q_s8(&dst[ 0*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[0]),  vget_low_s32(x4.val[0])))); 
    vst1q_s8(&dst[ 1*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[0]),  vget_low_s32(x5.val[0]))));
    vst1q_s8(&dst[ 2*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[0]),  vget_low_s32(x6.val[0]))));
    vst1q_s8(&dst[ 3*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[0]),  vget_low_s32(x7.val[0]))));
    vst1q_s8(&dst[ 4*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[1]),  vget_low_s32(x4.val[1]))));
    vst1q_s8(&dst[ 5*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[1]),  vget_low_s32(x5.val[1]))));
    vst1q_s8(&dst[ 6*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[1]),  vget_low_s32(x6.val[1]))));
    vst1q_s8(&dst[ 7*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[1]),  vget_low_s32(x7.val[1]))));

    vst1q_s8(&dst[ 8*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[0]), vget_high_s32(x4.val[0]))));
    vst1q_s8(&dst[ 9*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[0]), vget_high_s32(x5.val[0]))));
    vst1q_s8(&dst[10*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[0]), vget_high_s32(x6.val[0]))));
    vst1q_s8(&dst[11*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[0]), vget_high_s32(x7.val[0]))));
    vst1q_s8(&dst[12*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[1]), vget_high_s32(x4.val[1]))));
    vst1q_s8(&dst[13*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[1]), vget_high_s32(x5.val[1]))));
    vst1q_s8(&dst[14*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[1]), vget_high_s32(x6.val[1]))));
    vst1q_s8(&dst[15*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[1]), vget_high_s32(x7.val[1]))));

  // clang-format on
}

// Specialization for int8_t
inline void transpose(const int8_t *input, Index rows, Index cols, int8_t *output) {
  constexpr Index tile_size = 16;
  // assert(rows % tile_size == 0 && cols & tile_size == 0);
  for(Index i = 0; i < rows; i += tile_size) {
    for(Index j = 0; j < cols; j += tile_size) {
      _transpose_16x16(input, i, j, rows, cols, output);
    }
  }
}

#endif

/*
 * The following nomenclature comes from intgemm. The current state of code is to keep the
 * intgemm_interface.h diff minimal. There are possibly better abstractions.
 */

    static void SelectColumnsB(const int8_t *input,
                               int8_t *output,
                               Index width,
                               const Index *cols,
                               const Index *cols_end) {
      // B_prepared is expected to be col-major, for our implementation via ruy. If
      // col-major we can memcpy the respective column entries as they're
      // sequential. There are width=rows entries.
      Index num_cols = static_cast<Index>(std::distance(cols, cols_end));
      for(Index c = 0; c < num_cols; ++c) {
        std::memcpy(&(output[c * width]), &(input[cols[c] * width]), width);
      }
    }


    static void Multiply8Rui(const int8_t *input_A_prepared,
                         const int8_t *input_B_prepared,
                         float *output,
                         Index rows_A,
                         Index width,
                         Index cols_B,
                         float unquant_multiplier,
                         float * bias=nullptr) {
      // It is expected that somehow we have managed to call all prepare by the time
      // we are here, with inputs (prepared) in int8_t. All that's left to do is use
      // ruy for multiply and then start with the reverse ops to get to fp32.

      // Use ruy to multiply.
      // The following is adapted from
      // https://github.com/google/ruy/blob/878283640de7946a43053e8ebf4f15114fbc9156/example/example.cc#L129-L152

      ruy::Context context;
      ruy::Matrix<std::int8_t> lhs;
      ruy::MakeSimpleLayout(rows_A, width, ruy::Order::kRowMajor, lhs.mutable_layout());
      lhs.set_data(input_A_prepared);

      ruy::Matrix<std::int8_t> rhs;
      ruy::MakeSimpleLayout(width, cols_B, ruy::Order::kColMajor, rhs.mutable_layout());
      rhs.set_data(input_B_prepared);

      ruy::Matrix<std::int32_t> dst;
      ruy::MakeSimpleLayout(rows_A, cols_B, ruy::Order::kRowMajor, dst.mutable_layout());

      std::int32_t *dest_ptr = reinterpret_cast<std::int32_t *>(output);
      dst.set_data(dest_ptr);

      // When Dst is int32, mul_params is unused.
      ruy::MulParams<std::int32_t, std::int32_t> mul_params;
      ruy::Mul(lhs, rhs, mul_params, &context, &dst);

      // Unquantise:
      float32x4_t multiplier = vdupq_n_f32(unquant_multiplier);

      if (!bias) {
        for (Index i = 0; i < rows_A*cols_B; i+=4) {
          const int32x4_t *in = reinterpret_cast<const int32x4_t *>(&output[i]);
          float32x4_t *out = reinterpret_cast<float32x4_t *>(&output[i]);
          *out = vmulq_f32(vcvtq_f32_s32(*in), multiplier);
        }
      } else {
        for (Index i = 0; i < rows_A; i++) {
          for (Index j = 0; j < cols_B; j+=4) {
            const float32x4_t *mybias = reinterpret_cast<const float32x4_t *>(&bias[j]);
            const int32x4_t *in = reinterpret_cast<const int32x4_t *>(&output[i*cols_B + j]);
            float32x4_t *out = reinterpret_cast<float32x4_t *>(&output[i*cols_B + j]);
            *out = vaddq_f32(vmulq_f32(vcvtq_f32_s32(*in), multiplier), *mybias);
          }
        }
      }
    }

  static float MaxAbsolute(const float *begin, const float *end) {
    float result = 0;
    for(auto p = begin; p < end; ++p) {
      result = std::max(result, std::abs(*p));
    }
    return result;
  }


struct PrepareNode : public NaryNodeOp {
float quantMult_;
bool isB_;
bool transpose_;
  PrepareNode(Expr input, Expr quant_mult, bool transposeme=false, bool isB=false)
      : NaryNodeOp({input, quant_mult}, newShape(input, transposeme), Type::intgemm8), isB_(isB),
      // Since we are doing RowM x ColM, we should ALWAYS transpose B when prepare
      // UNLESS it's already transposed (eg the output layer).
      // On the other hand transposing A should work as normal
      transpose_(isB_ ? !transposeme : transposeme) {

    setMemoize(isB_); // Only Memoise if this is a BNodeOp
    set_name(input->name());

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return { [=]() {
      quantMult_ = *child(1)->val()->data();
      quantize(child(0)->val()->data(), /*input*/
                val_->data<int8_t>(), /*output*/
                *child(1)->val()->data(), /*Quant Mult*/
                rows(child(0)->val()),
                cols(child(0)->val()));

      if (transpose_) {
        int myrows = rows(child(0)->val());
        int mycols = cols(child(0)->val());
        auto allocator = New<TensorAllocator>(graph()->getBackend());

        Tensor temp;
        allocator->allocate(temp, shape(), Type::int8);

        transpose(val_->data<int8_t>(), myrows, mycols, temp->data<int8_t>());
        std::memcpy(val_->data<int8_t>(), temp->data<int8_t>(), sizeof(int8_t)*myrows*mycols);
        allocator->free(temp);
      }
    }};
  }

  static Shape newShape(Expr input, bool transposed) {
    Shape ret = input->shape();
    if (transposed) {
      ret.set(0, input->shape()[-1]);
      ret.set(1, input->shape()[0]);
    } else {
      ret = input->shape();
    }
    return ret;
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isB_){
      return "RuyPrepareB";
    } else {
      return "RuyPrepareA";
    }
  }
};

struct SelectColumnsBRuyNodeOp : public UnaryNodeOp {
public:
  float quantMult_;
  SelectColumnsBRuyNodeOp(Expr input, const std::vector<uint_least32_t>  &indices)
      : UnaryNodeOp(input, newShape(input, indices), input->value_type()),  indices_(indices) {

    set_name(input->name());
    setMemoize(false); // Enabling memoization leads to a massive memory leak.
                       // Still, I don't understand why setMemoize(true) still leaks.
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");
  }

  NodeOps forwardOps() override {
    return { [=]() {
      //We get the quantization multiplier from a PrepareB or directly from the input
      if (child(0)->type() == "RuyPrepareB") {
        auto bPreppedNode = std::static_pointer_cast<PrepareNode>(child(0));
        quantMult_ = bPreppedNode->quantMult_;
      } else {
        ABORT("We should not be here");
      }
      auto input = child(0)->val();
      SelectColumnsB(reinterpret_cast<int8_t *>(input->data()),
                    val_->data<int8_t>(),
                    rows(input),
                    indices_.data(),
                    indices_.data()+indices_.size());
    }};
  }

  const std::string type() override { return "RuySelectColumnsB"; }

  size_t hash() override {
    if (!hash_) {
      hash_ = NaryNodeOp::hash();
      for(auto i : indices_)
        util::hash_combine(hash_, i);
    }
    return hash_;
  }

  bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node)) return false;
    auto cnode = std::dynamic_pointer_cast<SelectColumnsBRuyNodeOp>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<uint_least32_t>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<uint_least32_t> indices_;
};

struct QuantMultRuyNodeOp : public UnaryNodeOp {
  bool isB_;
  QuantMultRuyNodeOp(Expr input, bool isB, std::string bname) : UnaryNodeOp(input, Shape({1}), Type::float32), isB_(isB) {
    if (isB_) {
      set_name(input->name() + "_QuantMultB");
    } else {
      setMemoize(false);
      set_name(bname + "_QuantMultA");
    }
  }
#pragma warning(push)
#pragma warning(disable: 4127) //VSCODE thinks line 222 is constant conditional expression, which it is only after the template resolution, not before.
  NodeOps forwardOps() override {
    return  {[=](){
      if (child(0)->type() == "RuySelectColumnsB") { // @TODO
        *val_->data() = std::static_pointer_cast<SelectColumnsBRuyNodeOp>(child(0))->quantMult_;
      } else if (isIntgemm(child(0)->value_type())) {                    /* So we can template*/
        *val_->data() = *(reinterpret_cast<float *>(reinterpret_cast<int8_t *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      } else {
        auto input = child(0)->val();
        *val_->data() = 127.0f / MaxAbsolute(input->data(), input->data() + input->size());
      }
    }};
  }
#pragma warning(pop)
  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isB_)
      return "RuyQuantMultB";
    else
      return "RuyQuantMultA";
  }

  /* @TODO This is not correct in the case of none-static alphas but we are leaving it for now to battle memory leaks. */
  bool equal(Expr node) override {
    if (!isB_) {
      return UnaryNodeOp::equal(node);
    }
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

};

class AffineOrDotNodeOp : public NaryNodeOp {

public:
  AffineOrDotNodeOp(Expr a, Expr b, Expr bias)
      : NaryNodeOp({a, b, bias}, newShape(a, b), Type::float32) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  AffineOrDotNodeOp(Expr a, Expr b)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }


  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return { [=]() {
          float aQuantMult = std::static_pointer_cast<PrepareNode >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "RuySelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBRuyNodeOp>(child(1))->quantMult_;
          } else if (child(1)->type() == "RuyPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareNode>(child(1))->quantMult_;
          } else {
            // Fetch the quant mult directly in case of already prepared B
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<int8_t *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);
          // The bias is either nullptr or the third child
          float * bias = nullptr;
          if (children().size() == 3) {
            bias = child(2)->val()->data();
          }
          Multiply8Rui(reinterpret_cast<const int8_t *>(child(0)->val()->data()), /*A*/
                                      reinterpret_cast<const int8_t *>(child(1)->val()->data()), /*B*/
                                      val_->data(), /*output*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()),
                                      cols(child(1)->val()),
                                      unquant_mult,
                                      bias);
    }};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};

static inline Expr affineOrDotRUI(Expr a, Expr b, Expr bias, bool transA, bool transB, float /*clipValue*/) {
    Type bElementType = b->value_type();
    Expr aQuantMult = nullptr;
    bool precomputedAlphas = b->graph()->getBackend()->isPrecomputedAlpha();
    if (precomputedAlphas) { //Shifting here maybe should check?
      aQuantMult = Expression<fetchAlphaFromModelNodeOp>(b);
    } else {
      aQuantMult = Expression<QuantMultRuyNodeOp>(a, false, b->name());
    }
    auto aQuant = Expression<PrepareNode>(a, aQuantMult, transA, false);
    Expr bQuantMult = Expression<QuantMultRuyNodeOp>(b, true, b->name());
    Expr bQuant = nullptr;
    if (isIntgemm(bElementType)) {
      //This is the case where we already run SelectColumnB or we loaded a prepacked model.
      bQuant = b;
    } else {
      bQuant = Expression<PrepareNode>(b, bQuantMult, transB, true);
    }
    if (bias) {
      return Expression<AffineOrDotNodeOp>(aQuant, bQuant, bias);
    } else {
      return Expression<AffineOrDotNodeOp>(aQuant, bQuant);
    }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
