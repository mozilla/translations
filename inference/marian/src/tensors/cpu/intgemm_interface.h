#pragma once
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

template<Type vtype>
struct PrepareANodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;
bool shifted_;
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue, bool shifted)
      : NaryNodeOp({input, quant_mult}, input->shape(), vtype), clipValue_(clipValue), shifted_(shifted) {

    if (!shifted) {
      set_name(input->name());
    } else {
      set_name(input->name() + "_shifted");
    }
    setMemoize(false);
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return { [=]() {
      quantMult_ = *child(1)->val()->data();
  #if defined(WASM)
      ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
        "Int16::PrepareA is not implemented for wasm.");
      ABORT_IF(!shifted_, "Int8::PrepareA is not implemented for wasm. Please use shifted version.");
      int8PrepareA(child(0)->val()->data(), // input
                  *child(1)->val()->data(), // Scale
                  0, // zero point
                  rows(child(0)->val()),
                  cols(child(0)->val()),
                  val_->data<int8_t>() /*output*/);
  #elif defined(USE_INTGEMM)
    typedef typename intgemm_<vtype>::type Integer;
      auto input = child(0)->val();
      if (!shifted_) {
        intgemm_<vtype>::width::PrepareA(input->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      } else {
        intgemm::Int8Shift::PrepareA(input->data(), /*input*/
                                      val_->data<int8_t>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      }
  #endif
  }};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareA"; }
};

template<Type vtype>
struct PrepareBNodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;
bool transposed_; /*This is only used for the output layer which has a different layout from
                   *all other matrices in the code. By default we shouldn't do an extra transpose.*/

  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue, bool transposed=false)
      : NaryNodeOp({input, quant_mult}, newShape(input, transposed), intgemm_<vtype>::intgemmType), clipValue_(clipValue), transposed_(transposed) {

    set_name(input->name());
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
    if (!transposed_) {
      ABORT_IF(input->shape()[-1] %8 != 0, "Columns of matrix: " + input->type() + " must be multiple of 8.");
    } else {
      ABORT_IF((input->shape().elements()/input->shape()[-1]) %8 != 0, "Rows of matrix: " + input->type() + " must be multiple of 8.");
    }
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
   return { [=]() {
      quantMult_ = *child(1)->val()->data();
      if (isIntgemm(child(0)->value_type())) {
        val_ = child(0)->val();
      } else {
#if defined(WASM)
        ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
                "Int16::PrepareB is not implemented for wasm.");
        if (!transposed_) {
          int8PrepareB(child(0)->val()->data(), //input
                      *child(1)->val()->data(), //Scale
                      0, //Zero point
                      rows(child(0)->val()), // width
                      cols(child(0)->val()), // cols_B
                      val_->data<int8_t>()); /*output*/
        } else {
          int8PrepareBFromTransposed(child(0)->val()->data(), //input
                               *child(1)->val()->data(), //Scale,
                               0, //Zero point
                               cols(child(0)->val()),
                               rows(child(0)->val()),
                               val_->data<int8_t>()); /*output*/
        }
#else
        if (!transposed_) {
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::PrepareB(child(0)->val()->data(), /*input*/
                                        val_->data<Integer>(), /*output*/
                                        *child(1)->val()->data(), /*Quant Mult*/
                                        rows(child(0)->val()),
                                        cols(child(0)->val()));
        } else {
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::PrepareBTransposed(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      cols(child(0)->val()), /*Cols and rows need to be swapped*/
                                      rows(child(0)->val())); /*Cols and rows need to be swapped*/
        }
#endif
      }
    }};
#else
   return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareB"; }

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
};

template<Type vtype>
struct SelectColumnsBNodeOp : public UnaryNodeOp {
public:
  float clipValue_;
  float quantMult_;
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices, float clipValue)
      : UnaryNodeOp(input, newShape(input, indices), intgemm_<vtype>::intgemmType), clipValue_(clipValue), indices_(indices) {

    set_name(input->name());
    setMemoize(false); // Enabling memoization leads to a massive memory leak. 
                       // Still, I don't understand why setMemoize(true) still leaks.
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check number of selected columns
    ABORT_IF(indices.size() % 8 != 0, "Shortlist selected vocabulary must be a multiple of 8.");
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return { [=]() {
      //We get the quantization multiplier from a PrepareB or directly from the input
      if (child(0)->type() == "intgemmPrepareB") {
        auto bPreppedNode = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(0));
        quantMult_ = bPreppedNode->quantMult_;
      } else {
        typedef typename intgemm_<vtype>::type Integer;
        quantMult_ = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      }
      auto input = child(0)->val();
  #if defined(WASM)
      ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
                "Int16::SelectColumnsB is not implemented for wasm.");
      Index num_cols = std::distance(indices_.begin(), indices_.end());
      int8SelectColumnsOfB(reinterpret_cast<int8_t *>(input->data()),
                    rows(input),
                    cols(input),
                    &*indices_.begin(),
                    num_cols,
                    val_->data<int8_t>());
  #else
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::SelectColumnsB(
                    reinterpret_cast<Integer *>(input->data()),
                    val_->data<Integer>(),
                    rows(input),
                    indices_.data(),
                    indices_.data()+indices_.size());
  #endif
    }};
#else
    return {NodeOp()};
#endif
  }

  const std::string type() override { return "intgemmSelectColumnsB"; }

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
    auto cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp<vtype>>(node);
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

template<Type vtype>
struct QuantMultNodeOp : public UnaryNodeOp {
  bool isA_;
  QuantMultNodeOp(Expr input, bool isA, std::string& bname) : UnaryNodeOp(input, Shape({1}), Type::float32), isA_(isA) {
    if (isA_) {
      setMemoize(false);
      set_name(bname + "_QuantMultA");
    } else {
      set_name(input->name() + "_QuantMultB");
    }
  }
#pragma warning(push)
#pragma warning(disable: 4127) //VSCODE thinks line 222 is constant conditional expression, which it is only after the template resolution, not before.
  NodeOps forwardOps() override {
    return  {[=](){
    #ifdef COMPILE_CPU
      if (vtype == Type::int16) {
        *val_->data() = 1024.0f;
      } else if (child(0)->type() == "intgemmSelectColumnsB") {
        *val_->data() = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(0))->quantMult_;
      } else if (isIntgemm(child(0)->value_type())) {                    /* So we can template*/
        typedef typename intgemm_<vtype>::type Integer;
        *val_->data() = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      } else {
        if (child(0)->graph()->getBackend()->DumpQuantMult()) {
          #if defined(USE_INTGEMM)
          intgemm::MeanStd meanstd = intgemm::GetVectorMeanStd(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements(), true);
          intgemm::MeanStd meanstd2 = intgemm::GetVectorMeanStd(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements());
          std::cerr << "Name: " << name() << " MeanAbs: " << meanstd.mean << " stddevAbs: " << meanstd.stddev << " Mean: " << meanstd2.mean << " stddev: "
          << meanstd2.stddev << " MaxAbs: " << intgemm::MaxAbsolute(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements()) << std::endl;
          #endif
        }
        auto input = child(0)->val();
        #if defined(USE_INTGEMM)
          *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->size());
        #endif
      }
    #endif // COMPILE_CPU
    }};
  }
#pragma warning(pop)
  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isA_)
      return "intgemmQuantMultA";
    else
      return "intgemmQuantMultB";
  }

  /* @TODO This is not correct in the case of none-static alphas but we are leaving it for now to battle memory leaks. */
  bool equal(Expr node) override {
    if (isA_) {
      return UnaryNodeOp::equal(node);
    }
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

};

class PrepareBiasForBNodeOp : public NaryNodeOp {
  bool alreadyPrepared_ = false;
public:
  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : NaryNodeOp({bias, inputB_preppd, a_quant_mult, b_quant_mult}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    if (bias->type() == "cols" && bias->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We shouldn't ever be here");
      alreadyPrepared_ = true;
    } else if (!bias->graph()->getBackend()->isPrecomputedAlpha()){
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    //std::cerr << "TrueBias: " << child(0)->name() << " type: " << child(0)->type() << " bQuantMult: " << this->child(3)->val()->data()[0] <<  " aQuantMult: " << this->child(2)->val()->data()[0] << std::endl;
    //std::cerr << "Bias name and val: " << child(0)->name() << " " << child(0)->val()->data()[0] << std::endl;
#ifdef COMPILE_CPU
    return { [=]() {
      if (alreadyPrepared_) {
        //God Knows why trying to assign the bias tensor to this node causes a crash, the second time it's referenced
        //even though it's supposed to work fine. We use a memory copy instead.
        ABORT("We shouldn't ever be here.");
        std::memcpy(val_->data(), child(0)->val()->data(), child(0)->shape()[-1]*sizeof(float));
        //val_ = child(0)->val();
      } else {
        auto bias = this->child(0)->val();
        auto b = this->child(1)->val();
        auto quant_mult_a = this->child(2)->val();
        auto quant_mult_b = this->child(3)->val();

    #if defined(WASM)
        float scale_a = *quant_mult_a->data();
        float scale_b = *quant_mult_b->data();
        int8PrepareBias((const int8_t *)b->data(), scale_a, 0.0 /*zero_point_a*/, scale_b, 0.0 /*zero_point_b*/, rows(b), cols(b), bias->data(), val_->data());
    #elif defined(USE_INTGEMM)
        float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
        intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias->data(), val_->data()));
    #else
        ABORT("PrepareBias should not be called on ARM");
    #endif
      }
    }};
#else
    return {NodeOps()};
#endif
  }

  const std::string type() override { return "prepareBias"; }
};

class PrepareFakeBiasForBNodeOp : public NaryNodeOp {
public:
  PrepareFakeBiasForBNodeOp(Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : NaryNodeOp({inputB_preppd, a_quant_mult, b_quant_mult}, {1, inputB_preppd->shape()[-1]}, Type::float32) {

    set_name(inputB_preppd->name() + "_FakeBias");
    if (!inputB_preppd->graph()->getBackend()->isPrecomputedAlpha()) {
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    //std::cerr << "FakeBias: " << child(0)->name() << " bQuantMult: " << this->child(2)->val()->data()[0] << " aQuantMult: " << this->child(1)->val()->data()[0] << std::endl;
#ifdef COMPILE_CPU
    return { [=]() {
    auto b = this->child(0)->val();
    auto quant_mult_a = this->child(1)->val();
    auto quant_mult_b = this->child(2)->val();

  #if defined(WASM)
    float scale_a = *quant_mult_a->data();
    float scale_b = *quant_mult_b->data();
    int8PrepareBias((const int8_t *)b->data(), scale_a, 0.0 /*zero_point_a*/, scale_b, 0.0 /*zero_point_b*/, rows(b), cols(b), nullptr/*input_bias*/, val_->data());
  #elif defined(USE_INTGEMM)
    float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
    intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
  #else
    ABORT("PrepareBias should not be called on ARM");
  #endif
    }};
#else
    return {NodeOp()};
#endif
  }

  const std::string type() override { return "prepareFakeBias"; }
};

template<Type vtype>
class DotNodeOp : public NaryNodeOp {
private:
float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32), scalar_(scalar) {
        setMemoize(false); // AFAIK dot is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return { [=]() {
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
      #if defined(WASM)
          ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
              "Int16::Multiply is not implemented for wasm.");
          ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm8,
              "Int8::Multiply is not implemented for wasm.");
      #elif defined(USE_INTGEMM)
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                           reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                           rows(child(0)->val()),
                                           cols(child(0)->val()),
                                           cols(child(1)->val()),
                                           intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
      #endif
    }};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmDot"; }
};

template<Type vtype>
class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool shifted_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, float scalar, bool shifted=false)
      : NaryNodeOp({a, b, Bias}, newShape(a, b), Type::float32), scalar_(scalar), shifted_(shifted) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return { [=]() {
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
      #if defined(WASM)
          ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
            "Int16::Multiply is not implemented for wasm.");
          ABORT_IF(!shifted_, "Int8::Multiply is not implemented for wasm.");

          int8MultiplyAndAddBias(reinterpret_cast<int8_t *>(child(0)->val()->data()), /*A*/
                                aQuantMult, /*Scale of A*/
                                0, /*zero point of A*/
                                reinterpret_cast<int8_t *>(child(1)->val()->data()), /*B*/
                                bQuantMult, /*Scale of B*/
                                0, /*zero point of B*/
                                child(2)->val()->data(), /*child(2) is bias*/
                                scalar_,
                                rows(child(0)->val()),
                                cols(child(0)->val()),
                                cols(child(1)->val()),
                                val_->data());
      #elif defined(USE_INTGEMM)
          typedef typename intgemm_<vtype>::type Integer;
          if (!shifted_) {
            intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                             reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                             rows(child(0)->val()),
                                             cols(child(0)->val()),
                                             cols(child(1)->val()),                                          /*child(2) is bias*/
                                             intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          } else {
            intgemm::Int8Shift::Multiply(reinterpret_cast<int8_t *>(child(0)->val()->data()), /*A*/
                                  reinterpret_cast<int8_t *>(child(1)->val()->data()), /*B*/
                                  rows(child(0)->val()),
                                  cols(child(0)->val()),
                                  cols(child(1)->val()),                                          /*child(2) is bias*/
                                  intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          }
      #endif
    }};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};

template<Type vtype>
static inline Expr quantMult(Expr a, bool isA=false, std::string Bname="") {
  return Expression<QuantMultNodeOp<vtype> >(a, isA, Bname);
}

template<Type vtype>
static inline Expr prepareA(Expr a, Expr quantMult, float clipValue, bool shifted=false) {
  return Expression<PrepareANodeOp<vtype> >(a, quantMult, clipValue, shifted);
}

template<Type vtype>
static inline Expr prepareB(Expr b, Expr quantMult, float clipValue, bool transposed=false) {
  return Expression<PrepareBNodeOp<vtype> >(b, quantMult, clipValue, transposed);
}

template<Type vtype>
static inline Expr selectColumnsB(Expr b, const std::vector<uint_least32_t> &cols, float clipValue) {
  return Expression<SelectColumnsBNodeOp<vtype > >(b, cols, clipValue);
}

template<Type vtype>
static inline Expr affine(Expr a, Expr b, Expr bias, bool transA, bool transB, float scale, float /* clipValue currently unused */ = 0.0f, bool shiftedBias=false) {
  Type bElementType = b->value_type();
  Expr aQuantMult = nullptr;
  bool precomputedAlphas = b->graph()->getBackend()->isPrecomputedAlpha();
  if (precomputedAlphas) { //Shifting here maybe should check?
    aQuantMult = Expression<fetchAlphaFromModelNodeOp>(b);
  } else {
    aQuantMult = quantMult<vtype>(a, true, b->name()); /*@TODO Do something about b->name() here*/
  }
  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a, aQuantMult, scale, shiftedBias);
  Expr bQuantMult = quantMult<vtype>(b);
  Expr bQuant = nullptr;
  if (isIntgemm(bElementType)) {
    //This is the case where we already run SelectColumnB or we loaded a prepacked model.
    bQuant = b;
  } else {
    bQuant = prepareB<vtype>(b, bQuantMult, scale, transB);
  }
  if (bias && precomputedAlphas && bias->name() == "none") {
    // This is the case of the preprocessed bias. It's hacky but otherwise node caching is broken.
    // The bias node is the shortlisted bias and it has been prepared before index_select was run
  } else if (shiftedBias && bias) {
    bias = Expression<PrepareBiasForBNodeOp>(bias, bQuant, aQuantMult, bQuantMult);
  } else if (shiftedBias) {
    bias = Expression<PrepareFakeBiasForBNodeOp>(bQuant, aQuantMult, bQuantMult);
  }

  if (bias) {
    return Expression<AffineNodeOp<vtype> >(aQuant, bQuant, bias, scale, shiftedBias);
  } else {
    return Expression<DotNodeOp<vtype> >(aQuant, bQuant, scale);
  }
}

template<Type vtype>
static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale, bool shiftedBias=false) {
  return affine<vtype>(a, b, nullptr, transA, transB, scale, 0 /*currently unused clipValue*/, shiftedBias);
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
