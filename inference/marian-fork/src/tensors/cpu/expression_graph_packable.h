#pragma once

#include "graph/expression_graph.h"
#include "fbgemm/packed_gemm.h"
#include "tensors/cpu/integer_common.h"

namespace { //Temporary annonymous transposition, until we figure out how to access the proper one
inline void transpose4x4_SSE(const float* A,
                             float* B,
                             const int lda,
                             const int ldb) {
  __m128 row1 = _mm_load_ps(&A[0 * lda]);
  __m128 row2 = _mm_load_ps(&A[1 * lda]);
  __m128 row3 = _mm_load_ps(&A[2 * lda]);
  __m128 row4 = _mm_load_ps(&A[3 * lda]);
  _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
  _mm_store_ps(&B[0 * ldb], row1);
  _mm_store_ps(&B[1 * ldb], row2);
  _mm_store_ps(&B[2 * ldb], row3);
  _mm_store_ps(&B[3 * ldb], row4);
}

// from
// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
#define ROUND_UP(x, s) (((x) + ((s)-1)) & -(s))

void Transpose10(marian::Tensor out, const marian::Tensor in) {
  const float* A = in->data();
  float* B = out->data();

  const int n = in->shape().elements() / in->shape()[-1];
  const int m = in->shape()[-1];

  const int block_size = 16;
  int lda = ROUND_UP(m, block_size);
  int ldb = ROUND_UP(n, block_size);

  for(int i = 0; i < n; i += block_size) {
    for(int j = 0; j < m; j += block_size) {
      int max_i2 = i + block_size < n ? i + block_size : n;
      int max_j2 = j + block_size < m ? j + block_size : m;
      for(int i2 = i; i2 < max_i2; i2 += 4) {
        for(int j2 = j; j2 < max_j2; j2 += 4) {
          transpose4x4_SSE(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
        }
      }
    }
  }
}
}

namespace marian {


// When FBGEMM based packed GEMM is used, some weight matrices need to be packed offline.
// The decision which weights can be packed or not should be done walking through the graph.
// This requires some more changes, but we temporarily do this just by name ("_W") of the weights.
// And, this introduces a low level packed_gemm.h apis interact with high level graph class.
// So, we make a subclass of ExpressionGraph and put those immature codes in this class.
// We will improve this in the near future. 
class ExpressionGraphPackable : public ExpressionGraph {
public:
  bool compressWemb = true;
  ExpressionGraphPackable()
    : ExpressionGraph( /* inference =  */ true) {} // Packable expression graph only supports inference

  virtual ~ExpressionGraphPackable() {}

  // Convert model weights into packed format and save to IO items.
  // @TODO: review this
  void packAndSave(const std::string& name, const std::string& meta, Type gemmElementType = Type::float32, Type saveElementType = Type::float32) {
    std::vector<io::Item> ioItems;

    // sorted by name in std::map
    for (auto p : params()->getMap()) {
      std::string pName = p.first;

      if (!namespace_.empty()) {
        if (pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
          pName = pName.substr(namespace_.size() + 2);
      }

      Tensor val = p.second->val();

      // save as packed format
      // @TODO Hardcoded to find packable weights
      // int8 - all the weights used for affine op and dot op
      // fp16 - all the weights used for affine op
      if ((gemmElementType == Type::packed8avx2 || gemmElementType == Type::packed8avx512)
        && (pName.find("_W") == pName.length() - 3 || pName.find("_W") == pName.length() - 2)) {
#if USE_FBGEMM
        using namespace marian::cpu::variant;
        // packing information - size
        int nrow;
        int ncol;
        uint64_t packsize;

        fbgemmPacked8PackInfo(val->shape(),
                              gemmElementType,
                              pName.find("Wemb") != std::string::npos,
                              nrow,
                              ncol,
                              packsize);

        auto allocator = New<TensorAllocator>(getBackend());

        // buffer tensor to save packed matrix
        Tensor packedTensor;
        allocator->allocate(packedTensor, { 1, (int32_t)packsize }, Type::uint8);

        //Pack B matrix into int8
        fbgemmPacked8Pack(packedTensor,
                          val->data(),
                          gemmElementType,
                          pName.find("Wemb") != std::string::npos,
                          nrow,
                          ncol,
                          packsize);
        io::Item item;
        item.name = pName;
        item.shape = val->shape();
        item.type = gemmElementType;

        // Use the actual memory as this will be aligned and padded.
        // When memory mapping this is required. Shape keeps track of
        // tensor size. Saving to *.npz will cut to size.
        auto mem = packedTensor->memory();
        item.bytes.resize(mem->size());
        copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

        ioItems.emplace_back(std::move(item));
#else
        ABORT("Packed type {} only supported when compiled with -DUSE_FBGEMM=on", gemmElementType);
#endif
      // fp16 quantization option
      } else if (gemmElementType == Type::packed16 && pName.find("_W") == pName.length() - 3) {
#if USE_FBGEMM
        using namespace marian::cpu::variant;

        // packing information
        int nrow, ncol, kernel_ncol_blocks, brow, bcol, last_brow, nbrow, nbcol;
        uint64_t packsize;

        fbgemmPacked16PackInfo(val->shape(),
          false,
          nrow,
          ncol,
          kernel_ncol_blocks,
          brow,
          bcol,
          last_brow,
          nbrow,
          nbcol,
          packsize);

        auto allocator = New<TensorAllocator>(getBackend());

        Tensor packedTensor;
        allocator->allocate(packedTensor, { 1, (int32_t)packsize }, Type::uint8);

        // fbgemmPacked16Pack
        fbgemmPacked16Pack(packedTensor,
          val->data(),
          false,
          nrow,
          ncol,
          kernel_ncol_blocks,
          brow,
          bcol,
          last_brow,
          nbrow,
          nbcol,
          packsize);
        io::Item item;
        item.name = pName;
        item.shape = val->shape();
        item.type = Type::packed16;

        // Use the actual memory as this will be aligned and padded.
        // When memory mapping this is required. Shape keeps track of
        // tensor size. Saving to *.npz will cut to size.
        auto mem = packedTensor->memory();
        item.bytes.resize(mem->size());
        copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

        ioItems.emplace_back(std::move(item));
#else
        ABORT("Packed type {} only supported when compiled with -DUSE_FBGEMM=on", gemmElementType);
#endif
      } else if ((gemmElementType == Type::intgemm8 || gemmElementType == Type::intgemm16) &&
      (pName.find("_W") == pName.length() - 3 || pName.find("_W") == pName.length() - 2  || ((pName.find("Wemb") != std::string::npos) && compressWemb))) {
#ifdef COMPILE_CPU
        using cpu::integer::cols;
        using cpu::integer::rows;
        auto allocator = New<TensorAllocator>(getBackend());

        Tensor paramMat; //This allocates extra 4 bytes at the end because of gemmElementType
        allocator->allocate(paramMat, val->shape(), gemmElementType);

        // Compute QuantMultiplier, compress matrix and store quantMult at the end.
        // We need to tranpose first, because of our architecture independet format requiring a transposed matrix
        Tensor tmp;
        if (pName.find("Wemb") != std::string::npos) { //Do not transpose the Wemb matrix. Hacky temporary solution
          tmp = val;
        } else {
          allocator->allocate(tmp, val->shape(), val->type());
          Transpose10(tmp, val);
        }
        if (gemmElementType == Type::intgemm8) {
#if defined(WASM)
          ABORT("Int8::PrepareA is not implemented for wasm.");
#elif defined(USE_INTGEMM)
          float quantMult = 127.0f / intgemm::MaxAbsolute(val->data(), val->data() + val->shape().elements());
          intgemm::Int8::PrepareA(tmp->data(), /*input*/
                                paramMat->data<int8_t>(), /*output*/
                                quantMult, /*Quant Mult*/
                                rows(val),
                                cols(val));
          //Put the quantMult at the back of the tensor
          *(reinterpret_cast<float *>(paramMat->data<int8_t>() + val->shape().elements())) = quantMult;
#else
    ABORT("Int8::PrepareA not implemented yet for ruy");
#endif
        } else {
#if defined(WASM)
          ABORT("Int16::PrepareA is not implemented for wasm.");
#elif defined(USE_INTGEMM)
          float quantMult = 1024.0f;
          intgemm::Int16::PrepareA(tmp->data(), /*input*/
                                paramMat->data<int16_t>(), /*output*/
                                quantMult, /*Quant Mult*/
                                rows(val),
                                cols(val));
          //Put the quantMult at the back of the tensor
          *(reinterpret_cast<float *>(paramMat->data<int16_t>() + val->shape().elements())) = quantMult;
#else
      ABORT("Int16::PrepareA is not implemented for wasm.");
#endif
        }

        //Save... Same as the fbgemm case
        io::Item item;
        item.name = pName;
        item.shape = val->shape();
        item.type = gemmElementType;

        auto mem = paramMat->memory();
        item.bytes.resize(mem->size());
        copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());
        ioItems.emplace_back(std::move(item));
#else
ABORT("Packed type {} only supported when compiled with -COMPILE_CPU=on", gemmElementType);
#endif
      } else {
        io::Item item;
        val->get(item, pName);
        item.convert(saveElementType);
        ioItems.emplace_back(std::move(item));
      }
    }

    if (!meta.empty())
      io::addMetaToItems(meta, "special:model.yml", ioItems);
    io::saveItems(name, ioItems);
  }
};

}  // namespace marian