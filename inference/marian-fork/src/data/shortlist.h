#pragma once

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/corpus_base.h"
#include "data/types.h"
#include "3rd_party/mio/mio.hpp"

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>

namespace marian {
namespace data {
// Magic signature for binary shortlist:
// ASCII and Unicode text files never start with the following 64 bits
const uint64_t BINARY_SHORTLIST_MAGIC = 0xF11A48D5013417F5;

bool isBinaryShortlist(const std::string& fileName);

class Shortlist {
private:
  std::vector<WordIndex> indices_;    // // [packed shortlist index] -> word index, used to select columns from output embeddings

public:
  Shortlist(const std::vector<WordIndex>& indices)
    : indices_(indices) {}

  const std::vector<WordIndex>& indices() const { return indices_; }
  WordIndex reverseMap(int idx) { return indices_[idx]; }

  int tryForwardMap(WordIndex wIdx) {
    auto first = std::lower_bound(indices_.begin(), indices_.end(), wIdx);
    if(first != indices_.end() && *first == wIdx)         // check if element not less than wIdx has been found and if equal to wIdx
      return (int)std::distance(indices_.begin(), first); // return coordinate if found
    else
      return -1;                                          // return -1 if not found
  }
};

class ShortlistGenerator {
public:
  virtual ~ShortlistGenerator() {}

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const = 0;

  // Writes text version of (possibly) pruned short list to file
  // with given prefix and implementation-specific suffixes.
  virtual void dump(const std::string& /*prefix*/) const {
    ABORT("Not implemented");
  }
};


// Intended for use during training in the future, currently disabled
#if 0
class SampledShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  size_t maxVocab_{50000};

  size_t total_{10000};
  size_t firstNum_{1000};

  size_t srcIdx_;
  size_t trgIdx_;
  bool shared_{false};

  // static thread_local std::random_device rd_;
  static thread_local std::unique_ptr<std::mt19937> gen_;

public:
  SampledShortlistGenerator(Ptr<Options> options,
                            size_t srcIdx = 0,
                            size_t trgIdx = 1,
                            bool shared = false)
      : options_(options),
        srcIdx_(srcIdx),
        trgIdx_(trgIdx),
        shared_(shared)
        { }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override {
    auto srcBatch = (*batch)[srcIdx_];
    auto trgBatch = (*batch)[trgIdx_];

    // add firstNum most frequent words
    std::unordered_set<WordIndex> indexSet;
    for(WordIndex i = 0; i < firstNum_ && i < maxVocab_; ++i)
      indexSet.insert(i);

    // add all words from ground truth
    for(auto i : trgBatch->data())
      indexSet.insert(i.toWordIndex());

    // add all words from source
    if(shared_)
      for(auto i : srcBatch->data())
        indexSet.insert(i.toWordIndex());

    std::uniform_int_distribution<> dis((int)firstNum_, (int)maxVocab_);
    if (gen_ == NULL)
      gen_.reset(new std::mt19937(std::random_device{}()));
    while(indexSet.size() < total_ && indexSet.size() < maxVocab_)
      indexSet.insert(dis(*gen_));

    // turn into vector and sort (selected indices)
    std::vector<WordIndex> idx(indexSet.begin(), indexSet.end());
    std::sort(idx.begin(), idx.end());

    // assign new shifted position
    std::unordered_map<WordIndex, WordIndex> pos;
    std::vector<WordIndex> reverseMap;

    for(WordIndex i = 0; i < idx.size(); ++i) {
      pos[idx[i]] = i;
      reverseMap.push_back(idx[i]);
    }

    Words mapped;
    for(auto i : trgBatch->data()) {
      // mapped postions for cross-entropy
      mapped.push_back(Word::fromWordIndex(pos[i.toWordIndex()]));
    }

    return New<Shortlist>(idx, mapped, reverseMap);
  }
};
#endif

class LexicalShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  Ptr<const Vocab> srcVocab_;
  Ptr<const Vocab> trgVocab_;

  size_t srcIdx_;
  bool shared_{false};

  size_t firstNum_{100};
  size_t bestNum_{100};

  std::vector<std::unordered_map<WordIndex, float>> data_; // [WordIndex src] -> [WordIndex tgt] -> P_trans(tgt|src) --@TODO: rename data_ accordingly

  void load(const std::string& fname);
  void prune(float threshold = 0.f);

public:
  LexicalShortlistGenerator(Ptr<Options> options,
                            Ptr<const Vocab> srcVocab,
                            Ptr<const Vocab> trgVocab,
                            size_t srcIdx = 0,
                            size_t /*trgIdx*/ = 1,
                            bool shared = false);

  virtual void dump(const std::string& prefix) const override;
  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override;
};

class BinaryShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  Ptr<const Vocab> srcVocab_;
  Ptr<const Vocab> trgVocab_;

  size_t srcIdx_;
  bool shared_{false};

  uint64_t firstNum_{100};  // baked into binary header
  uint64_t bestNum_{100};   // baked into binary header

  // shortlist is stored in a skip list
  // [&shortLists_[wordToOffset_[word]], &shortLists_[wordToOffset_[word+1]])
  // is a sorted array of word indices in the shortlist for word
  mio::mmap_source mmapMem_;
  uint64_t wordToOffsetSize_;
  uint64_t shortListsSize_;
  const uint64_t *wordToOffset_;
  const WordIndex *shortLists_;
  std::vector<char> blob_;  // binary blob

  struct Header {
    uint64_t magic; // BINARY_SHORTLIST_MAGIC
    uint64_t checksum; // util::hashMem<uint64_t, uint64_t> from &firstNum to end of file.
    uint64_t firstNum; // Limits used to create the shortlist.
    uint64_t bestNum;
    uint64_t wordToOffsetSize; // Length of wordToOffset_ array.
    uint64_t shortListsSize; // Length of shortLists_ array.
  };

  void contentCheck();
  // load shortlist from buffer
  void load(const void* ptr_void, size_t blobSize, bool check = true);
  // load shortlist from file
  void load(const std::string& filename, bool check=true);
  // import text shortlist from file
  void import(const std::string& filename, double threshold);
  // save blob to file (called by dump)
  void saveBlobToFile(const std::string& filename) const;

public:
  BinaryShortlistGenerator(Ptr<Options> options,
                           Ptr<const Vocab> srcVocab,
                           Ptr<const Vocab> trgVocab,
                           size_t srcIdx = 0,
                           size_t /*trgIdx*/ = 1,
                           bool shared = false);

  // construct directly from buffer
  BinaryShortlistGenerator(const void* ptr_void,
                           const size_t blobSize,
                           Ptr<const Vocab> srcVocab,
                           Ptr<const Vocab> trgVocab,
                           size_t srcIdx = 0,
                           size_t /*trgIdx*/ = 1,
                           bool shared = false, // Kept there for backward compatibility
                           bool check = true);

  ~BinaryShortlistGenerator(){
    mmapMem_.unmap();
  }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override;
  virtual void dump(const std::string& fileName) const override;
};

class FakeShortlistGenerator : public ShortlistGenerator {
private:
  std::vector<WordIndex> indices_;

public:
  FakeShortlistGenerator(const std::unordered_set<WordIndex>& indexSet)
      : indices_(indexSet.begin(), indexSet.end()) {
    std::sort(indices_.begin(), indices_.end());
  }

  Ptr<Shortlist> generate(Ptr<data::CorpusBatch> /*batch*/) const override {
    return New<Shortlist>(indices_);
  }
};

}  // namespace data
}  // namespace marian
