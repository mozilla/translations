#include "shortlist.h"
#include <queue>

namespace marian {
namespace data {

bool isBinaryShortlist(const std::string& fileName){
  uint64_t magic;
  io::InputFileStream in(fileName);
  in.read((char*)(&magic), sizeof(magic));
  return in && (magic == BINARY_SHORTLIST_MAGIC);
}

void LexicalShortlistGenerator::load(const std::string& fname) {
  io::InputFileStream in(fname);

  std::string src, trg;
  float prob;
  while(in >> trg >> src >> prob) {
    // @TODO: change this to something safer other than NULL
    if(src == "NULL" || trg == "NULL")
      continue;

    auto sId = (*srcVocab_)[src].toWordIndex();
    auto tId = (*trgVocab_)[trg].toWordIndex();

    data_.at(sId)[tId] = prob;
  }
}

void LexicalShortlistGenerator::prune(float threshold /* = 0.f*/) {
  for(auto& probs : data_) {
    std::vector<std::pair<float, WordIndex>> sorter;
    for(auto& it : probs)
      sorter.emplace_back(it.second, it.first);

    std::sort(
        sorter.begin(), sorter.end(), std::greater<std::pair<float, WordIndex>>()); // sort by prob

    probs.clear();
    for(auto& it : sorter) {
      if(probs.size() < bestNum_ && it.first > threshold)
        probs[it.second] = it.first;
      else
        break;
    }

  }
}

LexicalShortlistGenerator::LexicalShortlistGenerator(Ptr<Options> options,
                                                     Ptr<const Vocab> srcVocab,
                                                     Ptr<const Vocab> trgVocab,
                                                     size_t srcIdx /* = 0 */,
                                                     size_t /*trgIdx = 1 */,
                                                     bool shared /*= false*/)
    : options_(options),
      srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx),
      shared_(shared) {
  std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");

  ABORT_IF(vals.empty(), "No path to filter path given");
  std::string fname = vals[0];

  firstNum_ = vals.size() > 1 ? std::stoi(vals[1]) : 100;
  bestNum_ = vals.size() > 2 ? std::stoi(vals[2]) : 100;
  float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;
  std::string dumpPath = vals.size() > 4 ? vals[4] : "";
  LOG(info,
      "[data] Loading lexical shortlist as {} {} {} {}",
      fname,
      firstNum_,
      bestNum_,
      threshold);

  // Ensure that the data_ has enough entries for the src vocab.
  data_.resize(srcVocab_->size());
  
  // @TODO: Load and prune in one go.
  load(fname);
  prune(threshold);

  if(!dumpPath.empty())
    dump(dumpPath);
}

void LexicalShortlistGenerator::dump(const std::string& prefix) const {
  // Dump top most frequent words from target vocabulary
  LOG(info, "[data] Saving shortlist dump to {}", prefix + ".{top,dic}");
  io::OutputFileStream outTop(prefix + ".top");
  for(WordIndex i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
    outTop << (*trgVocab_)[Word::fromWordIndex(i)] << std::endl;

  // Dump translation pairs from dictionary
  io::OutputFileStream outDic(prefix + ".dic");
  for(WordIndex srcId = 0; srcId < data_.size(); srcId++) {
    for(auto& it : data_.at(srcId)) {
      auto trgId = it.first;
      outDic << (*srcVocab_)[Word::fromWordIndex(srcId)] << "\t" << (*trgVocab_)[Word::fromWordIndex(trgId)] << std::endl;
    }
  }
}

Ptr<Shortlist> LexicalShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  auto srcBatch = (*batch)[srcIdx_];

  // Add the most frequent words from the target vocab up to the "firstNum"
  // parameter.
  std::unordered_set<WordIndex> indexSet;
  for(WordIndex i = 0; i < firstNum_ && i < trgVocab_->size(); ++i) {
    indexSet.insert(i);
  }

  // Collect all of the unique words from the source batch. This is done in a
  // new unordered_set to deduplicate the list.
  std::unordered_set<WordIndex> srcSet;
  for(auto word : srcBatch->data()) {
    srcSet.insert(word.toWordIndex());
  }

  // Add the aligned target words from the source, and add the original source tokens
  // as well if the vocab is shared.
  for(auto srcIndex : srcSet) {
    if(shared_) {
      indexSet.insert(srcIndex);
    } else {
      // TODO - Shortlisting is not correct for split vocabs. Direct copy of tokens
      // from the src is not supported, as the source sentence needs to be tokenized
      // with the target vocab.
      // https://github.com/mozilla/translations/issues/1192
    }

    // Add all of the target probabilities.
    for(auto& it : data_.at(srcIndex)) {
      indexSet.insert(it.first);
    }
  }

  // Ensure that the generated vocabulary items from a shortlist are a multiple-of-eight
  // This is necessary until intgemm supports non-multiple-of-eight matrices.
  // TODO better solution here? This could potentially be slow.
  WordIndex i = static_cast<WordIndex>(firstNum_);
  while (indexSet.size() % 8 != 0) {
    indexSet.insert(i);
    i++;
  }

  // turn into vector and sort (selected indices)
  std::vector<WordIndex> indices(indexSet.begin(), indexSet.end());
  std::sort(indices.begin(), indices.end());

  return New<Shortlist>(indices);
}

void BinaryShortlistGenerator::contentCheck() {
  bool failFlag = 0;
  // The offset table has to be within the size of shortlists.
  for(int i = 0; i < wordToOffsetSize_-1; i++)
    failFlag |= wordToOffset_[i] >= shortListsSize_;

  // The last element of wordToOffset_ must equal shortListsSize_
  failFlag |= wordToOffset_[wordToOffsetSize_-1] != shortListsSize_;

  // The vocabulary indices have to be within the vocabulary size.
  size_t vSize = trgVocab_->size();
  for(int j = 0; j < shortListsSize_; j++)
    failFlag |= shortLists_[j] >= vSize;
  ABORT_IF(failFlag, "Error: shortlist indices are out of bounds");
}

// load shortlist from buffer
void BinaryShortlistGenerator::load(const void* ptr_void, size_t blobSize, bool check /*= true*/) {
  /* File layout:
   * header
   * wordToOffset array
   * shortLists array
   */
  ABORT_IF(blobSize < sizeof(Header), "Shortlist length {} too short to have a header", blobSize);

  const char *ptr = static_cast<const char*>(ptr_void);
  const Header &header = *reinterpret_cast<const Header*>(ptr);
  ptr += sizeof(Header);
  ABORT_IF(header.magic != BINARY_SHORTLIST_MAGIC, "Incorrect magic in binary shortlist");

  uint64_t expectedSize = sizeof(Header) + header.wordToOffsetSize * sizeof(uint64_t) + header.shortListsSize * sizeof(WordIndex);
  ABORT_IF(expectedSize != blobSize, "Shortlist header claims file size should be {} but file is {}", expectedSize, blobSize);

  if (check) {
    uint64_t checksumActual = util::hashMem<uint64_t, uint64_t>(&header.firstNum, (blobSize - sizeof(header.magic) - sizeof(header.checksum)) / sizeof(uint64_t));
    ABORT_IF(checksumActual != header.checksum, "checksum check failed: this binary shortlist is corrupted");
  }

  firstNum_ = header.firstNum;
  bestNum_ = header.bestNum;
  LOG(info, "[data] Lexical short list firstNum {} and bestNum {}", firstNum_, bestNum_);

  wordToOffsetSize_ = header.wordToOffsetSize;
  shortListsSize_ = header.shortListsSize;

  // Offsets right after header.
  wordToOffset_ = reinterpret_cast<const uint64_t*>(ptr);
  ptr += wordToOffsetSize_ * sizeof(uint64_t);

  shortLists_ = reinterpret_cast<const WordIndex*>(ptr);

  // Verify offsets and vocab ids are within bounds if requested by user.
  if(check)
    contentCheck();
}

// load shortlist from file
void BinaryShortlistGenerator::load(const std::string& filename, bool check /*=true*/) {
  std::error_code error;
  mmapMem_.map(filename, error);
  ABORT_IF(error, "Error mapping file: {}", error.message());
  load(mmapMem_.data(), mmapMem_.mapped_length(), check);
}

BinaryShortlistGenerator::BinaryShortlistGenerator(Ptr<Options> options,
                                                   Ptr<const Vocab> srcVocab,
                                                   Ptr<const Vocab> trgVocab,
                                                   size_t srcIdx /*= 0*/,
                                                   size_t /*trgIdx = 1*/,
                                                   bool shared /*= false*/)
    : options_(options),
      srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx),
      shared_(shared) {

  std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");
  ABORT_IF(vals.empty(), "No path to shortlist file given");
  std::string fname = vals[0];

  if(isBinaryShortlist(fname)){
    bool check = vals.size() > 1 ? std::stoi(vals[1]) : 1;
    LOG(info, "[data] Loading binary shortlist as {} {}", fname, check);
    load(fname, check);
  }
  else{
    firstNum_ = vals.size() > 1 ? std::stoi(vals[1]) : 100;
    bestNum_ = vals.size() > 2 ? std::stoi(vals[2]) : 100;
    float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;
    LOG(info, "[data] Importing text lexical shortlist as {} {} {} {}",
        fname, firstNum_, bestNum_, threshold);
    import(fname, threshold);
  }
}

BinaryShortlistGenerator::BinaryShortlistGenerator(const void *ptr_void,
                                                   const size_t blobSize,
                                                   Ptr<const Vocab> srcVocab,
                                                   Ptr<const Vocab> trgVocab,
                                                   size_t srcIdx /*= 0*/,
                                                   size_t /*trgIdx = 1*/,
                                                   bool shared /*= false*/,
                                                   bool check /*= true*/)
    : srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx),
      shared_(shared) {

  LOG(info, "[data] Loading binary shortlist from buffer with check={}", check);
  load(ptr_void, blobSize, check);
}

Ptr<Shortlist> BinaryShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  auto srcBatch = (*batch)[srcIdx_];
  size_t srcVocabSize = srcVocab_->size();
  size_t trgVocabSize = trgVocab_->size();

  // Since V=trgVocab_->size() is not large, anchor the time and space complexity to O(V).
  // Attempt to squeeze the truth tables into CPU cache
  std::vector<bool> srcTruthTable(srcVocabSize, 0);  // holds selected source words
  std::vector<bool> trgTruthTable(trgVocabSize, 0);  // holds selected target words

  // add firstNum most frequent words
  for(WordIndex i = 0; i < firstNum_ && i < trgVocabSize; ++i)
    trgTruthTable[i] = 1;

  // collect unique words from source
  // add aligned target words: mark trgTruthTable[word] to 1
  for(auto word : srcBatch->data()) {
    WordIndex srcIndex = word.toWordIndex();
    if(shared_)
      trgTruthTable[srcIndex] = 1;
    // If srcIndex has not been encountered, add the corresponding target words
    if (!srcTruthTable[srcIndex]) {
      for (uint64_t j = wordToOffset_[srcIndex]; j < wordToOffset_[srcIndex+1]; j++)
        trgTruthTable[shortLists_[j]] = 1;
      srcTruthTable[srcIndex] = 1;
    }
  }

  // Due to the 'multiple-of-eight' issue, the following O(N) patch is inserted
  size_t trgTruthTableOnes = 0;   // counter for no. of selected target words
  for (size_t i = 0; i < trgVocabSize; i++) {
    if(trgTruthTable[i])
      trgTruthTableOnes++;
  }

  // Ensure that the generated vocabulary items from a shortlist are a multiple-of-eight
  // This is necessary until intgemm supports non-multiple-of-eight matrices.
  for (size_t i = firstNum_; i < trgVocabSize && trgTruthTableOnes%8!=0; i++){
    if (!trgTruthTable[i]){
      trgTruthTable[i] = 1;
      trgTruthTableOnes++;
    }
  }

  // turn selected indices into vector and sort (Bucket sort: O(V))
  std::vector<WordIndex> indices;
  for (WordIndex i = 0; i < trgVocabSize; i++) {
    if(trgTruthTable[i])
      indices.push_back(i);
  }

  return New<Shortlist>(indices);
}

void BinaryShortlistGenerator::dump(const std::string& fileName) const {
  ABORT_IF(mmapMem_.is_open(),"No need to dump again");
  LOG(info, "[data] Saving binary shortlist dump to {}", fileName);
  saveBlobToFile(fileName);
}

void BinaryShortlistGenerator::import(const std::string& filename, double threshold) {
  io::InputFileStream in(filename);
  std::string src, trg;
  
  std::vector<std::unordered_map<WordIndex, float>> srcTgtProbTable(srcVocab_->size());
  float prob;

  // Read text file
  while(in >> trg >> src >> prob) {
    if(src == "NULL" || trg == "NULL")
      continue;

    auto sId = (*srcVocab_)[src].toWordIndex();
    auto tId = (*trgVocab_)[trg].toWordIndex();

    if(srcTgtProbTable[sId][tId] < prob)
      srcTgtProbTable[sId][tId] = prob;
  }

  // Create priority queue and count
  std::vector<std::priority_queue<std::pair<float, WordIndex>>> vpq;
  uint64_t shortListsSize = 0;

  vpq.resize(srcTgtProbTable.size());
  for(WordIndex sId = 0; sId < srcTgtProbTable.size(); sId++) {
    uint64_t shortListsSizeCurrent = 0;
    for(auto entry : srcTgtProbTable[sId]) {
      if (entry.first>=threshold) {
        vpq[sId].push(std::make_pair(entry.second, entry.first));
        if(shortListsSizeCurrent < bestNum_)
          shortListsSizeCurrent++;
      }
    }
    shortListsSize += shortListsSizeCurrent;
  }

  wordToOffsetSize_ = vpq.size() + 1;
  shortListsSize_ = shortListsSize;

  // Generate a binary blob
  blob_.resize(sizeof(Header) + wordToOffsetSize_ * sizeof(uint64_t) + shortListsSize_ * sizeof(WordIndex));
  struct Header* pHeader = (struct Header *)blob_.data();
  pHeader->magic = BINARY_SHORTLIST_MAGIC;
  pHeader->firstNum = firstNum_;
  pHeader->bestNum = bestNum_;
  pHeader->wordToOffsetSize = wordToOffsetSize_;
  pHeader->shortListsSize = shortListsSize_;
  uint64_t* wordToOffset = (uint64_t*)((char *)pHeader + sizeof(Header));
  WordIndex* shortLists = (WordIndex*)((char*)wordToOffset + wordToOffsetSize_*sizeof(uint64_t));

  uint64_t shortlistIdx = 0;
  for (size_t i = 0; i < wordToOffsetSize_ - 1; i++) {
    wordToOffset[i] = shortlistIdx;
    for(int popcnt = 0; popcnt < bestNum_ && !vpq[i].empty(); popcnt++) {
      shortLists[shortlistIdx] = vpq[i].top().second;
      shortlistIdx++;
      vpq[i].pop();
    }
  }
  wordToOffset[wordToOffsetSize_-1] = shortlistIdx;

  // Sort word indices for each shortlist
  for(int i = 1; i < wordToOffsetSize_; i++) {
    std::sort(&shortLists[wordToOffset[i-1]], &shortLists[wordToOffset[i]]);
  }
  pHeader->checksum = (uint64_t)util::hashMem<uint64_t>((uint64_t *)blob_.data()+2,
                                                        blob_.size()/sizeof(uint64_t)-2);

  wordToOffset_ = wordToOffset;
  shortLists_ = shortLists;
}

void BinaryShortlistGenerator::saveBlobToFile(const std::string& fileName) const {
  io::OutputFileStream outTop(fileName);
  outTop.write(blob_.data(), blob_.size());
}

}  // namespace data
}  // namespace marian
