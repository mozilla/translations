#include "text_processor.h"

#include <vector>

#include "annotation.h"
#include "common/cli_helper.h"
#include "common/options.h"
#include "data/types.h"
#include "definitions.h"

#if defined(WASM)
#include <emscripten.h>
#endif // defined(WASM)

namespace marian {
namespace bergamot {

#if !defined(WASM)
namespace {
ug::ssplit::SentenceStream::splitmode string2splitmode(const std::string &m) {
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
  if (m == "sentence") {
    return splitmode::one_sentence_per_line;
  } else if (m == "paragraph") {
    return splitmode::one_paragraph_per_line;
  } else if (m == "wrapped_text") {
    return splitmode::wrapped_text;
  } else {
    ABORT("Unknown ssplitmode {}, Please choose one of {sentence,paragraph,wrapped_text}");
  }
}

ug::ssplit::SentenceSplitter loadSplitter(const std::string &ssplitPrefixFile) {
  // Temporarily supports empty, will be removed when mozilla passes ssplitPrefixFile
  ug::ssplit::SentenceSplitter splitter;
  if (ssplitPrefixFile.size()) {
    std::string interpSsplitPrefixFile = marian::cli::interpolateEnvVars(ssplitPrefixFile);
    LOG(info, "Loading protected prefixes for sentence splitting from {}", interpSsplitPrefixFile);
    splitter.load(interpSsplitPrefixFile);
  } else {
    LOG(warn,
        "Missing list of protected prefixes for sentence splitting. "
        "Set with --ssplit-prefix-file.");
  }
  return splitter;
}

ug::ssplit::SentenceSplitter loadSplitter(const AlignedMemory &memory) {
  // Temporarily supports empty, will be removed when mozilla passes memory
  ug::ssplit::SentenceSplitter splitter;
  if (memory.size()) {
    std::string_view serialized(memory.begin(), memory.size());
    splitter.loadFromSerialized(serialized);
  }
  return splitter;
}

}  // namespace

TextProcessor::TextProcessor(Ptr<Options> options, const Vocabs &vocabs, const std::string &ssplit_prefix_file)
    : vocabs_(vocabs), ssplit_(loadSplitter(ssplit_prefix_file)) {
  parseCommonOptions(options);
}

TextProcessor::TextProcessor(Ptr<Options> options, const Vocabs &vocabs, const AlignedMemory &memory)
    : vocabs_(vocabs) {
  // This is not the best of the solutions at the moment, but is consistent with what happens among other structures
  // like model, vocabulary or shortlist. First, we check if the bytearray is empty. If not, we load from ByteArray. In
  // case empty, the string based loader which reads from file is called. However, ssplit allows for not supplying
  // ssplit-prefix-file where-in the purely regular expression based splitter is activated.
  //
  // For now, we allow not supplying an ssplit-prefix-file.

  if (memory.begin() == nullptr && memory.size()) {
    ssplit_ = loadSplitter(memory);
  } else {
    ssplit_ = loadSplitter(options->get<std::string>("ssplit-prefix-file", ""));
  }
  parseCommonOptions(options);
}

void TextProcessor::parseCommonOptions(Ptr<Options> options) {
  maxLengthBreak_ = options->get<size_t>("max-length-break");
  ssplitMode_ = string2splitmode(options->get<std::string>("ssplit-mode"));
}

void TextProcessor::process(std::string &&input, AnnotatedText &source, Segments &segments) const {
  source = std::move(AnnotatedText(std::move(input)));
  std::string_view input_converted(source.text.data(), source.text.size());
  auto sentenceStream = ug::ssplit::SentenceStream(input_converted, ssplit_, ssplitMode_);

  std::string_view sentenceStringPiece;

  while (sentenceStream >> sentenceStringPiece) {
    marian::string_view sentence(sentenceStringPiece.data(), sentenceStringPiece.size());

    std::vector<string_view> wordRanges;
    Segment segment = tokenize(sentence, wordRanges);

    // There are some cases where SentencePiece or vocab returns no words
    // after normalization. 0 prevents any empty entries from being added.
    if (segment.size() > 0) {
      // Wrap segment into sentences of at most maxLengthBreak_ tokens and
      // tell source about them.
      wrap(segment, wordRanges, segments, source);
    }
  }
}
#elif defined(WASM)
void TextProcessor::process(std::string&& input, AnnotatedText& source, Segments& segments) const {
  /**
   * This is an RAII guard for auto-freeing a malloc-allocated pointer when it leaves scope.
   *
   * The address of the internal pointer is passed to JavaScript where it is allocated
   * via `malloc`, however we are responsible for deallocating the memory via `free` on
   * the C++ side.
   *
   * Wrapping the deallocation in the checked destructor ensures that once the memory has been 
   * allocated on the JS side, it is guaranteed to be deallocated when this stack frame is popped,
   * whether from returning or from an exception being thrown.
   */
  struct ScopedPtr {
    int32_t* ptr = nullptr;

    const int32_t operator[](const size_t index) const {
      return ptr[index];
    }

    ~ScopedPtr() {
      if (ptr) {
        free(ptr);
        ptr = nullptr;
      }
    }
  };

  ScopedPtr starts;
  ScopedPtr ends;
  int32_t sentenceCount = 0;

  source = std::move(AnnotatedText(std::move(input)));
  std::string_view input_converted(source.text.data(), source.text.size());

  EM_ASM({
    // Attach a lazily initialized cache of the Intl.Segmenter to the WASM Module.
    // This is critical to prevent creating one instance of Intl.Segmenter per
    // translation request.
    if (!Module.getOrCreateSentenceSegmenter) {
      Module.getOrCreateSentenceSegmenter = (function() {
        let segmenters = new Map();

        return function(lang) {
          let segmenter = segmenters.get(lang);

          if (!segmenter) {
            segmenter = new Intl.Segmenter(lang, { granularity: "sentence" });
            segmenters.set(lang, segmenter);
          }

          return segmenter;
        };
      })();
    }

    // Convert the UTF-8 C++ strings into UTF-16 for JavaScript.
    const inputUTF16 = UTF8ToString($0);
    const lang = UTF8ToString($1);

    // Segment the UTF-16 input with the Intl.Segmenter.
    const segmenter = Module.getOrCreateSentenceSegmenter(lang);
    const sentencesUTF16 = Array.from(segmenter.segment(inputUTF16));
    const sentenceCount = sentencesUTF16.length;

    // Allocate enough space to mark the start and end UTF-8 byte indices for each sentence.
    const bytesPerInt = 4;
    const startsPtr = _malloc(sentenceCount * bytesPerInt);
    const endsPtr = _malloc(sentenceCount * bytesPerInt);

    if (!startsPtr || !endsPtr) {
      throw new Error("Failed to allocate WASM memory for segmentation.");
    }

    // Iterate through all of the segments and map the start and end of each
    // sentence as UTF-8 byte ranges so that the C++ code can operate on them.
    let sentenceEndUTF8 = 0;
    sentencesUTF16.forEach(({ segment: sentenceUTF16 }, index) => {
      const sentenceStartUTF8 = sentenceEndUTF8;
      sentenceEndUTF8 += lengthBytesUTF8(sentenceUTF16);

      setValue(startsPtr + index * bytesPerInt, sentenceStartUTF8, 'i32');
      setValue(endsPtr + index * bytesPerInt, sentenceEndUTF8, 'i32');
    });

    setValue($2, sentenceCount, 'i32');
    setValue($3, startsPtr, 'i32');
    setValue($4, endsPtr, 'i32');
  }, input_converted.data(), sourceLanguage_.c_str(), &sentenceCount, &starts.ptr, &ends.ptr);

  if (sentenceCount <= 0) {
    return;
  }

  int32_t paragraphStart = starts[0];
  int32_t paragraphEnd = ends[sentenceCount - 1];
  int32_t length = paragraphEnd - paragraphStart;

  marian::string_view paragraph(source.text.data() + paragraphStart, length);
  std::vector<string_view> wordRanges;
  Segment segment = tokenize(paragraph, wordRanges);

  // There are some cases where SentencePiece or vocab returns no words
  // after normalization. 0 prevents any empty entries from being added.
  if (segment.size() > 0) {
    // Wrap segment into sentences of at most maxLengthBreak_ tokens and
    // tell source about them.
    wrap(segment, wordRanges, segments, source);
   }
}
#endif // defined(WASM)

Segment TextProcessor::tokenize(const string_view &segment, std::vector<string_view> &wordRanges) const {
  // vocabs_->sources().front() is invoked as we currently only support one source vocab
  return vocabs_.sources().front()->encodeWithByteRanges(segment, wordRanges, /*addEOS=*/false, /*inference=*/true);
}

void TextProcessor::wrap(Segment &segment, std::vector<string_view> &wordRanges, Segments &segments,
                         AnnotatedText &source) const {
  // There's an EOS token added to the words, manually. SentencePiece/marian-vocab is set to not append EOS. Marian
  // requires EOS to be at the end as a marker to start translating. So while we're supplied maxLengthBreak_ from
  // outside, we need to ensure there's space for EOS in each wrapped segment.
  Word sourceEosId = vocabs_.sources().front()->getEosId();
  size_t wrapStep = maxLengthBreak_ - 1;

  for (size_t offset = 0; offset < segment.size(); offset += wrapStep) {
    auto start = segment.begin() + offset;

    // Restrict the range within bounds.
    size_t left = segment.size() - offset;
    size_t diff = std::min(wrapStep, left);

    segments.emplace_back(start, start + diff);
    segments.back().push_back(sourceEosId);

    auto astart = wordRanges.begin() + offset;

    // Construct a part vector of string_view representing wrapped segment, use the last string_view to create an EOS
    // string_view manually.
    std::vector<string_view> partWordRanges(astart, astart + diff);
    string_view &last = partWordRanges.back();
    const char *end = last.data() + last.size();
    partWordRanges.emplace_back(end, 0);
    // diff > 0
    source.recordExistingSentence(partWordRanges.begin(), partWordRanges.end(), astart->data());
  }
}

void TextProcessor::processFromAnnotation(AnnotatedText &source, Segments &segments) const {
  std::string copySource = source.text;
  AnnotatedText replacement(std::move(copySource));

  for (size_t s = 0; s < source.numSentences(); s++) {
    // This is our sentenceStream
    ByteRange sentenceByteRange = source.sentenceAsByteRange(s);

    // Fool tokenization using ByteRanges into looking at replacement. They're same, so okay.
    marian::string_view sentence{&replacement.text[sentenceByteRange.begin], sentenceByteRange.size()};

    std::vector<string_view> wordRanges;
    Segment segment = tokenize(sentence, wordRanges);

    // Manually add EoS
    Word sourceEosId = vocabs_.sources().front()->getEosId();
    segment.push_back(sourceEosId);

    if (!wordRanges.empty()) {
      string_view &last = wordRanges.back();  // this is a possible segfault if wordRanges is empty. So guard.
      const char *end = last.data() + last.size();
      wordRanges.emplace_back(end, 0);
    } else {
      const char *end = sentence.data() + sentence.size();
      wordRanges.emplace_back(end, 0);
    }

    segments.push_back(std::move(segment));
    replacement.recordExistingSentence(wordRanges.begin(), wordRanges.end(), wordRanges.begin()->data());
  }

  source = replacement;
}

}  // namespace bergamot
}  // namespace marian
