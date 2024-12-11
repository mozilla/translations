#include "annotation.h"

#include <cassert>

namespace marian {
namespace bergamot {

AnnotatedText::AnnotatedText(std::string &&t) : text(std::move(t)) {
  // Treat the entire text as a gap that recordExistingSentence will break.
  annotation.token_begin_.back() = text.size();
}

void AnnotatedText::appendSentence(string_view prefix, std::vector<string_view>::iterator begin,
                                   std::vector<string_view>::iterator end) {
  assert(annotation.token_begin_.back() == text.size());

  // prefix is just end of the previous one.
  handleEndingWhitespace(prefix, /* isBetweenSentences */ true);

  // Appending sentence text.
  std::size_t offset = text.size();
  for (std::vector<string_view>::iterator token = begin; token != end; ++token) {
    offset += token->size();
    annotation.token_begin_.push_back(offset);
  }
  if (begin != end) {
    text.append(begin->data(), (end - 1)->data() + (end - 1)->size());
    assert(offset == text.size());  // Tokens should be contiguous.
  }

  // Add the gap after the sentence.  This is empty for now, but will be
  // extended with handleEndingWhitespace or another appendSentence.
  annotation.gap_.push_back(annotation.token_begin_.size() - 1);
  annotation.token_begin_.push_back(offset);
}

/// A simple helper function to check if a string starts with a prefix.
/// The std::string object only has a starts_with() method in C++20, which
/// is not what we are currently compiling with.
bool startsWith(string_view prefix, string_view str) {
  return str.size() >= prefix.size() && prefix == str.substr(0, prefix.size());
}

bool AnnotatedText::shouldOmitSpaceBetweenSentences() const {
  if (targetLanguage_.empty()) {
    // The target language is not specified, so we should not make assumptions about
    // whether or not the language's script should omit whitespace.
    return false;
  }

  // TODO(https://github.com/mozilla/translations/issues/950)
  // More robustly handle which language tags should omit whitespace between sentences.
  return (
    // Japanese does not use space between sentences.
    startsWith("ja", targetLanguage_) ||
    // Korean does not use space between sentences.
    startsWith("ko", targetLanguage_) ||
    // Chinese does not use space between sentences.
    startsWith("zh", targetLanguage_)
  );
}

bool AnnotatedText::shouldEnsureSpaceBetweenSentences() const {
  if (targetLanguage_.empty()) {
    // The target language is not specified, so we should not make assumptions about
    // whether or not the language's script should omit whitespace.
    return false;
  }

  return !shouldOmitSpaceBetweenSentences();
}

void AnnotatedText::maybeAppendHTMLTagsFromGap(string_view gap) {
  // We can be sure that the gap between sentences is one of the following:
  //  - Empty
  //  - Whitespace
  //  - One or more well-formed HTML tags, e.g. "</b></em>".
  //  - A mixture of whitespace and HTML tags, e.g. "</b></em>  ".
  size_t currentIndex = 0;
  while (currentIndex < gap.size()) {
    // Find the next open bracket '<' for an HTML tag.
    size_t tagStart = gap.find('<', currentIndex);
    if (tagStart == string_view::npos) {
      // No more HTML tags were found.
      return;
    }

    // Find the matching closing bracket '>' for this HTML tag.
    size_t tagEnd = gap.find('>', tagStart + 1);
    if (tagEnd == string_view::npos) {
      // The tag is missing its closing angle bracket.
      // This should never happen, since the DOM parser should ensure the tags are well formed.
      // But if we do encounter this case, the best thing we can do at this point ignore the tag.
      return;
    }

    size_t tagLength = 1 + tagEnd - tagStart;
    string_view tag = gap.substr(tagStart, tagLength);
    text.append(tag.data(), tag.size());

    currentIndex = tagEnd + 1;
  }
}

void AnnotatedText::handleEndingWhitespace(string_view gap, bool isBetweenSentences) {
  if (gap.find("\n") != string_view::npos) {
    // The gap contains a line break, so we should preserve it regardless.
    text.append(gap.data(), gap.size());
  } else if (shouldOmitSpaceBetweenSentences()) {
    // Even if we are supposed to omit gap between sentences, the gap between
    // the sentences may contain HTML tags that we still need to preserve.
    maybeAppendHTMLTagsFromGap(gap);
  } else if (!gap.empty()) {
    // We are not explicitly omitting gap and there is gap to preserve.
    text.append(gap.data(), gap.size());
  } else if (
    // This is a gap between sentences (i.e. not at the end of the whole text).
    isBetweenSentences &&
    // This the current language/script should have space between sentences.
    shouldEnsureSpaceBetweenSentences() &&
    // The previous sentence is not empty.
    !text.empty()
  ) {
    // The given gap was empty, but but target language requires a space between sentences.
    text += ' ';
  }

  annotation.token_begin_.back() = text.size();
}

void AnnotatedText::recordExistingSentence(std::vector<string_view>::iterator begin,
                                           std::vector<string_view>::iterator end, const char *sentence_begin) {
  assert(sentence_begin >= text.data());
  assert(sentence_begin <= text.data() + text.size());
  assert(begin == end || sentence_begin == begin->data());
  assert(!annotation.token_begin_.empty());
  assert(annotation.token_begin_.back() == text.size());
  // Clip off size token ending.
  annotation.token_begin_.pop_back();
  for (std::vector<string_view>::iterator i = begin; i != end; ++i) {
    assert(i->data() >= text.data());                                  // In range.
    assert(i->data() + i->size() <= text.data() + text.size());        // In range
    assert(i + 1 == end || i->data() + i->size() == (i + 1)->data());  // Contiguous
    annotation.token_begin_.push_back(i->data() - text.data());
  }
  // Gap token after sentence.
  annotation.gap_.push_back(annotation.token_begin_.size());
  if (begin != end) {
    annotation.token_begin_.push_back((end - 1)->data() + (end - 1)->size() - text.data());
  } else {
    // empty sentence.
    annotation.token_begin_.push_back(sentence_begin - text.data());
  }
  // Add back size token ending.
  annotation.token_begin_.push_back(text.size());
}

}  // namespace bergamot
}  // namespace marian
