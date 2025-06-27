#include "marian.h"

#include "common/cli_helper.h"
#include "common/cli_wrapper.h"
#include "common/definitions.h"
#include "common/options.h"
#include "data/vocab.h"

#include <iostream>
#include <string>

using namespace marian;
using namespace marian::cli;

int main(int argc, char **argv) {
  auto options = New<Options>();

  YAML::Node config;
  auto w = New<CLIWrapper>(config);
  w->add<std::string>("-v,--vocab", "Path to vocab file");

  w->parse(argc, argv);
  options->merge(config);

  auto vocabPath = options->get<std::string>("vocab");
  size_t batchIndex = 0;
  auto vocab = New<Vocab>(options, batchIndex);
  vocab->load(vocabPath);

  auto printAnnotated
      = [](const std::string &reference, const std::vector<string_view> &byteRanges) {
          string_view reference_view(reference);
          const char *p = reference_view.data();
          for(auto &word : byteRanges) {
            // Print text before.
            string_view pre(p, word.data() - p);
            std::cout << pre;

            // Annotate and print token.
            std::cout << "[";
            std::cout << word;
            std::cout << "]";

            p = word.data() + word.size();
          }

          // Print tail text.
          const char *end = reference_view.data() + reference_view.size();
          string_view tail(p, end - p);
          std::cout << tail;
        };

  // Test encode with byteRanges
  std::string line, decoded;
  while(std::getline(std::cin, line)) {
    // If the ByteRanges coming out of SentencePiece are correct, expected
    // result is sourceToken that a word corresponding points to. The output
    // here is compared to an expected output containing unnormalized strings
    // and matches, test passes. Testing requires a SentencePiece model, and
    // hence designed as a test-app with tests in marian-regression-tests.

    std::vector<string_view> encodedByteRanges;
    auto words
        = vocab->encodeWithByteRanges(line, encodedByteRanges, /*addEOS=*/true, /*inference=*/true);
    std::cout << "Original: ";
    printAnnotated(line, encodedByteRanges);
    std::cout << std::endl;

    // Decode with ByteRanges on the same set of words should give a different
    // expected output onto an input string and string_views that are valid.

    std::vector<string_view> decodedByteRanges;
    vocab->decodeWithByteRanges(words, decoded, decodedByteRanges, /*ignoreEOS=*/true);
    std::cout << "Decoded: ";
    printAnnotated(decoded, decodedByteRanges);
    std::cout << std::endl;
  }

  return 0;
}
