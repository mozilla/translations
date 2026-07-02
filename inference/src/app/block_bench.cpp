// Block benchmark harness for the marian-fork reference.
//
// Loads the model once, then translates a stream of *blocks* — one batched call
// per block — timing each. This matches the production/Wasm path: one document
// per call on a loaded model, whose sentences bergamot batches together.
//
// Input (stdin): blank-line-delimited blocks, one sentence per line, an empty
// line between blocks. We parse that structure here and hand each block to
// bergamot as a single source string (sentences newline-joined). With
// `ssplit-mode: sentence` bergamot splits it back into exactly these sentences —
// no re-splitting — and batches them. Block boundaries are resolved before
// bergamot sees anything, so there is no newline ambiguity between the
// per-sentence and per-block levels.
//
// Output: each block's translation on stdout (blank-line separated); one JSON
// timing span per block on stderr, e.g. `[block] {"block":0,"sentences":3,"wall_ms":12.3}`.

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

using namespace marian::bergamot;

int main(int argc, char *argv[]) {
  ConfigParser<BlockingService> configParser("Block benchmark (one batched translate per block)",
                                             /*multiOpMode=*/false);
  configParser.parseArgs(argc, argv);
  auto &config = configParser.getConfig();

  BlockingService service(config.serviceConfig);
  auto modelConfig = parseOptionsFromFilePath(config.modelConfigPaths.front());
  std::shared_ptr<TranslationModel> model = std::make_shared<TranslationModel>(modelConfig);

  // Parse blank-line-delimited blocks from stdin. `blocks[i]` is one block's
  // sentences newline-joined; `counts[i]` is its sentence count.
  std::vector<std::string> blocks;
  std::vector<int> counts;
  std::string cur, line;
  int nSent = 0;
  auto flush = [&]() {
    if (!cur.empty()) {
      blocks.push_back(cur);
      counts.push_back(nSent);
      cur.clear();
      nSent = 0;
    }
  };
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) {
      flush();
    } else {
      if (!cur.empty()) cur += '\n';
      cur += line;
      ++nSent;
    }
  }
  flush();

  ResponseOptions responseOptions;
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    std::vector<std::string> sources{blocks[bi]};
    std::vector<ResponseOptions> opts{responseOptions};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Response> responses = service.translateMultiple(model, std::move(sources), opts);
    auto t1 = std::chrono::steady_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << responses[0].target.text << "\n\n";
    std::cerr << "[block] {\"block\":" << bi << ",\"sentences\":" << counts[bi]
              << ",\"wall_ms\":" << wallMs << "}\n";
  }
  return 0;
}
