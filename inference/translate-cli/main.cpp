#include "translator/byte_array_util.h"
#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"
#include "translator/utils.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  using namespace marian::bergamot;

  ConfigParser<BlockingService> configParser("Translate CLI", /*multiOpMode=*/false);
  configParser.parseArgs(argc, argv);
  auto &config = configParser.getConfig();
  LOG(info, "Config received.\n");

  BlockingService service(config.serviceConfig);
  LOG(info, "Single threaded CPU service created");

  // Construct a model.
  auto options = parseOptionsFromFilePath(
    config.modelConfigPaths.front(), true /* validate */);

  LOG(info, "The model options are built.");

  marian::Ptr<TranslationModel> model = marian::New<TranslationModel>(options);

  LOG(info, "TranslationModel created.");

  std::vector<std::string> inputs {};
  std::vector<ResponseOptions> responseOptions {};

  inputs.push_back(readFromStdin());
  responseOptions.push_back(ResponseOptions {});

  LOG(info, "Preparing to translate.");
  std::vector<Response> responses = service.translateMultiple(model, std::move(inputs), responseOptions);

  LOG(info, "Translation complete");
  for (const auto& response : responses) {
    std::cout << response.target.text;
  }

  return 0;
}
