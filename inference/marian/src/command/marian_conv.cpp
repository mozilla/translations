#include "marian.h"

#include "common/cli_wrapper.h"

#include <sstream>

#include "data/shortlist.h"
#include "tensors/cpu/expression_graph_packable.h"
#include "onnx/expression_graph_onnx_exporter.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  auto options = New<Options>();
  {
    YAML::Node config; // @TODO: get rid of YAML::Node here entirely to avoid the pattern. Currently not fixing as it requires more changes to the Options object.
    auto cli = New<cli::CLIWrapper>(
        config,
        "Convert a model in the .npz format and normal memory layout to a mmap-able binary model which could be in normal memory layout or packed memory layout\n"
        "or convert a text lexical shortlist to a binary shortlist with {--shortlist,-s} option",
        "Allowed options",
        "Examples:\n"
        "  ./marian-conv -f model.npz -t model.bin --gemm-type packed16");
    cli->add<std::string>("--from,-f", "Input model", "model.npz");
    cli->add<std::string>("--to,-t", "Output model", "model.bin");
    cli->add<std::string>("--export-as", "Kind of conversion: marian-bin or onnx-{encode,decoder-step,decoder-init,decoder-stop}", "marian-bin");
    cli->add<std::string>("--gemm-type,-g", "GEMM Type to be used: float32, packed16, packed8avx2, packed8avx512, intgemm8, intgemm16", "float32");
    cli->add<std::vector<std::string>>("--vocabs,-V", "Vocabulary file, required for ONNX export and shortlist conversion");
    cli->add<std::vector<std::string>>("--shortlist,-s", "Shortlist conversion: filePath firstNum bestNum threshold");
    cli->add<std::string>("--dump,-d", "Binary shortlist dump path","lex.bin");
    cli->parse(argc, argv);
    options->merge(config);
  }

  // shortlist conversion example:
  // ./marian-conv --shortlist lex.esen.s2t 100 100 0 --dump lex.esen.bin --vocabs vocab.esen.spm vocab.esen.spm
  if(options->hasAndNotEmpty("shortlist")){
    auto vocabPaths = options->get<std::vector<std::string>>("vocabs");
    auto dumpPath = options->get<std::string>("dump");

    Ptr<Vocab> srcVocab = New<Vocab>(options, 0);
    srcVocab->load(vocabPaths[0]);
    Ptr<Vocab> trgVocab = New<Vocab>(options, 1);
    trgVocab->load(vocabPaths[1]);

    Ptr<const data::ShortlistGenerator> binaryShortlistGenerator
        = New<data::BinaryShortlistGenerator>(options, srcVocab, trgVocab, 0, 1, vocabPaths[0] == vocabPaths[1]);
    binaryShortlistGenerator->dump(dumpPath);
    LOG(info, "Finished");
    return 0;
  }

  auto modelFrom = options->get<std::string>("from");
  auto modelTo = options->get<std::string>("to");

  auto exportAs = options->get<std::string>("export-as");
  auto vocabPaths = options->get<std::vector<std::string>>("vocabs");// , std::vector<std::string>());
  
  auto saveGemmTypeStr = options->get<std::string>("gemm-type", "float32");
  Type saveGemmType;
  if(saveGemmTypeStr == "float32") {
    saveGemmType = Type::float32;
  } else if(saveGemmTypeStr == "packed16") {  // packed16 (fbgemm) only supports AVX2. AVX512 might be added later
    saveGemmType = Type::packed16;
  } else if(saveGemmTypeStr == "packed8avx2") { // packed8 for AVX2 (fbgemm)
    saveGemmType = Type::packed8avx2;
  } else if(saveGemmTypeStr == "packed8avx512") { // packed8 for AVX512 (fbgemm)
    saveGemmType = Type::packed8avx512;
  } else if(saveGemmTypeStr == "intgemm8") { // intgemm 8 bit format
    saveGemmType = Type::intgemm8;
  } else if(saveGemmTypeStr == "intgemm16") { // intgemm 16 bit format
    saveGemmType = Type::intgemm16;
  } else {
    ABORT("Unknown gemm-type: {}", saveGemmTypeStr);
  }

  LOG(info, "Outputting {}, precision: {}", modelTo, saveGemmType);

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", modelFrom);
  configStr << config;

  auto load = [&](Ptr<ExpressionGraph> graph) {
    graph->setDevice(CPU0);
    graph->getBackend()->setInt8(false);  // Since win run graph->forward() we need to make sure it does not get converted to an intgemm format during it.
    graph->getBackend()->setInt16(false); // We manually do the compression later.

    graph->load(modelFrom);
    graph->forward();  // run the initializers
  };


  if (exportAs == "marian-bin") {
    auto graph = New<ExpressionGraphPackable>();
    load(graph);
    // added a flag if the weights needs to be packed or not
    graph->packAndSave(modelTo, configStr.str(), /* --gemm-type */ saveGemmType, Type::float32);
  }
  else if (exportAs == "onnx-encode") {
#ifdef USE_ONNX
    auto graph = New<ExpressionGraphONNXExporter>();
    load(graph);
    auto modelOptions = New<Options>(config)->with("vocabs", vocabPaths, "inference", true);

    graph->exportToONNX(modelTo, modelOptions, vocabPaths);
#else
    ABORT("--export-as onnx-encode requires Marian to be built with USE_ONNX=ON");
#endif // USE_ONNX
  }
  else
    ABORT("Unknown --export-as value: {}", exportAs);

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
