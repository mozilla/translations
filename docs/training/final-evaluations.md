# Final Evaluation

After models are trained, final evaluations can be triggered.

Run an evaluation:

```sh
task eval -- --config taskcluster/configs/eval.yml
```

Make sure and update the `eval.yml` file to run it for specific metrics, translators etc. The evals will be logged to `trigger-eval.log` and uploaded to the bucket specified.

See the example config [taskcluster/configs/eval.yml](taskcluster/configs/eval.yml) for configuration details.

## Storage
The evaluation results are saved on GCS as JSON files with the following path templates:

```
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__<timestamp>__translations.json
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__<timestamp>__<metric>.metrics.json
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__<timestamp>__<metric>.scores.json
# + copied as "latest" for easy access
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__latest__translations.json
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__latest__<metric>.metrics.json
gs://<bucket>/final_evals/<src>-<trg>__<dataset>__<translator>__<model>__latest__<metric>.scores.json
```
Examples:
```
final_evals/en-ru__wmt24pp__google__v2__latest__translations.json
final_evals/en-ru__bouquet__opusmt__opus-mt-en-ru__latest__comet22.metrics.json
final_evals/en-ru__bouquet__bergamot__retrain_base-memory_KJ23-iDVTcymG1ZldWY17w__20251122T003231__bleu.metrics.json
```

When running on the production bucket, by default it will not overwrite previous results if "latest" file is present on GCS.
To rerun the specific evaluation specify `override: true` in the config. 
It will add evaluations with a new timestamp and replace "latest" files.

## Language pairs

Any language pair which has a two-letter ISO code can be used (some tools require code mapping, see [pipeline/eval/langs.py](pipeline/eval/langs.py)).

Non English-centric language pairs have limited support. 
We use two latest (or specified in the config) Bergamot models to run pivot translation through English.

Even if a Bergamot model for a language pair is absent in the storage, it is still possible to run evaluation for the other translators.

## Datasets

Only the latest high-quality datasets with good language coverage are used.

- Flores200-plus
- WMT24++
- Bouquet

## Translators
- Bergamot (Firefox models)
- Google Translate API
- Azure Translate API
- opusmt HF models
- NLLB 600M
- Argos Translate

Bergamot runs the final quantized models that we deploy in Firefox with bergamot-translator inference engine compiled in native mode. 
It is different from WASM mode used in Firefox.

### Models

Each translator can discover its available models. 
For Bergamot, it discovers all the models exported to the Bergamot format for the language pairs that are available on GCS. 

For example:
```
gs://moz-fx-translations-data--303e-prod-translations-data/models/en-ru/retrain_base-memory_KJ23-iDVTcymG1ZldWY17w/exported/lex.50.50.enru.s2t.bin.gz
gs://moz-fx-translations-data--303e-prod-translations-data/models/en-ru/retrain_base-memory_KJ23-iDVTcymG1ZldWY17w/exported/model.enru.intgemm.alphas.bin.gz
gs://moz-fx-translations-data--303e-prod-translations-data/models/en-ru/retrain_base-memory_KJ23-iDVTcymG1ZldWY17w/exported/vocab.enru.spm.gz
```

Other translators can also be extended to discover more than one model or API version.

The IDs of the models correspond to the model names in the file names. 

For example: 
```
retrain_base-memory_KJ23-iDVTcymG1ZldWY17w
v2
opus-mt-en-ru
```

To run evaluation only for the latest uploaded Bergamot models per language pair set `models: ["latest"]` in the config.

## Metrics

Supported metrics include:
- chrF
- chrF++
- BLEU
- COMET22
- MetricX-24 XL
- MetricX-24 XL QE (referenceless)
- LLM (reference-based)

### LLM Evaluation

An LLM can provide an evaluation using the OpenAI API. This will provide an analysis for an evaluation datasets of the following metrics, with a score of 1-5 and an explanation of the score:

 * adequacy
 * fluency
 * terminology
 * hallucination
 * punctuation

See [pipeline/eval/eval-batch-instructions.md](https://github.com/mozilla/translations/blob/main)(pipeline/eval/eval-batch-instructions.md) for the full prompt for this analysis.

This evaluation can be viewed using the [LLM Evals dashboard](https://mozilla.github.io/translations/llm-evals) by providing the root URL to where the JSON files are located.

## Running locally


Run under Docker with
```bash
task docker
```

Make sure `translator-cli` is compiled with 
```bash
task inference-build
```

Running some metrics, datasets and translators require setting environment variables with secrets:
```bash
# Hugging Face token to use restricted HF datasets ("bouquet", "flores200-plus")
export HF_TOKEN=...
# To use "llm-ref" OPEN AI API based metric
export OPENAI_API_KEY=...
# To use "microsoft" translator API
export AZURE_TRANSLATOR_KEY=...
# To use "google" translator API
export GOOGLE_APPLICATION_CREDENTIALS=<path>/creds.json
```
The output files are stored on disk in the `--artifacts` folder (by default `data/final_evals/`).

Run the evals script:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 
export PYTHONPATH=$(pwd) 
python pipeline/eval/final_eval.py \
  --config=taskcluster/configs/eval.yml \
  --artifacts=data/final_evals \
  --bergamot-cli=inference/build/src/app/translator-cli
```