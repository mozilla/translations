
Training pipelines and the inference engine for Firefox Translations machine translation models.

The trained models are hosted in [firefox-translations-models](https://github.com/mozilla/firefox-translations-models/) repository,
compatible with [bergamot-translator](https://github.com/mozilla/bergamot-translator) and
power the Firefox web page translation starting with version 118.

The pipeline was originally developed as a part of [Bergamot](https://browser.mt/) project  that focuses on improving client-side machine translation in a web browser.

[Documentation](https://mozilla.github.io/translations/)

## Pipeline

The pipeline is capable of training a translation model for a language pair end to end.
Translation quality depends on the chosen datasets, data cleaning procedures and hyperparameters.
Some settings, especially low resource languages might require extra tuning.

We use fast translation engine [Marian](https://marian-nmt.github.io).

You can find more details about the pipeline steps in the [documentation](docs/training/pipeline-steps.md).

## Orchestrators

An orchestrator is responsible for workflow management and parallelization.

- [Taskcluster](https://taskcluster.net/) - Mozilla task execution framework. It is also used for Firefox CI.
  It provides access to the hybrid cloud workers (GCP + on-prem) with increased scalability and observability.
  [Usage instructions](docs/training/task-cluster.md).
- [Snakemake](https://snakemake.github.io/) - a file based orchestrator that allows to run the pipeline locally or on a Slurm cluster.
  [Usage instructions](docs/training/snakemake.md). (The integration is not maintained since Mozilla has switched to Taskcluster. Contributions are welcome.)

## Experiment tracking

[Public training dashboard in Weights & Biases](https://wandb.ai/moz-translations/projects)

Marian training metrics are parsed from logs and published using a custom module within the `tracking` directory.
More information is available [here](docs/training/tracking.md).

## Contributing

Contributions are welcome! See the [documentation on Contributing](docs/contributing/index.md) for more details.

Feel free to ask questions in our Matrix channel [#firefoxtranslations:mozilla.org](https://matrix.to/#/#firefoxtranslations:mozilla.org).

## Useful Links

- [Reference papers](docs/README.md#references)
- [Model training guide](docs/training/README.md) - practical advice on how to use the pipeline
- [High level overview post on Mozilla Hacks](https://hacks.mozilla.org/2022/06/training-efficient-neural-network-models-for-firefox-translations/)
- [The Training Pipeline DAG](https://docs.google.com/presentation/d/1HkypImI_hbA3n1ljU57ZPAzW8PuQqdv2wrXqj688KtQ/edit?slide=id.g3421e8f521e_1_419#slide=id.g3421e8f521e_1_419)
- [Lightning Talk on the Training Pipeline Overview](https://www.youtube.com/watch?v=TfDEAYCeF6s)
- [Training and Experiment Dashboard](https://docs.google.com/spreadsheets/d/1Kiz9xUjo2jpeeVGtaL3jA_cLiCiiyz8GvIoQADMyYqo/edit?gid=0#gid=0)
- [moz-fx-translations-data--303e-prod-translations-data](https://console.cloud.google.com/storage/browser/moz-fx-translations-data--303e-prod-translations-data) - Uploaded models
- [Models in released to Firefox](https://mozilla.github.io/translations/firefox-models/)
- [Documentation of the Firefox integration](https://firefox-source-docs.mozilla.org/toolkit/components/translations/index.html)

## Acknowledgements

This project uses materials developed by:

- Bergamot project ([github](https://github.com/browsermt), [website](https://browser.mt/)) that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825303
- HPLT project ([github](https://github.com/hplt-project), [website](https://hplt-project.org/)) that has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546]
- OPUS-MT project ([github](https://github.com/Helsinki-NLP/Opus-MT), [website](https://opus.nlpl.eu/))
- Many other open source projects and research papers (see [References](docs/README.md#references))
