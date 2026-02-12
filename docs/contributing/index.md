# How to Contribute

There are many ways you can contribute to add support for your language or improve translation quality!

Don't hesitate to ask questions in the Github issues or on Matrix [#firefoxtranslations:mozilla.org](https://matrix.to/#/#firefoxtranslations:mozilla.org).

## Native speakers

### Providing feedback

Feedback from native speakers of the languages we already support is very valuable to us. 

If you are a native speaker of a language we translate to and know the source language to some extent, you can help us understand the issues with the trained models.
Please test web-page translation functionality in Firefox Nightly on your favourite websites and report any issues you see by creating one Github issue per target language.
We will label it later as ["feedback"](https://github.com/mozilla/translations/issues?q=is%3Aissue+is%3Aopen+label%3Afeedback+) and discuss whether a fix can be implemented. Here is a [good example](https://github.com/mozilla/translations/issues/816) of a contributor reporting an issue.
Feel free to do it in a format convenient to you.

We are currently working on better feedback mechanisms, integrated in Firefox and as a standalone contribution platform.

### Finding new datasets

You can look at the [past training configs](https://github.com/mozilla/translations/tree/main/configs) and play with the [config-generator](https://github.com/mozilla/translations/blob/main/utils/config_generator.py) to see what open-source datasets are currently available for training.
If you know some datasets that are not in the configs, please create an issue or consider mentioning it in the existing issues labeled as ["data sources"](https://github.com/mozilla/translations/labels/data%20sources).

We need both parallel translation corpus and high-quality monolingual datasets, especially for lower resource languages.

See also [Data importers docs](../data-and-cleaning/datasets.md).

### Contributing datasets

We use [OPUS](https://opus.nlpl.eu/) as the main resource for parallel translation datasets. If you know of a dataset that is not on OPUS, please create an issue in 
[OPUS Ingest](https://github.com/Helsinki-NLP/OPUS-ingest/issues).

Another way is to submit your dataset to Hugging Face and let us know that we should use it. 

### Inspecting datasets

Another way to contribute is to look at specific parallel datasets for your language pair.

The main tool for this is [OpusCleaner](https://github.com/hplt-project/OpusCleaner).
If the dataset looks too noisy, create a pull request with exclusion rules in the [config-generator](https://github.com/mozilla/translations/blob/main/utils/config_generator.py).
One way to do this is by adding the dataset to the `skip_datasets` list, then it will be excluded for all languages.

You can also use OpusCleaner to design custom cleaning rules for a dataset.
See the examples of custom configs in the [/pipeline/clean/opuscleaner/configs](https://github.com/mozilla/translations/tree/main/pipeline/clean/opuscleaner/configs).

See also [documentation about OpusCleaner](https://mozilla.github.io/translations/docs/data-and-cleaning)

The trick is to not filter too much. Unfortunately, it's hard to say how the filters will affect the translation quality without training the model.

## Software developers

If you know Python, we're looking forward to your help with our backlog of Github issues related to model training. 
Feel free to start with the issues labelled as ["good first issue"](https://github.com/mozilla/translations/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

Issues labelled as ["help wanted"](https://github.com/mozilla/translations/labels/help%20wanted) typically require more expertise.

See [Development docs](development.md) to start with configuring local development environment.

Other ideas:

- Adding support for a new data importer
- Writing tests (we are far from the full code coverage)
- Contributing to the tools we use ([OpusCleaner](https://github.com/hplt-project/OpusCleaner), [OpusTrainer](https://github.com/hplt-project/OpusTrainer))
- Helping to figure out how to run the pipeline locally (Either with Taskcluster, see [this issue](https://github.com/mozilla/translations/issues/403) or with updating [Snakemake](../infrastructure/snakemake.md))

## ML engineers and researchers

### Giving feedback

If you have experience with Neural Machine Translation, NLP, Transformers, Knowledge Distillation, Quantization or other areas relevant to the training pipeline, 
you can look at how we train models and give feedback.

A good starting point would be the [model training guide](../training/README.md), the [/pipeline](https://github.com/mozilla/translations/tree/main/pipeline) directory where all the training code lives and the Github issues labelled
as ["quality"](https://github.com/mozilla/translations/labels/quality) and ["language coverage"](https://github.com/mozilla/translations/labels/language-coverage). 
You can leave feedback in those issues or create new ones if something is missing.

### Training models

Unfortunately, it's not possible at this time to use the Taskcluster training pipeline locally. It works only with the Mozilla infrastructure on GCP.

If you feel confident about giving it a try to train a new language pair or improve the quality of the existing ones,
reach out to us on Matrix, and we'll consider your request. 

The starting point is looking at the [model training guide](../training/README.md).
Then you can generate training configs locally with configs generator and look at the datasets (it's described in the "Inspecting datasets" section).
When the config is ready and you have a Taskcluster account, follow the [Taskcluster docs](../infrastructure/task-cluster.md) to run training.
You can monitor the training with the Tascluster UI and see ML charts on [Weights and Biases dashboards](https://wandb.ai/moz-translations/projects).
