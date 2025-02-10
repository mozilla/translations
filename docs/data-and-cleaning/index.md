# Data Cleaning

Making datasets less noisy to improve quality of translation.

## Regular pipeline

Config setting:
```
  use-opuscleaner: false
```

### Dataset fixing

Some datasets require fixes like detokenization.
Dataset and language specific fixes are implemented in [https://github.com/mozilla/translations/tree/main/pipeline/clean/fixes](https://github.com/mozilla/translations/tree/main/pipeline/clean/fixes).
Naming convention:
- `<dataset_name>.sh` for parallel dataset cleaning
- `<dataset_name>.<lang>.sh` for language specific cleaning of parallel or monolingual dataset
- `/` in dataset name should be replaced with `_`

### Cleaning scripts

Make sure the language is present in [clean_parallel](https://github.com/mozilla/translations/tree/main/pipeline/clean/tools/clean_parallel.py#L19) script.


### Bicleaner

It is recommended to use Bicleaner ML models to filter noisy data.
See the [bicleaner documentation](bicleaner.md) for more details on how to configure it.


## OpusCleaner

Another option is to use an all-in-one cleaning tool [OpusCleaner](https://github.com/hplt-project/OpusCleaner) by HPLT project.

Config setting:
```
  use-opuscleaner: "true"
```

To enable custom per-dataset filter configs add:
```
  opuscleaner-mode: "custom"
```


## Custom filter configs

The idea behind OpusCleaner is customizing filter rules for each language pair and dataset
to get a training corpus with less noise and train higher quality translation models.

Filtering rules can be tuned in an interactive UI.

### Installation

Install the OpusCleaner UI on a server. 
See the installation instructions in the [OpusCleaner readme](https://github.com/hplt-project/OpusCleaner).

For local usage: run from a poetry shell `task opuscleaner`.
Then go to `http://0.0.0.0:8000`.

### Making filters

Choose a language pair and download the required OPUS datasets. 
They will correspond to `opus_...` training datasets in the training pipeline config.

Configure cleaning rules for the datasets in the UI.

Copy JSON files for the produced filters `data/train-parts/*.filter.json` to 
`pipeline/clean/opuscleaner/configs/<src-lang-code>-<trg-lang-code>/` for langauge pair and dataset specific filters 
(such filters will also apply to the opposite langauge pair)

or to 

`pipeline/clean/opuscleaner/configs/` for dataset specific filters that will apply to all language pairs.

Make sure to replace the language codes to the template values `<src>` and `<trg>`. See examples in the directory.

### Default config

If no custom config was specified for the dataset, 
the [default config template](https://github.com/mozilla/translations/tree/main/pipeline/clean/opuscleaner/configs/default.filters.json) will be used.

Modify if needed. Some rules require specifying source or target language. 
The `<src>` and `<trg>` in the template will be automatically replaced with the trained language pair.
The generated default config will be copied to the target dataset cleaning directory.

### Running 

Enable OpusCleaner in the training pipeline config and run the pipeline as usual. 
OpusCleaner will replace the default [clean-corpus](https://github.com/mozilla/translations/tree/main/pipeline/clean/clean-corpus.sh) script.
