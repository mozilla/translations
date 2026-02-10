# Data Cleaning

Making datasets less noisy to improve quality of translation.

## OpusCleaner

We use an all-in-one cleaning tool [OpusCleaner](https://github.com/hplt-project/OpusCleaner) by HPLT project.

### Custom filter configs

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

Make sure to replace the language codes to the template values `<src>` and `<trg>` and remove absolutes paths from the `"files"` section. 
See examples in the directory.

### Default configs

Set `opuscleaner-mode: custom` (this is the default when generating a config) in the training config to use custom per-dataset and per-language pair configs.

If no custom config was specified for the dataset, 
the [default config template](https://github.com/mozilla/translations/tree/main/pipeline/clean/opuscleaner/configs/default.filters.json) will be used.

Modify if needed. Some rules require specifying source or target language. 
The `<src>` and `<trg>` in the template will be automatically replaced with the trained language pair.
The generated default config will be copied to the target dataset cleaning directory.

The config is chosen based on this search order:
1. Dataset and language specific: `configs/<language-pair>/<dataset>.filter.json`
2. Language specific: `configs/<language-pair>/default.filter.json`
3. Dataset specific: `configs/<dataset>.filter.json`
4. Default: `configs/default.filter.json`

The first found config will be applied.

If the desired behaviour is to apply only the default config template and skip all possible custom configs
for the current language pair and/or datasets, set `opuscleaner-mode: defaults`.

## Bicleaner

It is recommended to use Bicleaner ML models to filter noisy data.
See the [bicleaner documentation](bicleaner.md) for more details on how to configure it.
