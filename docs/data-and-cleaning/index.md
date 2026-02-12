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
`pipeline/clean/opuscleaner/configs/<src-lang-code>-<trg-lang-code>/` for language pair and dataset specific filters
(such filters will also apply to the opposite language pair)

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

### Language codes

OpusCleaner uses many external tools in its filters. It means support of language code schemes for specific tools can differ.
For some languages it's required to replace `<src>`, `<trg>` in the `filter.json` to the tools specific language codes.

For example, for Chinese Traditional we use:

- Pipeline code: `zh_hant`
- It maps to OpusCleaner code: `zh_Hant`
- It is changed in the [OpusCleaner config](https://github.com/mozilla/translations/blob/main/pipeline/clean/opuscleaner/configs/en-zh_hant/default.filters.json) for fasttext filter with openlid-v2 model to `cmn`:
```json
    {
      "filter": "fasttext_filter",
      "parameters": {
        "FASTTEXT_MODEL_TYPE": "openlid-v2",
        "LANG1": "eng",
        "LANG2": "cmn"
    }
```

See more details about the supported languages and language code mappings [here](../training/languages.md).

## Bicleaner

It is recommended to use Bicleaner ML models to filter noisy data.
See the [bicleaner documentation](bicleaner.md) for more details on how to configure it.


## Monolingual cleaning

Currently, it does not run OpusCleaner as monolingual filters are not fully supported. 
It runs legacy Bergamot cleaning scripts that include alphabet ratios and fast text. 
Also, it runs [Monocleaner](https://github.com/bitextor/monocleaner) to filter based on fluency.

Monocleaner thresholds can be adjusted in the training config:

```yaml
  # Monocleaner filters sentences in monolingual corpus based on language fluency
  # Use sanitized dataset names for compatibility with Taskcluster (replace ".", "/", ":", "[", "]" to "_")
  monocleaner:
    mono-src:
      # News-crawl is typically clean, enable on dataset by dataset basis
      default-threshold: 0.0
      dataset-thresholds:
        # We already filter it by document score, remove only the noisiest segments
        hplt_mono_v2_0: 0.5
        # Filter only garbage from NLLB
        opus_NLLB_v1: 0.5
    mono-trg:
      # News-crawl is typically clean, enable on dataset by dataset basis
      default-threshold: 0.0
      dataset-thresholds:
        # We already filter HPLT by document score, so it's relatively clean,
        # but let's still apply the default threshold for monocleaner to get more fluent target texts for back-translations
        hplt_mono_v2_0: 0.7
        # Sentences for back-translations should be in fluent language, apply even more aggressive threshold for NLLB
        opus_NLLB_v1: 0.8
```
