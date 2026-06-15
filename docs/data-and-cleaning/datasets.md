# Dataset importers

Dataset importers can be used in `datasets` sections of the [training config](https://github.com/mozilla/translations/blob/main/taskcluster/configs/config.prod.yml).

Example:
```
  train:
    - opus_ada83/v1
    - mtdata_newstest2014_ruen
```

Data source | Prefix     | Name examples                                                                                 | Type     | Comments
--- |------------|-----------------------------------------------------------------------------------------------|----------| ---
[MTData](https://github.com/thammegowda/mtdata) | mtdata     | newstest2017_ruen                                                                             | parallel | Supports many datasets. Run `mtdata list -l ru-en` to see datasets for a specific language pair.
[OPUS](https://opus.nlpl.eu/) | opus       | ParaCrawl/v7.1                                                                                | parallel   | Many open source datasets. Go to the website, choose a language pair, check links under Moses column to see what names and version is used in a link.
[SacreBLEU](https://github.com/mjpost/sacrebleu) | sacrebleu  | wmt20                                                                                         | parallel   | Official evaluation datasets available in SacreBLEU tool. Recommended to use in `datasets:test` config section. Look up supported datasets and language pairs in `sacrebleu.dataset` python module.
[Flores](https://github.com/facebookresearch/flores) | flores     | dev, devtest                                                                                  | parallel   | Evaluation dataset from Facebook that supports 100 languages.
[NTREX-128](https://github.com/MicrosoftTranslator/NTREX) | ntrex     | devtest                                                                                  | parallel   | Evaluation dataset from Microsoft that supports 128 languages.
Custom parallel | url        | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.[LANG].zst` | parallel   | A custom zst compressed parallel dataset, for instance uploaded to GCS. The language pairs should be split into two files. the `[LANG]` will be replaced with the `to` and `from` language codes.
[News crawl](http://data.statmt.org/news-crawl) | news-crawl | news.2019                                                                                     | mono     | Monolingual news datasets from [WMT](https://www.statmt.org/wmt21/translation-task.html)
[OPUS](https://opus.nlpl.eu/) | opus       | tldr-pages/v2023-08-29                                                                        | mono     | Monolingual dataset from OPUS.
[HPLT](https://hplt-project.org/datasets/v2.0) | hplt       | mono/v3.0                                                                                     | mono     | HPLT monolingual corpus (mostly from Internet Archive, but also from Common Crawl).
[Hugging Face](https://huggingface.co/datasets) | hf | orai-nlp/ZelaiHandi:train:text@fb8101a | mono | Hugging Face monolingual dataset. Dataset names have to include Hugging Face repository identifier, git commit revision (optional) dataset split and dataset field where text is located. The format is `{repo_name}:{subset}:{split}:{field}@{revision}`. Note that this importer is only for **manually** included datasets and `find-corpus` does not include Hugging Face search.
[Hugging Face](https://huggingface.co/datasets) parallel data | hfp | ayymen/Weblate-Translations:default:train:source\_string:target\_string@fb8101a | parallel | Hugging Face parallel dataset. Dataset names have to include Hugging Face repository identifier, subset, split, field as srouce, field as target and git commit revision (optional). For further details see [here](#huggingface-datasets). Note that this importer is only for **manually** included datasets and `find-corpus` does not include Hugging Face search.
Custom mono | url        | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.ru.zst`     | mono     | A custom zst compressed monolingual dataset, for instance uploaded to GCS.

You can also use [find-corpus](https://github.com/mozilla/translations/blob/main/utils/find_corpus.py) tool to find all datasets for an importer and get them formatted to use in config.

Set up a local [poetry](https://python-poetry.org/) environment.
```bash
task find-corpus -- en ru
```

The config generator uses `find-corpus` to generate a training config automatically and include all the available datasets:
```bash
task config-generator -- ru en --name test
```

Make sure to check licenses of the datasets before using them.

## HuggingFace datasets
The HF dataset importers use the following parameters (formatted as `{repo_name}:{subset}:{split}:{field}@{revision}` or `{repo_name}:{subset}:{split}:{src_field}:{trg_field}@{revision}`):
 - `repo_name`: the Hugging Face dataset repository identifier (includes organization and name).
 - `subset`: the dataset subset. It is common that this is not shown in the dataset viewer at the HF page, in this case subset will probably be `default`. For parallel data it is also common to specify the language pair here.
 - `split`: the dataset split. Typically `train`.
 - `field` or (`src_field` and `trg_field` for parallel): the row field (column) where the text is located.

### Monlingual
Monolingual HF dataset importer assumes each row is a document and will convert to a plain text without document boundaries and preserving endlines (sentences or paragraphs). No additial processing is performed.

### Parallel
Parallel data HF importer supports two types of parallel datasets. Both can be used with the same pipeline config format (`hfp_{repo_name}:{subset}:{split}:{src_field}:{trg_field}@{revision}`) and the importer will automatically detect the type.
However there are other types, described below, that are not supported.

The first type is the Translation HF format, which only has one feature (or column) called 'translation' and each row is a JSON containing language codes as keys and text for each translation side as values.
Examples of datasets using this format are the Helsinki-NLP or WMT organizations datasets.
To use a dataset using this format, like [Helsinki-NLP/euconst](https://huggingface.co/datasets/Helsinki-NLP/euconst), you should format the config entry like this: `hfp_Helsinki-NLP/euconst:en-fr:train:fr:en`, where `fr` and `en` are the JSON keys that will be used as source text and target text respectively.

The second type is for datasets that do not use the Translation HF format but instead a normal text dataset format. In this case the HF dataset viewer will show several columns for each row instead of a JSON.
For this type specify in `src_field` and `trg_field` the dataset columns that have to be used as source and target sentence.
For example to use this dataset: `hfp_ayymen/Weblate-Translations:en-fr:train:source_string:target_string`.

Other types of dataset for translation datasets that are currently not supported:
 - Datasets which have the same sentences for all languages and each language is separated in a subset or split,
 - Datasets with multiple language pairs in the same subset+split that need to be filtered using a field (column).
For example [Flores+](https://huggingface.co/datasets/openlanguagedata/flores_plus) has this two formats and cannot be used with the HF importer.


## Adding a new importer

Add Python code [here for parallel data importers](https://github.com/mozilla/translations/tree/main/pipeline/data/parallel_downloaders.py) or [here for monolingual ones](https://github.com/mozilla/translations/tree/main/pipeline/data/mono_importer.py)
