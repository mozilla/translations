# Dataset importers

Dataset importers can be used in `datasets` sections of the [training config](https://github.com/mozilla/translations/tree/main/configs/config.test.yml).

Example:
```
  train:
    - opus_ada83/v1
    - mtdata_newstest2014_ruen
```

Data source | Prefix     | Name examples | Type | Comments
--- |------------| --- | ---| ---
[MTData](https://github.com/thammegowda/mtdata) | mtdata     | newstest2017_ruen | corpus | Supports many datasets. Run `mtdata list -l ru-en` to see datasets for a specific language pair.
[OPUS](opus.nlpl.eu/) | opus       | ParaCrawl/v7.1 | corpus | Many open source datasets. Go to the website, choose a language pair, check links under Moses column to see what names and version is used in a link.
[SacreBLEU](https://github.com/mjpost/sacrebleu) | sacrebleu  | wmt20 | corpus | Official evaluation datasets available in SacreBLEU tool. Recommended to use in `datasets:test` config section. Look up supported datasets and language pairs in `sacrebleu.dataset` python module.
[Flores](https://github.com/facebookresearch/flores) | flores     | dev, devtest | corpus | Evaluation dataset from Facebook that supports 100 languages.
Custom parallel | url        | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.[LANG].zst` | corpus | A custom zst compressed parallel dataset, for instance uploaded to GCS. The language pairs should be split into two files. the `[LANG]` will be replaced with the `to` and `from` language codes.
[News crawl](http://data.statmt.org/news-crawl) | news-crawl | news.2019 | mono | Monolingual news datasets from [WMT](https://www.statmt.org/wmt21/translation-task.html)
[OPUS](opus.nlpl.eu/) | opus       | tldr-pages/v2023-08-29 | mono | Monolingual dataset from OPUS.
[HPLT](https://hplt-project.org/datasets/v2.0) | hplt       | mono/v2.0 | mono | HPLT monolingual corpus (mostly from Internet Archive, but also from Common Crawl).
Custom mono | url        | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.ru.zst` | mono | A custom zst compressed monolingual dataset, for instance uploaded to GCS.

You can also use [find-corpus](https://github.com/mozilla/translations/tree/main/pipeline/utils/find-corpus.py) tool to find all datasets for an importer and get them formatted to use in config.

Set up a local [poetry](https://python-poetry.org/) environment.
```
task find-corpus -- en ru
```
Make sure to check licenses of the datasets before using them.

## Adding a new importer

Just add a shell script to [corpus](https://github.com/mozilla/translations/tree/main/pipeline/data/importers/corpus) or [mono](https://github.com/mozilla/translations/tree/main/pipeline/data/importers/mono) which is named as `<prefix>.sh` 
and accepts the same parameters as the other scripts from the same folder.
