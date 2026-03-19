---
layout: default
title: Bicleaner
parent: Data cleaning
---
# Bicleaner


[Bicleaner AI](https://github.com/bitextor/bicleaner-ai) is a tool that aims at detecting noisy sentence pairs in a parallel corpus. 
The classifier scores parallel sentences from 0 to 1 where 0 means a very noisy translation and 1 is a good translation.
If a specialized model for a language pair is not available it will fallback to downloading a multilingual en-xx model.

For supported languages see:

  * [Bicleaner AI Releases][ai-releases]

## How to configure for training

The configuration specifies a default threshold and a per-dataset threshold. A sentence pair will be kept if its score is **above** the given threshold.

- `0.5` should be a [good default value].
- Increase the threshold for noisier datasets.
- Set the threshold to `0` to skip cleaning entirely.

## Recommendations for specific datasets

| Data set      | Threshold | Reason                   |
| ------------- | --------- | -------                  |
| OpenSubtitles | 0.8       | This is a noiser dataset |
| ParaCrawl     | 0         | This dataset has already been cleaned by bicleaner. See [Bicleaner AI: Bicleaner Goes Neural], section 4.2.2 |

## Example config:

```
  bicleaner:
    default-threshold: 0.5
    dataset-thresholds:
      opus_CCAligned/v1: 0.7
      opus_OpenSubtitles/v2018: 0.8
      opus_ParaCrawl/v9: 0
      ...
```

[good default value]: https://github.com/bitextor/bicleaner-ai/wiki/How-to-train-your-Bicleaner-AI#bicleaning-a-corpus
[ai-releases]: https://github.com/bitextor/bicleaner-ai-data/releases
[releases]: https://github.com/bitextor/bicleaner-data/releases
[Bicleaner AI: Bicleaner Goes Neural]: https://aclanthology.org/2022.lrec-1.87.pdf


## Models and caching

In the current implementation an appropriate model is downloaded for a language pair on demand and then cached. 
For example for `en-ru` a multilingual `en-xx` will be downloaded since a dedicated model is not available for this pair.
For `pt-en` `en-pt` will be downloaded since all the models have English as the first language.

The downloaded model will be cached in Taskcluster under the requested language pair (`en-ru`, `pt-en`). 
If a new model is added to [Hugging Face repo](https://huggingface.co/bitextor) it would be a good idea to invalidate
the caches manually by editing `pipeline/bicleaner/download_pack.py`. 
We do not do this automatically in the current implementation. We will rethink this strategy if this happens often.
