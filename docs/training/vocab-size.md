---
layout: default
title: Vocab Size
parent: Pipeline steps
---

# Vocab Size

The vocab size can be changed in the config via `experiments.spm-vocab-size`. The vocab is shared between the source and the target languages. The vocab is built by [SentencePiece](https://github.com/google/sentencepiece).

In "Finding the Optimal Vocabulary Size for Neural Machine Translation", the
researchers tests vocab sizes between 500 and 64,000. A general value table
depends on the vocabulary size.

| Corpus Size | Optimal vocab size |
| ----------- | ------------------ |
|   4,500,000 |    32,000 - 64,000 |
|   1,000,000 |              8,000 |
|     500,000 |     4,000 - 16,000 |
|      30,000 |              1,000 |

See section 4 for their conclusions.
  https://aclanthology.org/2020.findings-emnlp.352.pdf

The larger the vocab, the slower the training and the inference will be as each
token in the vocab contributes to the amount of probabilities that will need
to be generated when making a token prediction. This is a trade off on the quality
of the translation, and the performance.

We usually use 32k by default and 64k for languages with significantly different scripts (e.g. CJK) when using a shared vocabulary 
(using separate vocabularies for CJK proved to be better, so it's now always 32k).
