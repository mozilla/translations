---
layout: default
title: Pipeline steps
nav_order: 3
has_children: true
---

# Pipeline steps

The pipeline steps are based on the [distillation-student-model-train](https://github.com/browsermt/students/tree/master/distillation-student-model-train)
recipe. For descriptions of every task see [Task Descriptions](./task-descriptions.md). For a visualization of the pipeline see [Training Pipeline DAGs](https://docs.google.com/presentation/d/1HkypImI_hbA3n1ljU57ZPAzW8PuQqdv2wrXqj688KtQ/edit?slide=id.g3421e8f521e_1_419#slide=id.g3421e8f521e_1_419) which visually breaks down the various steps.

## Toolchain

Installs dependencies and compiles Marian and other tools.

## Data downloading

Downloads datasets and samples an appropriate amount from the sentences. The time
depends on dataset size. The sampling of huge mono datasets (100M+ sentences) is the most intensive operation.

Uses `datasets.train` configuration section.

## Analyze data

Runs data analysis on the downloaded datasets and outputs charts.
For example a distribution of sentence length in a dataset.

## Data cleaning

Basic preprocessing, dataset specific, language specific, rule based and other attempts to clean noisy data in parallel
and monolingual datasets.
Good parallelization across CPU cores. To make cleaning of a new language more efficient add it
to [clean_parallel.py](https://github.com/mozilla/translations/tree/main/pipeline/clean/tools/clean_parallel.py).

Alternatively use [OpusCleaner](../data-and-cleaning/index.md#opuscleaner).

## Bicleaner AI

Filters noisy sentence pairs in a parallel corpus using [bicleaner-ai](https://github.com/bitextor/bicleaner-ai)
classifier.
Cleaning thresholds are configurable per dataset.

See more details on the [Bicleaner](../data-and-cleaning/bicleaner.md) page.

## Merge and dedupe

Merges clean datasets into one and applies deduplication.

## Training vocabulary

Trains [SentencePiece](https://github.com/google/sentencepiece) vocabulary/tokenizer model on parallel corpus.

See more details on choosing the size of vocabulary [here](vocab-size.md).

## Training backward model

Trains a shallow sequence to sequence RNN model in an opposite direction. It is useful for back-translations and cross
entropy filtering.
It is based on
a [marian example](https://github.com/marian-nmt/marian-examples/tree/master/training-basics-sentencepiece).

## Augmentation with back-translations

Translates monolingual corpus combined from monolingual datasets in target language using the backward model.

It is more useful for low-resource languages but is still recommended for high-resource ones as well.

## Training alignments

Trains whitespace-tokenized alignments accepted by OpusTrainer using [eflomal](https://github.com/robertostling/eflomal)
library.

It trains alignments separately for origninal parallel, backtranslated and student corpus.
Backtranslated and student steps use eflobal priors extracted from the alignments trained for the original parallel
corpus.
It can improve accuracy for a smaller corpus as well as performance.

It works with uncompressed datasets, so it can be heavy on disk.

## Training teacher

Trains several big transformer models on the augmented dataset. They will be later used for decoding as an ensemble of
models.

## Translation by teachers

Translates the corpus and the monolingual data in source language (configurable in `datasets.mono-src`) using the
ensemble of teacher models.

This is the heaviest part of the pipeline but highly parallelizable.

## Cross-entropy filtering

Scores the translated corpus with the backward model and removes a part of the corpus with the lowest scores to reduce
noise.

At this point we work with huge datasets that can be very disk intensive.

## Training shortlist

Trains SentencePiece tokenized alignments using [elfomal](https://github.com/robertostling/eflomal) similar to the
alignments steps and then
extracts lexical shortlist using [extract_lex](https://github.com/marian-nmt/extract-lex) tool.

Some tools require uncompressed datasets on disk, and they are huge at this point. Good CPU parallelization.

## Training student

Trains a small transformer student model on the filtered data and using the alignments.
OpusTrainer remaps the alignment to SentencePiece. See more details on the [OpusTrainer page](opus-trainer.md).

## Fine-tuning student

Fine-tunes the student model by emulating 8bit GEMM during training.
Converges very quickly and then degrades.

## Quantization

Applies 8 bit quantization to the fined-tuned student model.
Marian CPU threads must be set to 1 for this step.

## Evaluation

Calculates metrics for all models (BLEU, chrF) using [SacreBLEU](https://github.com/mjpost/sacrebleu).
It runs Marian decoding GPU for all models except the quantized ones that it runs on CPU.

It uses `datasets.test` configuration section.

## Export

Exports the trained model and the shortlist to (bergamot-translator)(https://github.com/mozilla/bergamot-translator)
format.

## Resource usage

 Step                                | Bottleneck     
-------------------------------------|----------------
 Installation                        | CPU            
 Data downloading                    | Network, Disk  
 Analyze data                        | CPU, Disk      
 Data cleaning                       | CPU            
 Bicleaner                           | GPU            
 Merge and dedupe                    | CPU, Disk      
 Training vocabulary                 | CPU            
 Training backward model             | GPU            
 Augmentation with back-translations | GPU            
 Training alignments                 | CPU, Disk      
 Training teacher                    | GPU            
 Translation by teacher              | GPU            
 Cross-entropy filtering             | GPU, CPU, Disk 
 Training shortlist                  | CPU, Disk      
 Training student                    | GPU            
 Fine-tuning student                 | GPU            
 Quantizaiton                        | CPU            
 Evaluation                          | GPU            
 Export                              |                                                                 
