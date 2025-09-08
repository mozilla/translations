---
layout: default
title: Using pretrained models
nav_order: 9
---

# Using Pretrained Models

Pretrained models are machine learning models trained previously that can be used as the starting point for your training tasks.
Utilizing pretrained models can reduce training time and resource usage.

## Configuration Parameters

To download and use models from previous training runs or external sources, use the `continuation.models` parameter in the training config. The keys in this parameter correspond to the training task `kinds` capable of using pretrained models. This is currently `train-teacher-model` and `backtranslations-train-backwards-model`. See [#515](https://github.com/mozilla/translations/issues/515) for `distillation-student-model-train` support.

```yaml
continuation:
  models:
    # Continue training a teacher model.
    teacher:
      urls:
        # Replace the following {task_id} with the "train-teacher-model" task id.
        - https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/{task_id}/artifacts/public/build
      mode: continue
      type: default

    # Re-use an existing backwards model from a Google Cloud Storage bucket. This must
    # be the original (non-quantized) student model.
    backwards:
      url: https://storage.googleapis.com/releng-translations-dev/models/en-fi/opusmt/student/
      mode: use
      type: default
```

To find models from older training runs see the `gs://releng-translations-dev/models` bucket.

For instance you can see the models available for the following commands:

```sh
gsutil ls gs://releng-translations-dev/models
```

And then use the URLs from:

```sh
gs://releng-translations-dev/models/en-fi/opusmt/student
```

This directory should contain the various `.npz` and `.decoder.yml` for the models, as well as the `vocab.spm`. If the `vocab.spm` is not present then run something like:

```sh
gsutil cp \
  gs://releng-translations-dev/models/en-fi/opusmt/vocab/vocab.spm \
  gs://releng-translations-dev/models/en-fi/opusmt/student/vocab.spm
```

### The URLs Key

The `urls` key is a list that specifies the locations from which the pretrained models are downloaded.

### The Mode Key

#### Use Mode

In `use` mode, the pipeline only downloads the model without further training. The tasks that depend on the training task will use the downloaded model artifacts as they are.

#### Continue Mode

In `continue` mode the pipeline uses the downloaded model artifacts from the previous training run as a "checkpoint" and continues training. This is useful to `continue` training a model on the same corpus.

#### Init Mode

In `init` mode, the pipeline initializes model weights with the downloaded model using the `--pretrained-model` flag in `marian`. This is useful for fine-tuning an existing model on a different corpus.

### The Type Key

`default` is the `npz` format that we are using for the model artifacts, this was added with `opusmt` in mind.

## Recipes

### Train a new teacher

If a teacher needs to be retrained, it can use an existing corpus.

```yaml
continuation:
  vocab:
    src: https://example.com/vocab.ru.spm
    trg: https://example.com/vocab.en.spm
  models:
    backwards:
      url: https://example.com/ru-en/backwards
      mode: use
      type: default
  corpora:
    backtranslations:
      src: https://example.com/backtranslations.ru.zst
      trg: https://example.com/backtranslations.en.zst
      # Optional:
      tok-src: https://example.com/backtranslations.tok-icu.ru.zst
      tok-trg: https://example.com/backtranslations.tok-icu.en.zst
      alignments: https://example.com/backtranslations.aln.zst
    parallel:
      src: https://example.com/parallel.ru.zst
      trg: https://example.com/parallel.en.zst
      # Optional:
      tok-src: https://example.com/parallel.tok-icu.ru.zst
      tok-trg: https://example.com/parallel.tok-icu.en.zst
      alignments: https://example.com/parallel.aln.zst
```

### Generate distillation data and train a new student

After training a teacher model, this continuation configuration will create the
distillation corpus, and continue on to train the student model. Note that a backwards
model is still required for scoring the distillation data.

```yaml
continuation:
  vocab:
    src: https://example.com/vocab.ru.spm
    trg: https://example.com/vocab.en.spm
  models:
    backwards:
      url: https://example.com/ru-en/backwards
      mode: use
      type: default
    teachers:
      urls:
        - https://example.com/ru-en/teacher
      mode: use
      type: default
  corpora:
    backtranslations:
      src: https://example.com/backtranslations.ru.zst
      trg: https://example.com/backtranslations.en.zst
      # Optional:
      tok-src: https://example.com/backtranslations.tok-icu.ru.zst
      tok-trg: https://example.com/backtranslations.tok-icu.en.zst
      alignments: https://example.com/backtranslations.aln.zst
    parallel:
      src: https://example.com/parallel.ru.zst
      trg: https://example.com/parallel.en.zst
      # Optional:
      tok-src: https://example.com/parallel.tok-icu.ru.zst
      tok-trg: https://example.com/parallel.tok-icu.en.zst
      alignments: https://example.com/parallel.aln.zst
```


### Distill a student from existing data

If the existing distillation corpus is available, this configuration will allow for just
distilling a new student. Note that the student can be a completely different architecture.
   
```yaml    
continuation:
  vocab:
    src: https://example.com/vocab.ru.spm
    trg: https://example.com/vocab.en.spm
  corpora:
    distillation:
      src: https://example.com/distillation.ru.zst
      trg: https://example.com/distillation.en.zst
      # Optional:
      tok-src: https://example.com/distillation.tok-icu.ru.zst
      tok-trg: https://example.com/distillation.tok-icu.en.zst
      alignments: https://example.com/distillation.aln.zst
```

### Finetune an existing model

Sometimes we want to experiment with fine-tuning of an existing student or teacher model on an external corpus.
Find a Taskcluster task for `distillation-student-train` and copy Task ID for the student model and the vocab. 
Use `init` mode. Marian will initialize the model weights using the final output model of the task.

```yaml  
continuation:
  models:
    student:
      url: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/VF2YHgQXQ4SOKUjnMLzzhw/artifacts/public/build
      mode: init
      type: default
  vocab:
    src: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/VF2YHgQXQ4SOKUjnMLzzhw/runs/0/artifacts/public%2Fbuild%2Fvocab.en.spm
    trg: https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/VF2YHgQXQ4SOKUjnMLzzhw/runs/0/artifacts/public%2Fbuild%2Fvocab.ja.spm
  corpora:
    distillation:
      src: https://storage.googleapis.com/releng-translations-dev/data/llm/en-ja_JP/qwen-3-235b-a22b-fp8-vllm/qererank8/diverse_sample.10M.filtered2.en.zst
      trg: https://storage.googleapis.com/releng-translations-dev/data/llm/en-ja_JP/qwen-3-235b-a22b-fp8-vllm/qererank8/diverse_sample.10M.filtered2.ja.zst

```