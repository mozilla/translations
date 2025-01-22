---
layout: default
title: Using pretrained models
nav_order: 9
---

# Using Pretrained Models

Pretrained models are machine learning models trained previously that can be used as the starting point for your training tasks.
Utilizing pretrained models can reduce training time and resource usage.

## Configuration Parameters

To download and use models from previous training runs or external sources, use the `pretrained-models` parameter in the training config. The keys in this parameter correspond to the training task `kinds` capable of using pretrained models. This is currently `train-teacher` and `train-backwards`. See [#515](https://github.com/mozilla/translations/issues/515) for `train-student` support.

```yaml
experiment:
  pretrained-models:
    # Continue training a teacher model.
    train-teacher:
      urls:
        # Replace the following {task_id} with the "train-teacher" task id.
        - https://firefox-ci-tc.services.mozilla.com/api/queue/v1/task/{task_id}/artifacts/public/build
      mode: continue
      type: default

    # Re-use an existing backwards model from a Google Cloud Storage bucket. This must
    # be the original (non-quantized) student model.
    train-backwards:
      urls:
        - https://storage.googleapis.com/releng-translations-dev/models/en-fi/opusmt/student/
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
