# OpusTrainer


[OpusTrainer](https://github.com/hplt-project/OpusTrainer) is a training tool developed by the HPLT project. 
It feeds training data to Marian and provides the ability to do useful manipulations with the data, 
such as shuffling, mixing multiple datasets in the specified proportion, splitting training into multiple stages and augmentation.

See [this paper](https://arxiv.org/pdf/2311.14838.pdf) for more details and recommendations on how to set augmentation values.

## Data augmentation

Data augmentation helps make translation models more robust, which is especially useful for usage with noisy internet pages.

OpusTrainer augments data on the fly, meaning it will generate unique data for each epoch of training.

Supported augmentations:
- **UpperCase** - make some sentences from the dataset upper case
- **TitleCase** - use title case for some sentences from the dataset
- **RemoveEndPunct** - removes terminal punctuation mark from the source and target sentences if it matches by type (e.g. `.` and `。`)
- **Typos** - add random typos in some words
- **Noise** - insert lines with random unicode noise
- **Tags (inline noise)** - add emojis and other random Unicode symbols in the source and target sentences in the appropriate positions
  (requires whitespace tokenized alignments for the training corpus)

It is possible to specify the probability of augmentation 
(which will roughly correspond to the percentage of augmented sentences):
```yaml
modifiers:
- UpperCase: 0.1 # Apply randomly to 10% of sentences
```

See [OpusTrainer Readme](https://github.com/hplt-project/OpusTrainer?tab=readme-ov-file#modifiers) for detailed documentation.

## Curriculum learning

Curriculum learning is the ability to split training into multiple stages. Each stage is configurable to use a mix of different datasets.

We use it to pretrain the teacher model on the augmented dataset that includes the original parallel corpus and 
back-translations and then continue training on the original parallel corpus only
(see [teacher config](https://github.com/mozilla/translations/tree/main/pipeline/train/configs/opustrainer/teacher.yml)) and `curriculum key`.

To switch to a one stage training use a config option:

```yaml
experiment:
  ...
  teacher-mode: "one-stage"
```
This is useful when the model stops training too early on the fine-tuning stage which usually indicates having a high quality back-translated data and noisy original parallel data.
It likely will be the case when using a pre-trained student model as a backward model as it has higher quality than a shallow s2s model that we train as a part of the pipeline.

## Configuration

OpusTrainer configuration files for the trained models are located in 
the [/pipeline/train/configs/opustrainer/](https://github.com/mozilla/translations/tree/main/pipeline/train/configs/opustrainer/) directory. There are also a few custom keys used by the `pipeline/train/train.py`
script to configure training. They should be documented in the config.

`{dataset0}`, `{dataset1}` and `{vocab}` will be replaced by the training datasets and a path to Sentencepiece `vocab.spm` passed in `pipeline/train/train.py` script.

See more details on configuration in the OpusTrainer [readme](https://github.com/hplt-project/OpusTrainer).

#### Tokenization and alignments

`Tags` modifiers requires whitespace, Moses or ICU tokenized alignments as input. 
Marian requires Sentencepiece tokenized alignments and raw text input. 
To make them compatible `Tags` modifier can remap the alignments in the end using the passed Sentencepiece model `spm_vocab_*: vocab.spm` (student model use case). 
If the `spm_vocab_trg` argument is missing `Tags` modifier will remove alignments and output only the parallel sentences (teacher model use case). 

Currently, ICUs-tokenized text and its alignments are passed to OpusTrainer (to work around CJK languages where whitespace-based tokenization doesn't make sense). 
Whitespaces are represented with a special symbol "▁" to allow for lossless text reconstruction on OpusTrainer side. 
`custom_detok_icu:{src,trg}` OpusTrainer modifiers are applied to detokenize text after inline noise is added. 
Then the detokenized text is passed to Marian together with the alignments remapped to SentencePiece tokenization.

## Models

Current strategy is to run as many supported augmentations as possible for the teacher 
and student models and skip augmentaiton entirely for the backward model. 
This is mostly based on the intuition that we do not need the backward model to be robust and would rather prioritize quality that is usually affected by the noisier data.
Even though the student is supposed to learn on the exact output of the teacher model, training on augmented data seems to be working in practice.

We might rethink this strategy in future after running more experiments.


## Evaluation

To test the effects of the data augmentation on the trained models, the data downloader supports augmentation of the evaluation datasets.
It allows running the validation while training and the final evaluation on an augmented datasets.

Add an augmentation modifier to any dataset in the training config in the following format:

`<dataset-importer>_<augmentation-modifier>_<dataset-name>`

For example:

```yaml
- flores_aug-title-strict_devtest
- sacrebleu_aug-mix_wmt19/dev
- opus_aug-typos_ada83/v1
```


### Supported modifiers

`aug-typos` - applies 4 random typos to all sentences in the dataset

`aug-title` - applies title case to the whole dataset

`aug-upper` -  applies upper case to the whole dataset

`aug-punct` -  applies modification of punctuation

`aug-noise` -  generates extra lines with noise (1 line of noise for each line of the dataset, so the dataset becomes twice longer)

`aug-inline-noise` -  inserts the same random noise in the appropriate positions of the source and target sentences based on dynamically generated alignments. 
It uses unsupervised aligner [SimAlign](https://github.com/cisnlp/simalign) which is based on BERT and quite slow, 
so it should only be used on small evaluation datasets.

`aug-mix` - applies all the existing modifiers with 0.05 probability each. Only
modifiers that work for the language's script will be chosen.

### Example training config
```yaml
  # datasets for validation while training
  devtest:
    - flores_aug-mix_dev
    - sacrebleu_aug-mix_wmt19/dev
  # datasets for the final evaluation
  test:
    - flores_devtest
    - flores_aug-mix_devtest
    - flores_aug-title_devtest
    - flores_aug-upper_devtest
    - flores_aug-punct_devtest
    - flores_aug-typos_devtest
    - flores_aug-noise_devtest
    - flores_aug-inline-noise_devtest
```

### Language scripts and augmentations

Not all augmentations can be applied to all types of scripts. For instance, it doesn't make sense to apply spelling errors to Chinese characters, which are singular and ideographic. While an alphabetic text will benefit from having the spellings scrambled. Not all languages have uppercase, and lowercase. The model will learn unregulated behavior if the target sentences are a mix of upper and lower case, if the source sentence doesn't have any casing information. However, in the opposite direction it is fine for different casing to translate to the same non-cased translation.

See [pipeline/data/lang_script.py](https://github.com/mozilla/translations/blob/main/pipeline/data/lang_script.py) for detailed information about script types.
