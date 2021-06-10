# Bergamot training
Training pipelines for Bergamot machine translation models.
The trained models are hosted in [bergamot-models](https://github.com/mozilla-applied-ml/bergamot-models/),
compatible with [bergamot-translator](https://github.com/mozilla/bergamot-translator) and can be used by
[bergamot-browser-extension](https://github.com/mozilla-extensions/bergamot-browser-extension).

The pipeline is capable of training a model for a language pair end to end.
Quality will depend on chosen datasets and data cleaning procedures. Some settings might require extra cleaning.
It was tested on relatively high resource language pair `ru-en`. Low resource pairs might require pipeline fixes.

## System requirements

- Ubuntu 18.04 (it can work on other Linux distributions, but might require `setup` scripts fixes).
- One or several Nvidia GPUs with CUDA drivers installed and at least 8 GB of memory.
- At least 16 CPU cores ( some steps of the pipelines utilize multiple cores pretty well, so the more the better).
- 64GB RAM
- 200+ GB of disk space ( mostly for datasets and transformations ). 
  It depends on chosen datasets and can be significantly higher.
  
It was tested on: 
- Ubuntu 18.04
- 56 core Xeon server
- 128 GB of RAM
- x8 NVIDIA RTX 2080 GPUs with 12 GB of memory
- CUDA 11.2
- 100 GB of local disk space
- Many terabytes of sshfs mounted storage

## Running

### Using a target Linux machine
```
git clone <this repo>
cd bergamot-training
# change settings in config.sh or modify code if needed
bash run.sh
```

To run a specific script, do:

```
source ./config.sh
bash ./pipeline/.../<script>.sh <args>
```


### Using Snakepit

See Snakepit installation (https://github.com/mozilla/snakepit-client)

#### To run end to end
```
git clone <this repo>
cd bergamot-training
# change settings in config.sh or modify code if needed
pit run --log "bergamot-training-ru-en" "[8:g2080]"
```

#### Interactive usage:

1. Create an empty directory 
2. Create file `.compute` like this:
```
# install any development dependencies
apt-get update
apt-get install -y tmux htop nano
curl -fsSL https://code-server.dev/install.sh | sh

while true; do : ; sleep 1000; done
```
3. Run `pit run "<user-name>-interactive" "[8:g2080]"`
4. Run `pit status` to check the job id
4. Run `pit exec <job-id> -- bash`
5. After attaching run `tmux`
6. (Optional) `code-server --bind-addr 0.0.0.0:8080` and 
   then `cat ~/.config/code-server/config.yaml` to conveniently edit files in a browser
7. To port forward, run in a separate terminal `pit forward <job-id> 8080 6006`
8. Change settings in config.sh or modify code if needed
9. `bash run.sh` to run end to end or
to run a specific script:
```
source ./config.sh
bash ./pipeline/.../<script>.sh <args>
```
 
#### To download exported models:

```
pit pull home bergamot-training/models/ru-en/exported/model.ruen.intgemm.alphas.bin.gz .
pit pull home bergamot-training/models/ru-en/exported/lex.50.50.ruen.s2t.bin.gz .
pit pull home bergamot-training/models/ru-en/exported/vocab.ruen.spm.gz .
```

## Pipeline steps

Step | Description | Bottleneck | Comments
--- | --- | --- | ---
Installation | Installing dependencies and compiling | CPU | Takes ~1 hour
Data downloading | Downloads datasets, samples sentences | Network, Disk | Time depends on dataset size, sampling of huge mono datasets (100M+ sentences) is the most intensive operation.
Data cleaning | Basic preprocessing, language specific, rule based, deduplication and other attempts to clean noisy data | CPU | Good parallelizaiton across CPU cores. To make cleaning of a new language more efficient add it to [clean_parallel.py](/pipeline/clean/clean_parallel.py).
Training s2s | Trains a backward shallow s2s model, which is useful for back-translations and ce-filtering | GPU | 
Augmentation with back-translations | Translates mono corpus combined from `MONO_DATASETS_TRG` using shallow s2s model. | GPU | It is more useful for low-resource languages and can be skipped for others.
Training teacher | Trains one or multiple big transformer models | GPU | You might want to adjust [early stopping](pipeline/train/configs/training/teacher.transformer.train.yml) parameters depending on datasets size.
Translation with a teacher | Translates a corpus and monolingual data combined from `MONO_DATASETS_SRC` using the teacher model (ensemble is not supported yet) | GPU | The slowest part of the pipeline. Can take days. It is possible to speed it up launching the same scripts ([corpus](pipeline/translate/translate-corpus.sh), [mono](pipeline/translate/translate-mono.sh)) in parallel from another machine with access to the same network directory.
Cross-entropy filtering | Scores translated corpus with backward s2s model and removes a part of the corpus with the lowest scores to reduce noise | GPU, CPU, Disk | At this point we work with huge datasets, so it utilizes copying to a local disk to make things faster.
Training alignments and shortlist | Trains alignments using [fast_align](https://github.com/clab/fast_align) and extracts lexical shortlist using [extract_lex](https://github.com/marian-nmt/extract-lex) tool | CPU, Disk | Some tools requires uncompressed datasets on disk and they are huge at this point. Data is copied to a local disk to make things faster. Might take 100+GB of local disk depending on a dataset size. Good CPU parallelization.
Training student | Trains a small transformer student model on filtered data and using alignments | GPU | Run [Tensorboard](pipeline/train/tensorboard/tensorboard.sh) manually to see training visualization.
Fine-tuning student | Finetunes the student model by emulating 8bit GEMM during training | GPU | Converges very quickly and then degrades. It's quick but you might want to reduce early stopping threshold.
Quantizaiton |  Applies 8 bit quantization to the fined-tuned student model and evaluates on CPU | CPU | CPU threads must be set to 1 for this step.
Export | Exports trained model and shortlist to (bergamot-translator)(https://github.com/mozilla/bergamot-translator) format | |

## Datasets importers

Dataset importers can be used in `TRAIN_DATASETS, DEVTEST_DATASETS, MONO_DATASETS_SRC, MONO_DATASETS_TRG` config settings.

Example:
```
TRAIN_DATASETS="opus_OPUS-ParaCrawl/v7.1 mtdata_newstest2019_ruen"
```

Data source | Prefix | Name example | Type | Comments
--- | --- | --- | ---| ---
[MTData](https://github.com/thammegowda/mtdata) | mtdata | newstest2017_ruen | corpus | Supports many datasets. Run `mtdata list -l ru-en` to see datasets for a specific language pair.
[OPUS](opus.nlpl.eu/) | opus | OPUS-ParaCrawl/v7.1 | corpus | Many open source datasets. Go to the website, choose a language pair, check links under Moses column to see what names and version is used in a link.
[Paracrawl](https://paracrawl.eu/) | paracrawl-mono | paracrawl8 | mono | Datasets from a crawled web. Only [mono datasets](https://paracrawl.eu/index.php/moredata) are used in this importer. Corpus is available using opus importer.
[News crawl](http://data.statmt.org/news-crawl) | news-crawl | news.2019 | mono | Some news monolingual datasets from [WMT21](https://www.statmt.org/wmt21/translation-task.html)
[Common crawl](https://commoncrawl.org/) | commoncrawl | wmt16 | mono | Huge web crawl datasets. The links are posted on [WMT21](https://www.statmt.org/wmt21/translation-task.html)

## Evaluation datsets

Only [SacreBLEU](https://github.com/mjpost/sacrebleu) datasets are supported at the moment.

Example:
```
TEST_DATASETS="wmt20 wmt18"
```

To see what datasets are available for a language pair (for example, `ru-en`) run:
```
sacrebleu --list -l ru-en
```

## Development

- All scripts work with respect to repo root directory which should be written to `WORKDIR` environment variable. 
  It allows to not think about relative paths and execution folders.
  
- Scripts inside `pipeline` directory are independent and operate only using input arguments 
  and global envs from `config.sh`.
  They don't use any extra knowledge of data naming or locations.
  
- All scripts have a description and definition of input arguments.

- All scripts test expected environment variables early.

- If a script step fails, it can be safely retried.

- Ideally every script should start from the last unfinished step, 
  checking presence of intermediate results of previous steps.

- A script fails as early as possible.

- Maximum bash verbosity is set for easy debugging.

- Input data is always read only.

- Output data is placed to a new folder for script results.
  
- It is expected that the specified output folder might not exist and should be created by the script.

- A script creates a folder for intermediate files and cleans it in the end.

- Network disks are too slow for some operations, so a script can copy and work with intermediate data on a local disk.
  This ability is limited by a local disk size (this is the case for Snakepit cluster).
  An exception is when parallelization across multiple machines is required.
    
- Global variables are upper case, local variable are lower case.

- ALl variables that are global for the whole pipeline are set in `config.sh`.

- Scripts should automatically inspect resources available for computation and utilize them to make things faster
  (number of cores, memory).
  
## TODO

1. Add [bicleaner](https://github.com/bitextor/bicleaner/)
2. Add translation with an ensemble of teacher models

## Used projects and tools

### Training
https://marian-nmt.github.io

https://github.com/google/sentencepiece

https://github.com/clab/fast_align

https://github.com/marian-nmt/extract-lex

### Evaluation
https://github.com/mjpost/sacrebleu

### Pipeline recipes
https://github.com/marian-nmt/marian-examples/tree/master/wmt2017-transformer

https://github.com/browsermt/students/tree/master/train-student


### Workflow
https://github.com/mozilla/snakepit-client

https://github.com/ufal/marian-tensorboard

### Data
https://opus.nlpl.eu/ 

https://paracrawl.eu/

https://commoncrawl.org/

https://www.statmt.org/wmt21/translation-task.html

https://www.statmt.org/wmt20/translation-task.html

https://github.com/thammegowda/mtdata