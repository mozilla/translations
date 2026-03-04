---
layout: default
title: Snakemake
nav_order: 2
parent: Orchestrators
---

# Snakemake

This section included the instructions on how to run the pipeline 
using [Snakemake](https://snakemake.github.io/) orchestrator (locally or on a Slurm cluster).

**NOTICE: Mozilla has switched to Taskcluster for model training, and the Snakemake pipeline is not maintained. 
Feel free to contribute if you find bugs.**

Snakemake workflow manager infers the DAG of tasks implicitly from the specified inputs and outputs of the steps. 
The workflow manager checks which files are missing and runs the corresponding jobs either locally or on a cluster depending on the configuration. 

Snakemake parallelizes steps that can be executed simultaneously. 
It is especially useful for teacher ensemble training and translation.

The main Snakemake process (scheduler) should be launched interactively. 
It runs the job processes on the worker nodes in cluster mode or on a local machine in local mode.


## System requirements

### Local mode

- Ubuntu 18.04 (it can work on other Linux distributions, but might require `setup` scripts fixes; see more details in [marian installation instructions](https://marian-nmt.github.io/quickstart/)).
- One or several Nvidia GPUs with CUDA drivers installed and at least 8 GB of memory.
- CUDNN installed
- At least 16 CPU cores ( some steps of the pipeline utilize multiple cores pretty well, so the more the better).
- 64 GB RAM (128 GB+ might be required for bigger datasets)
- 200+ GB of disk space ( mostly for datasets and transformations ). 
  It depends on chosen datasets and can be significantly higher.
  
It was tested on: 
- Ubuntu 18.04
- 56 core Xeon server
- 128 GB of RAM
- x8 NVIDIA RTX 2080 GPUs with 12 GB of memory
- CUDA 11.2
- 100 GB of local disk space
- Many terabytes of NFS mounted storage

### Cluster mode

- Slurm cluster with CPU and Nvidia GPU nodes
- CUDA 11.2 ( it was also tested on 11.5)
- CUDNN library installed
- Singularity module if running with containerization (recommended)
- If running without containerization, there is no procedure to configure the environment automatically.
  All the required modules (for example `parallel`) should be preinstalled and loaded in ~/.bashrc

It was tested on Mozilla Slurm cluster using Singularity containers.
The pipeline can also be launched on [CSD3 HPC](https://docs.hpc.cam.ac.uk/hpc/index.html) but it was not fully tested.

### Cloud mode

Snakemake workflows can work on Kubernetes, Google Cloud Life Sciences and other cloud platforms. 
The pipeline was not tested in this mode and might require modification.

Please refer to [Cloud execution](https://snakemake.readthedocs.io/en/stable/executing/cloud.html) section of Snakemake documentation.

It is also possible to deploy Slurm cluster in the cloud. For example, using [Slurm on Google Cloud Platform](https://github.com/SchedMD/slurm-gcp).

## Configuration

0. Clone the repo:
``` 
git clone https://github.com/mozilla/translations.git
cd translations
```
1. Choose a [Snakemake profile](https://github.com/Snakemake-Profiles) from `profiles/` or create a new one 
2. Adjust paths in the `Makefile` if needed and set `PROFILE` variable to the name of your profile
3. Adjust Snakemake and workflow settings in the `profiles/<profile>/config.yaml`, see [Snakemake CLI reference](https://snakemake.readthedocs.io/en/stable/executing/cli.html) for details
4. Configure experiment and datasets in `configs/config.prod.yml` (or `configs/config.test.yml` for test run)
5. Change source code if needed for the experiment
6. **(Cluster mode)** Adjust cluster settings in the cluster profile.
   For `slurm-moz`: `profiles/slurm-moz/config.cluster.yml`
   You can also modify `profiles/slurm-moz/submit.sh` or create a new Snakemake [profile](https://github.com/Snakemake-Profiles).
7. **(Cluster mode)** It might require further tuning of requested resources in `Snakemake` file:
    - Use `threads` for a rule to adjust parallelism
    - Use `resources: mem_mb=<memory>` to adjust total memory requirements per task 
      (default is set in `profiles/slurm-moz/config.yaml`)

## Installation

See also [Snakemake installation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)

1. Install Mamba - fast Conda package manager

```
make conda
```

2. Install Snakemake

```
make snakemake
```

3. Update git submodules

```
make git-modules
```

4. (Optional) Install Singularity if running with containerization 

Local mode: See [Singularity installation](https://sylabs.io/guides/3.8/user-guide/quick_start.html), requries root

Cluster mode: 

For example,
```
module load singularity
```
but the way to load Singularity depends on cluster installation

5. (Optional) Prepare a container image if using Singularity

    
Either pull the prebuilt image:

```
make pull
```

Or build it (requires root):

```
make build
```

## Running

Dry run first to check that everything was installed correctly:

```
make dry-run
```

To run the pipeline:
```
make run
```

To test the whole pipeline end to end (it is supposed to run relatively quickly and does not train anything useful):

```
make test
```
You can also run a speicific profile or config by overriding variables from Makefile
```
make run PROFILE=slurm-moz CONFIG=configs/config.test.yml
```

### Specific target

By default, all Snakemake rules are executed. To run the pipeline up to a specific rule use:
```
make run TARGET=<non-wildcard-rule-or-path>
```
For example, collect corpus first:
```
make run TARGET=merge_corpus
```

You can also use the full file path, for example:
```
make run TARGET=/models/ru-en/bicleaner/teacher-base0/model.npz.best-ce-mean-words.npz
```
### Rerunning

If you want to rerun a specific step or steps, you can delete the result files that are expected in the Snakemake rule output.
Snakemake might complain about a missing file and suggest to run it with `--clean-metadata` flag. In this case run:
```
make clean-meta TARGET=<missing-file-name>
```
and then as usual:
```
make run
```

### Reporting

To create a Snakemake [html report](https://snakemake.readthedocs.io/en/stable/snakefiles/reporting.html), run:
```
make report
```


### Tensorboard

It is possible to look at the training graphs in Tensorboard.

Adjust the path to match the model directories in makefile `tensorboard`  command and remove `--offline` to automtically update while training.

Run server:

```bash
task tensorboard
```

Then go to `http://localhost:6006` in the browser

![Tensorboard](../assets/tensorboard.png)

Known issue: the [marian-tensorboard](https://github.com/marian-nmt/marian-tensorboard) tool we're using
parses the trainig logs only for the student models and validation logs for all models for some reason.


### Results

See [Directory Structure](#directory-structure) section.

The main directories inside `SHARED_ROOT` are:
- `data/<lang_pair>/<experiment>` - data produced by the pipeline jobs
- `logs/<lang_pair>/<experiment>` - logs of the jobs for troubleshooting
- `experiments/<lang_pair>/<experiment>` - saved experiment settings for future reference
- `models/<lang_pair>/<experiment>` - all models produced by the pipeline. The final compressed models are in `exported` folder.

#### Exported models example

```
/models/ru-en/test/exported/model.ruen.intgemm.alphas.bin.gz
/models/ru-en/test/exported/lex.50.50.ruen.s2t.bin.gz
/models/ru-en/test/exported/vocab.ruen.spm.gz
```


## Directory structure
    
    ├ data
    │   └ ru-en
    │      └ test
    │        ├ original
    │        │   ├ corpus
    │        │   │   ├ mtdata_JW300.en.gz
    │        │   │   └ mtdata_JW300.ru.gz
    │        │   ├ devset
    │        │   │   ├ flores_dev.en.gz
    │        │   │   └ flores_dev.ru.gz
    │        │   ├ eval
    │        │   │   ├ sacrebleu_wmt20.en.gz
    │        │   │   └ sacrebleu_wmt20.ru.gz
    │        │   ├ mono
    │        │   │   ├ news-crawl_news.2020.ru.gz
    │        │   │   └ news-crawl_news.2020.en.gz
    │        │   ├ devset.ru.gz
    │        │   └ devset.en.gz
    │        ├ clean
    │        │   ├ corpus
    │        │   │   ├ mtdata_JW300.en.gz
    │        │   │   └ mtdata_JW300.ru.gz
    │        │   ├ mono
    │        │   │   ├ news-crawl_news.2020.ru.gz
    │        │   │   └ news-crawl_news.2020.en.gz
    │        │   ├ mono.ru.gz
    │        │   └ mono.en.gz
    │        ├ biclean
    │        │   ├ corpus
    │        │   │   ├ mtdata_JW300.en.gz
    │        │   │   └ mtdata_JW300.ru.gz
    │        │   ├ corpus.ru.gz
    │        │   ├ corpus.en.gz
    │        ├ translated
    │        │   ├ mono.ru.gz
    │        │   └ mono.en.gz
    │        ├ augmented
    │        │   ├ corpus.ru.gz
    │        │   └ corpus.en.gz
    │        ├ alignment
    │        │   ├ corpus.aln.gz
    │        │   └ lex.s2t.pruned.gz
    │        ├ merged
    │        │   ├ corpus.ru.gz
    │        │   └ corpus.en.gz
    │        └ filtered
    │            ├ corpus.ru.gz
    │            └ corpus.en.gz
    ├ models
    │   └ ru-en
    │       └ test
    │          ├ backward
    │          ├ teacher-base0
    │          ├ teacher-base1
    │          ├ teacher-finetuned0
    │          ├ teacher-finetuned1
    │          ├ student
    │          ├ student-finetuned
    │          ├ speed
    │          ├ evaluation
    │          │  ├ backward
    │          │  ├ teacher-base0
    │          │  ├ teacher-base1
    │          │  ├ teacher-finetuned0
    │          │  ├ teacher-finetuned1
    │          │  ├ teacher-ensemble
    │          │  ├ student
    │          │  ├ student-finetuned
    │          │  └ speed
    │          └ exported
    │
    ├ experiments
    │   └ ru-en
    │      └ test
    │         └ config.sh
    ├ logs
    │   └ ru-en
    │      └ test
    │         └ clean_corpus.log
