#!make

.ONESHELL:
SHELL=/bin/bash

### change these settings
SHARED_ROOT=/data/rw/group-maml
CUDA_DIR=/usr/loca/cuda
GPUS=8
WORKSPACE=12000
CLUSTER_CORES=16
CONFIG=config.prod.yml
###

CONDA_ACTIVATE=source $(SHARED_ROOT)/mambaforge/etc/profile.d/conda.sh ; conda activate ; conda activate

install-conda:
	wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$$(uname)-$$(uname -m).sh
	bash Mambaforge-$$(uname)-$$(uname -m).sh -p $(SHARED_ROOT)/mambaforge

install-snakemake:
	git submodule update --init --recursive
	$(CONDA_ACTIVATE) base
	mamba create -c conda-forge -c bioconda -n snakemake snakemake

activate:
	$(CONDA_ACTIVATE) snakemake

install-singularity: activate
	conda install singularity

build-container: activate
	sudo singularity build Singularity.sif Singularity.def

pull-container: activate
	singularity pull Singularity.sif library://evgenypavlov/default/bergamot:0.1

install-git-modules:
	bash pipeline/setup/install-git-modules.sh

config:
	cp configs/$(CONFIG) config.yml
	sed -i "s/<cuda-dir>/\"$(CUDA_DIR)\"/" config.yml
	sed -i "s/<shared-root>/\"$(SHARED_ROOT)\"/" config.yml
	sed -i "s/<gpus>/$(GPUS)/" config.yml
	sed -i "s/<workspace>/$(WORKSPACE)/" config.yml

dry-run: activate
	snakemake \
	  --use-conda \
	  --cores all \
	  -n

all: install-conda install-snakemake install-singularity pull-container install-git-modules config dry-run

run-local: activate
	snakemake \
	  --use-conda \
	  --use-singularity \
	  --reason \
	  --cores all \
	  --resources gpu=$(GPUS) \
	  --singularity-args="--bind $(SHARED_ROOT),$(CUDA_DIR) --nv"

run-slurm: activate
	chmod +x profiles/slurm/*
	snakemake \
	  --use-conda \
	  --use-singularity \
	  --reason \
	  --cores $(CLUSTER_CORES) \
	  --profile=profiles/slurm \
	  --singularity-args="--bind $(SHARED_ROOT),$(CUDA_DIR) --nv"

report: activate
	REPORTS=$$(python -c "from config import reports_dir; print(reports_dir)"); \
	mkdir -p $$REPORTS && \
	snakemake --report $$REPORTS/report.html

dag:
	snakemake --dag | dot -Tpdf > DAG.pdf

lint:
	snakemake --lint

install-monitor:
	conda create --name panoptes
	conda install -c panoptes-organization panoptes-ui

run-monitor:
	$(CONDA_ACTIVATE) panoptes
	panoptes

run-with-monitor:
	snakemake \
	  --use-conda \
	  --cores all \
	  --wms-monitor http://127.0.0.1:5000

containerize: activate
	pip install spython
	snakemake --containerize > Dockerfile
	spython recipe Dockerfile &> Singularity.def

run-file-server: activate
	python -m  http.server --directory $(SHARED_ROOT)/bergamot/reports 8000

tensorboard: activate
	MODELS=$$(python -c "from config import models_dir; print(models_dir)"); \
	ls -d $$MODELS/*/*/* > tb-monitored-jobs; \
	tensorboard --logdir=$$MODELS --host=0.0.0.0 &; \
	python utils/tb_log_parser.py --prefix=

install-snakepit-scheduler:
	mkdir -p $(SHARED_ROOT)/snakepit
	cd $(SHARED_ROOT)/snakepit

	curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
	sudo apt install nodejs

	if [ ! -e snakepit-client ]; then
	  git clone https://github.com/mozilla/snakepit-client.git
	fi
	cd snakepit-client
	npm install
	sudo npm link

	echo "http://10.2.224.243" > /root/.pitconnect.txt

	pit status


run-snakepit: activate
	chmod +x profiles/snakepit/*
	snakemake \
	  --use-conda \
	  --cores all \
	  --profile=profiles/snakepit
