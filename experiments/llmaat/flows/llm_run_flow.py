import os

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    step,
    conda,
    environment,
    nvct,
    gpu_profile,
    pypi,
    kubernetes,
    huggingface_hub,
    model,
    project,
    Config,
    resources,
)

# pylint: disable=import-error

GCS_PROJECT_NAME = "moz-fx-dev-releng"
GCS_BUCKET_NAME = "releng-translations-dev"
# Model blob to be uploaded to GCS
DATA_STORAGE_PATH = "data/llm/en-%s/%s/%s/%s"
WANDB_PROJECT = "llm-evals"
PYTHON_VERSION = "3.10.8"


@project(name="llmaat")
class LlmRunFlow(FlowSpec):
    """
    LLM translation evaluation flow

    How to run:
        Create a virtual env e.g. conda create -n outerbounds python=3.11
        pip install -r requirements.outerbounds.txt
        brew install micromamba
        export METAFLOW_CONDA_DEPENDENCY_RESOLVER=/usr/local/opt/micromamba/bin/mamba
        export HUGGING_FACE_HUB_TOKEN=
        To run from a laptop add:
        CONDA_OVERRIDE_GLIBC=2.17 CONDA_CHANNELS=conda-forge CONDA_PKGS_DIRS=.conda

       python llm_run_flow.py \
            --environment=pypi --config config ./configs/config.vllm.greedy.json run --experiment finetune10M \
            --model gemma-3-27b-vllm --data_size 10 --lang ru_RU --part_size 500000 --max-workers 4


        to run locally add METAFLOW_PROFILE=local
        also remove @nvct and @kubernetes

        --max-workers controls parallelizm
    """

    config = Config("config", default="./configs/config.greedy.json")

    model_name = Parameter(
        "model",
        help="Model ID",
        type=str,
        default="x-alma-13b",
    )

    experiment = Parameter(
        "experiment",
        help="Experiment id",
        type=str,
    )

    size = Parameter(
        "data_size",
        help="Size of the dataset: 1,10 or 50 million",
        type=int,
    )

    lang = Parameter(
        "lang",
        help="Translate to this language",
        type=str,
    )

    part_size = Parameter("part_size", help="Size of each data partition", type=int, default=50000)

    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @pypi(python=PYTHON_VERSION, packages={"huggingface-hub": "0.29.3"})
    @kubernetes(compute_pool="obp-c2-standard-4", disk=145000)
    @kubernetes
    @huggingface_hub
    @step
    def start(self):
        from llm_runner import Runner

        runner = Runner(self.model_name)
        self.llm = current.huggingface_hub.snapshot_download(
            repo_id=runner.get_repo(self.lang),
            allow_patterns=[
                "*.safetensors",
                "*.json",
                # exclude redundant weights from original/ for Llama models
                "original/tokenizer.*",
                "tokenizer.*",
            ],
            max_workers=100,
        )
        self.next(self.split)

    @pypi(
        python=PYTHON_VERSION,
        packages={"mozmlops": "0.1.4", "zstandard": "0.22.0", "toolz": "1.0.0"},
    )
    @kubernetes(disk=40000)
    @step
    def split(self):
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient
        import toolz
        import zstandard

        data_path = f"data/mono-llm/diverse_sample.{self.size}M.en.zst"
        storage_client = CloudStorageAPIClient(
            project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        )
        print("Downloading data")
        storage_client.fetch(
            remote_path=data_path,
            local_path="sample.zst",
        )
        print("Decompressing")
        with zstandard.open("sample.zst", "r") as f:
            lines = [line.strip() for line in f]

        self.parts = list(toolz.partition_all(self.part_size, lines))
        self.next(self.decode, foreach="parts")

    @pypi(
        python="3.12",
        packages={
            # vllm also installs pytorch and transformers
            "vllm": "0.8.3",
            "tqdm": "4.67.1",
            "toolz": "1.0.0",
        },
    )
    @card
    @gpu_profile(interval=1)
    @model(load=["llm"])
    @nvct(gpu=1, gpu_type="H100")
    # change to gpu=4 for Llama 70b and Qwen3 235B a22b fp8
    # @nvct(gpu=4, gpu_type="H100")
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
            # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    @step
    def decode(self):
        import torch
        from datetime import datetime
        from llm_runner import Runner

        print(f"Gpu available: {torch.cuda.is_available()}")

        model_path = current.model.loaded["llm"]
        source = self.input
        print("Creating model")
        runner = Runner(self.model_name)
        runner.create(model_path, params=dict(self.config))

        print(f"Decoding {len(source)} lines")
        start = datetime.utcnow()
        translations = runner.translate(
            source,
            from_lang="en_US",
            to_lang=self.lang,
            params=dict(self.config),
        )
        print("Finished decoding")
        finish = datetime.utcnow()
        self.time_sec = (finish - start).seconds
        self.ex_num = len(source)
        self.char_num = sum(len(line) for line in source)
        print(f"Time: {self.time_sec} seconds")

        # save the original input
        self.source = source
        # Single output decoding
        # self.translations = translations
        # self.next(self.join)
        # QE reranking (n > 1)
        self.candidates = translations
        self.next(self.pick_best)

    @card
    @conda(
        python=PYTHON_VERSION,
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    @gpu_profile(interval=1)
    @nvct(gpu=1, gpu_type="H100")
    @step
    def pick_best(self):
        import os
        from datetime import datetime

        # no conda distribution
        os.system(
            "pip3 install transformers==4.50.1 sentencepiece==0.2.0 datasets==3.4.1 accelerate==0.26.0"
        )
        from evals import select_best

        start = datetime.utcnow()
        print("Start selecting best candidates with MetricX QE")
        self.translations, self.scores = select_best(
            self.source, self.candidates, model_size="large", batch_size=128
        )
        print("Finished")
        finish = datetime.utcnow()
        time_sec = (finish - start).seconds
        print(f"Time: {time_sec} sec")
        self.next(self.join)

    @pypi(python=PYTHON_VERSION, packages={"mozmlops": "0.1.4", "zstandard": "0.22.0"})
    @kubernetes(disk=20000, memory=16000)
    @step
    def join(self, inputs):
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient
        import zstandard

        all_translations = [tr for input in inputs for tr in input.translations]
        all_source = [tr for input in inputs for tr in input.source]
        all_scores = [sc for input in inputs for sc in input.scores]

        print("Saving data")

        def save_data(all_lines, suffix):
            cctx = zstandard.ZstdCompressor()
            compressed_bytes = cctx.compress(("\n".join([str(l) for l in all_lines])).encode())
            file_name = f"diverse_sample.{self.size}M.{suffix}.zst"
            print(f"Uploading data to gcs {file_name}")
            # init client
            storage_client = CloudStorageAPIClient(
                project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
            )
            storage_client.store(
                data=compressed_bytes,
                storage_path=DATA_STORAGE_PATH
                % (self.lang, self.model_name, self.experiment, file_name),
            )

        save_data(all_source, "en")
        save_data(all_translations, self.lang)
        save_data(all_scores, "scores")

        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """

        print(
            """
            Flow complete.

            """
        )


if __name__ == "__main__":
    LlmRunFlow()
