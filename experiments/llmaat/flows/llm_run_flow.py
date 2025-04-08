import os

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    step,
    environment,
    nvidia,
    conda,
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
        export WANDB_PROJECT=llmaat
        export WANDB_API_KEY=
        CONDA_OVERRIDE_GLIBC=2.17 CONDA_CHANNELS=conda-forge CONDA_PKGS_DIRS=.conda python llm_run_flow.py \
            --environment=pypi --config config ./config.beam-sample.json run  --experiment greedy --max-workers 32

        to run locally add METAFLOW_PROFILE=local
        also remove @nvidia and @kubernetes

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

    @pypi(python=PYTHON_VERSION, packages={"huggingface-hub": "0.29.3"})
    @resources(disk=50000)
    @kubernetes
    @huggingface_hub
    @step
    def start(self):
        from llm_runner import Runner

        runner = Runner(self.model_name)
        self.llm = current.huggingface_hub.snapshot_download(
            repo_id=runner.get_repo(self.lang), max_workers=100
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

        data_path = (
            f"gs://releng-translations-dev/data/mono-llm/diverse_sample.{self.size}M.en.zst"
        )
        storage_client = CloudStorageAPIClient(
            project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        )
        storage_client.fetch(
            remote_path=data_path,
            local_path="sample.zst",
        )
        with zstandard.open("sample.zst", "r") as f:
            lines = [line for line in f]

        self.parts = toolz.partition_all(self.part_size, lines)
        self.next(self.decode, foreach="parts")

    @conda(
        python="3.11.0",
        packages={
            "pytorch::pytorch-cuda": "12.4",
            "pytorch::pytorch": "2.4.0",
            "conda-forge::tqdm": "4.67.1",
            "conda-forge::toolz": "1.0.0",
            "conda-forge::accelerate": "1.5.2",
            "conda-forge::sentencepiece": "0.2.0",
        },
    )
    @card
    @gpu_profile(interval=1)
    @model(load=["llm"])
    # change to gpu=4 for Llama 70b
    @nvidia(gpu=1, gpu_type="H100")
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @step
    def decode(self):
        import os
        import torch
        from datetime import datetime
        from llm_runner import Runner

        # latest versions are not on conda
        os.system("pip3 install transformers==4.50.3")
        print(f"Gpu available: {torch.cuda.is_available()}")

        model_path = current.model.loaded["llm"]
        source_lines = self.input
        print("Creating model")
        runner = Runner(self.model_name)
        runner.create(model_path, params=dict(self.config))

        print(f"Decoding {len(source_lines)} lines")
        start = datetime.utcnow()
        translations = runner.translate(
            source_lines,
            from_lang="en",
            to_lang=self.lang,
            params=dict(self.config),
        )
        print("Finished decoding")
        finish = datetime.utcnow()
        self.time_sec = (finish - start).seconds
        self.ex_num = len(source_lines)
        self.char_num = sum(len(line) for line in source_lines)
        print(f"Time: {self.time_sec} seconds")

        self.translations = translations
        self.next(self.upload_to_gcs)
        # QE reranking (num_return_sequences > 1)
        # self.candidates = translations
        # self.next(self.pick_best)

    # @card
    # @conda(
    #     python=PYTHON_VERSION,
    #     packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    # )
    # @gpu_profile(interval=1)
    # @nvidia(gpu=1, gpu_type="H100")
    # @step
    # def pick_best(self):
    #     import os
    #     from datetime import datetime
    #
    #     # no conda distribution
    #     os.system(
    #         "pip3 install transformers==4.50.1 sentencepiece==0.2.0 datasets==3.4.1 accelerate==0.26.0"
    #     )
    #     from evals import select_best
    #
    #     start = datetime.utcnow()
    #     print("Start selecting best candidates with MetricX QE")
    #     self.translations = select_best(
    #         self.data[0], self.candidates, model_size="xl", batch_size=8
    #     )
    #     print("Finished")
    #     finish = datetime.utcnow()
    #     time_sec = (finish - start).seconds
    #     print(f"Time: {time_sec} sec")
    #     self.next(self.upload_to_gcs)

    @pypi(python=PYTHON_VERSION, packages={"mozmlops": "0.1.4"})
    @kubernetes
    @step
    def join(self, inputs):
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient

        all_translations = [tr for input in inputs for tr in input.translations]

        print("Uploading data to gcs")
        # init client
        storage_client = CloudStorageAPIClient(
            project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        )
        text_bytes = ("\n".join(all_translations)).encode()
        file_name = f"diverse_sample.{self.size}M.{self.lang}"
        storage_client.store(
            data=text_bytes,
            storage_path=DATA_STORAGE_PATH
            % (self.lang, self.model_name, self.experiment, file_name),
        )
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
