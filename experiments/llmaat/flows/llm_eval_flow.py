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
    IncludeFile
)
from metaflow.cards import Markdown


# pylint: disable=import-error

GCS_PROJECT_NAME = "moz-fx-dev-releng"
GCS_BUCKET_NAME = "releng-translations-dev"
# Model blob to be uploaded to GCS
DATA_STORAGE_PATH = "data/llm-evals/wmt24pp/%s/en-%s/%s/translations.txt"
WANDB_PROJECT = "llm-evals"


@project(name="llmaat")
class LlmEvalFlow(FlowSpec):
    """
    LLM translation evaluation flow

    How to run:
        Create a virtual env e.g. conda create -n outerbounds python=3.11
        pip install -r requirements.outerbounds.txt
        export METAFLOW_CONDA_DEPENDENCY_RESOLVER=mamba
        export HUGGING_FACE_HUB_TOKEN=
        export WANDB_PROJECT=llmaat
        export WANDB_API_KEY=
        CONDA_OVERRIDE_GLIBC=2.17 CONDA_CHANNELS=conda-forge CONDA_PKGS_DIRS=.conda python llm_eval_flow.py \
            --environment=pypi run --offline False --experiment greedy

        to run locally add METAFLOW_PROFILE=local
        also remove @nvidia and @kubernetes
    """

    config = Config("config", default="./config.json")

    offline_wandb = Parameter(
        "offline",
        help="Do not connect to W&B",
        type=bool,
        default=False,
    )

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

    @step
    def start(self):
        self.langs = self.config.langs
        self.next(self.load_data, foreach='langs')

    @pypi(python="3.11.9", packages={"datasets": "3.4.1"})
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @kubernetes
    @step
    def load_data(self):
        from evals import load_data

        self.lang = self.input
        print(f"Downloading dataset for {self.lang}")
        self.data = load_data(self.lang)
        self.next(self.load_model)

    @pypi(python="3.11.9", packages={"huggingface-hub": "0.29.3"})
    @huggingface_hub
    @step
    def load_model(self):
        from llm_runner import Runner

        runner = Runner(self.model_name)
        self.llm = current.huggingface_hub.snapshot_download(
            repo_id=runner.get_repo(self.lang),
        )
        self.next(self.decode)

    @conda(
        python="3.11.0",
        packages={
            "pytorch::pytorch-cuda": "12.4",
            "pytorch::pytorch": "2.4.0",
            "conda-forge::transformers": "4.49.0",
            "conda-forge::tqdm": "4.67.1",
            "conda-forge::toolz": "1.0.0",
            "conda-forge::accelerate": "1.5.2",
            "conda-forge::sentencepiece": "0.2.0",
        },
    )
    @gpu_profile(interval=1)
    @model(load=["llm"])
    @nvidia(gpu=1, gpu_type="H100")
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @step
    def decode(self):
        import torch
        from datetime import datetime
        from llm_runner import Runner

        print(f"Gpu available: {torch.cuda.is_available()}")

        model_path = current.model.loaded["llm"]
        source_lines = self.data[0]
        print("Creating model")
        runner = Runner(self.model_name)
        runner.create(model_path)

        print("Decoding")
        start = datetime.utcnow()
        self.translations = runner.translate(
            source_lines, from_lang="en", to_lang=self.lang,
            batch_size=self.config.batch_size,
            params=self.config.decoding
        )
        print("Finished decoding")
        finish = datetime.utcnow()
        self.time_sec = (finish - start).seconds
        self.ex_num = len(source_lines)
        self.char_num = sum(len(line) for line in source_lines)
        print(f'Time: {self.time_sec} seconds')
        self.next(self.upload_to_gcs)

    @pypi(python="3.11.9", packages={"mozmlops": "0.1.4"})
    @kubernetes
    @step
    def upload_to_gcs(self):
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient

        print("Uploading data to gcs")
        # init client
        storage_client = CloudStorageAPIClient(
            project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        )
        text_bytes = ('\n'.join(self.translations)).encode()
        storage_client.store(
            data=text_bytes, storage_path=DATA_STORAGE_PATH % (self.model_name, self.lang, self.experiment)
        )
        self.next(self.eval)

    @card
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }
    )
    @conda(
        python="3.11.9",
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    @gpu_profile(interval=1)
    @nvidia(gpu=1, gpu_type="H100")
    @step
    def eval(self):
        import os

        # no conda distribution
        os.system("pip3 install unbabel-comet==2.2.2")
        from evals import eval

        self.comet_score = eval(self.data[0], self.translations, self.data[1])
        print(f"COMET score: {self.comet_score}")
        self.next(self.join)

    @pypi(
        python="3.11.9",
        packages={"mozmlops": "0.1.4"},
    )
    @kubernetes
    @card
    @step
    def join(self, inputs):
        import wandb

        # self.merge_artifacts(inputs, include=["comet_score"])
        self.scores = [input.comet_score for input in inputs]

        if not self.offline_wandb:
            print("Reporting results to W&B")
            lang = self.input
            tracking_run = wandb.init(
                project=WANDB_PROJECT, name=f"{self.model_name}_{self.experiment}",
                config=dict(self.config)
            )
            wandb_url = tracking_run.get_url()
            current.card.append(Markdown("# Weights & Biases"))
            current.card.append(Markdown(f"Your training run is tracked [here]({wandb_url})."))
            for input in inputs:
                title = f"wmt24++ en-{lang}"
                lang = input.lang
                score = input.comet_score
                speed_ls = input.ex_num / input.time_sec
                speed_cs = input.char_num / input.time_sec

                tracking_run.log(
                    {
                        title: wandb.plot.bar(
                            wandb.Table(
                                columns=["Metric", "Value"],
                                data=[
                                    ["COMET22", score],
                                    ["lines/sec", speed_ls],
                                    ["char/sec", speed_cs],
                                ],
                            ),
                            "Metric",
                            "Value",
                            title=title,
                        )
                    }
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
    LlmEvalFlow()
