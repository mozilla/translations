import os

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    step,
    environment,
    nvct,
    conda,
    gpu_profile,
    pypi,
    kubernetes,
    huggingface_hub,
    model,
    project,
    Config,
)
from metaflow.cards import Markdown

# pylint: disable=import-error

GCS_PROJECT_NAME = "moz-fx-dev-releng"
GCS_BUCKET_NAME = "releng-translations-dev"
# Model blob to be uploaded to GCS
DATA_STORAGE_PATH = "data/llm-evals/wmt24pp/%s/en-%s/%s/translations.txt"
WANDB_PROJECT = "llm-evals"
PYTHON_VERSION = "3.10.8"


@project(name="llmaat")
class LlmEvalFlow(FlowSpec):
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

        To run from a laptop add:
        CONDA_OVERRIDE_GLIBC=2.17 CONDA_CHANNELS=conda-forge CONDA_PKGS_DIRS=.conda

        python llm_eval_flow.py \
            --environment=pypi --config config ./configs/config.vllm.greedy.json run --experiment greedy --model gemma-3-4b-vllm  --max-workers 4

        to run locally add METAFLOW_PROFILE=local
        also remove @nvct and @kubernetes
    """

    config = Config("config", default="./configs/config.greedy.json")

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
        self.next(self.load_data, foreach="langs")

    @pypi(python=PYTHON_VERSION, packages={"datasets": "3.4.1"})
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

    @pypi(python=PYTHON_VERSION, packages={"huggingface-hub": "0.34.3"})
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @kubernetes(compute_pool="obp-c2-standard-4", disk=145000)
    @huggingface_hub
    @step
    def load_model(self):
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
        self.next(self.decode)

    @pypi(
        python="3.12",
        packages={
            # vllm also installs pytorch and transformers
            "vllm": "0.10.0",
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
        # it doesn't install from the decorator
        # os.system(
        #     "pip3 install openai-harmony"
        # )
        # install manually for gpt-oss
        # os.system(
        #     "pip3 install --pre vllm==0.10.1+gptoss --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128")

        import torch
        from datetime import datetime
        from llm_runner import Runner

        print(f"Gpu available: {torch.cuda.is_available()}")

        model_path = current.model.loaded["llm"]
        source_lines = self.data[0]
        print("Creating model")
        runner = Runner(self.model_name)
        runner.create(model_path, params=dict(self.config))

        print("Decoding")
        start = datetime.utcnow()
        translations = runner.translate(
            source_lines,
            from_lang="en_US",
            to_lang=self.lang,
            params=dict(self.config),
        )
        print("Finished decoding")
        finish = datetime.utcnow()
        self.time_sec = (finish - start).seconds
        self.ex_num = len(source_lines)
        self.char_num = sum(len(line) for line in source_lines)
        print(f"Time: {self.time_sec} seconds")

        # Single output decoding
        # self.translations = translations
        # self.next(self.upload_to_gcs)
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
            self.data[0], self.candidates, model_size="large", batch_size=256
        )
        print("Finished")
        finish = datetime.utcnow()
        time_sec = (finish - start).seconds
        print(f"Time: {time_sec} sec")
        self.next(self.upload_to_gcs)

    @pypi(python=PYTHON_VERSION, packages={"mozmlops": "0.1.4"})
    @kubernetes
    @step
    def upload_to_gcs(self):
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient

        print("Uploading data to gcs")
        # init client
        storage_client = CloudStorageAPIClient(
            project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        )
        text_bytes = ("\n".join(self.translations)).encode()
        storage_client.store(
            data=text_bytes,
            storage_path=DATA_STORAGE_PATH % (self.model_name, self.lang, self.experiment),
        )
        self.next(self.eval_comet)

    @card
    @conda(
        python=PYTHON_VERSION,
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    @gpu_profile(interval=1)
    @nvct(gpu=1, gpu_type="H100")
    @step
    def eval_comet(self):
        import os

        # no conda distribution
        os.system("pip3 install unbabel-comet==2.2.2")
        from evals import eval_comet

        self.comet_score = eval_comet(self.data[0], self.translations, self.data[1])
        print(f"COMET score: {self.comet_score}")
        self.next(self.eval_metricx)

    @card
    @conda(
        python=PYTHON_VERSION,
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    @gpu_profile(interval=1)
    @nvct(gpu=1, gpu_type="H100")
    @step
    def eval_metricx(self):
        import os

        # no conda distribution
        os.system(
            "pip3 install transformers==4.50.1 sentencepiece==0.2.0 datasets==3.4.1 accelerate==0.26.0"
        )
        from evals import eval_metricx

        self.metricx_scores = eval_metricx(
            self.data[0], self.translations, self.data[1], model_size="xxl", batch_size=32
        )
        print(f"MetricX scores: {self.metricx_scores}")
        self.next(self.join)

    @pypi(
        python=PYTHON_VERSION,
        packages={"mozmlops": "0.1.4"},
    )
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        }
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
            tracking_run = wandb.init(
                project=WANDB_PROJECT,
                name=f"{self.model_name}_{self.experiment}",
                config=dict(self.config),
            )
            wandb_url = tracking_run.get_url()
            current.card.append(Markdown("# Weights & Biases"))
            current.card.append(Markdown(f"Your training run is tracked [here]({wandb_url})."))
            lang_metrics = {}
            for input in inputs:
                lang = input.lang
                # make compatible with earlier runs
                if lang == "ru_RU":
                    lang = "ru"
                comet_score = input.comet_score
                metricx_scores = input.metricx_scores
                speed_ls = input.ex_num / input.time_sec
                speed_cs = input.char_num / input.time_sec
                metrics = {
                    "COMET22": comet_score,
                    "lines/sec": speed_ls,
                    "char/sec": speed_cs,
                    "batch_size": self.config.batch_size,
                }
                metrics.update(metricx_scores)
                lang_metrics[lang] = metrics

            metric_langs = {}
            for lang, metrics in lang_metrics.items():
                for metric, val in metrics.items():
                    if metric not in metric_langs:
                        metric_langs[metric] = []
                    metric_langs[metric].append([f"en-{lang}", val])

            for metric_name, metric_values in metric_langs.items():
                title = f"wmt24++ {metric_name}"
                tracking_run.log(
                    {
                        title: wandb.plot.bar(
                            wandb.Table(
                                columns=["Lang", "Value"],
                                data=metric_values,
                            ),
                            "Lang",
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
