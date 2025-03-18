import os

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    step,
    environment,
    nvidia,
    conda,
    gpu_profile,
    pypi,
)
from metaflow.cards import Markdown


# pylint: disable=import-error


class LlmEvalFlow(FlowSpec):
    """
    LLM translation evaluation flow

    Run command:
        export HUGGING_FACE_HUB_TOKEN=
        export WANDB_PROJECT=
        export WANDB_API_KEY=
        CONDA_OVERRIDE_GLIBC=2.17 CONDA_CHANNELS=conda-forge CONDA_PKGS_DIRS=.conda python test_gpu.py --environment=conda run
    """

    # You can import the contents of files from your file system to use in flows.
    # This is meant for small files—in this example, a bit of config.
    example_config = IncludeFile("example_config", default="./example_config.json")

    offline_wandb = Parameter(
        "offline",
        help="Do not connect to W&B servers when training",
        type=bool,
        default=True,
    )

    @pypi(python="3.11.0", packages={"datasets": "3.4.1"})
    @environment(
        vars={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        }
    )
    @step
    def data(self):
        from evals import load_data

        self.data = load_data("ru")
        self.next(self.decode)

    # ⬇️ Alternatively, enable this to test @conda on Mac OS X (no CUDA)
    # @conda_base(python='3.11.0', packages={'pytorch::pytorch': '2.4.0'})
    # ⬇️ Enable this to test @conda with GPU or on workstations
    @conda(
        python="3.11.0",
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    # @pypi(python="3.11.0", packages={"torch": "2.4.0"})
    @gpu_profile(interval=1)
    @nvidia(gpu=1, gpu_type="H100")
    @step
    def decode(self):
        from llm_runner import Runner

        runner = Runner("x-alma", target_lang="ru")
        self.translations = runner.translate(
            self.data[0], from_lang="en", to_lang="ru", batch_size=32
        )
        self.next(self.end)

    @card
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
        }
    )
    @conda(
        python="3.11.0",
        packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
    )
    @gpu_profile(interval=1)
    @nvidia(gpu=1, gpu_type="H100")
    @step
    def end(self):
        from evals import eval
        import wandb

        self.metrics = eval(self.data[0], self.translations, self.data[1])
        if not self.offline_wandb:
            tracking_run = wandb.init(project=os.getenv("WANDB_PROJECT"))
            wandb_url = tracking_run.get_url()
            current.card.append(Markdown("# Weights & Biases"))
            current.card.append(Markdown(f"Your training run is tracked [here]({wandb_url})."))
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
