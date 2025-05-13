from taskgraph.transforms.base import TransformSequence
from urllib.parse import urljoin
import os

CONTINUE_TRAINING_ARTIFACTS = (
    "devset.out",
    "model.npz",
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-bleu-detok.npz.decoder.yml",
    "model.npz.best-ce-mean-words.npz",
    "model.npz.best-ce-mean-words.npz.decoder.yml",
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
    "final.model.npz.best-chrf.npz.decoder.yml",
    "model.npz.best-chrf.npz.decoder.yml",
    "model.npz.decoder.yml",
    "model.npz.optimizer.npz",
    "model.npz.progress.yml",
    "model.npz.yml",
    "train.log",
    "valid.log",
    "vocab.spm",
)

INITIALIZE_MODEL_ARTIFACTS = (
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-ce-mean-words.npz",
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
    "vocab.spm",
)


def get_artifact_mount(url, directory, artifact_name):
    normalized_url = f"{url}/" if not url.endswith("/") else url
    artifact_url = urljoin(normalized_url, artifact_name)
    return {
        "content": {
            "url": artifact_url,
        },
        "file": os.path.join(directory, artifact_name),
    }


def get_artifact_mounts(urls, directory, artifact_names):
    for url in urls:
        artifact_mounts = []
        for artifact_name in artifact_names:
            artifact_mounts.append(get_artifact_mount(url, directory, artifact_name))
        yield artifact_mounts


transforms = TransformSequence()


@transforms.add
def add_pretrained_model_mounts(config, jobs):
    pretrained_models = config.params["training_config"]["experiment"].get("pretrained-models", {})
    for job in jobs:
        pretrained_models_training_artifact_mounts = {
            pretrained_model: get_artifact_mounts(
                pretrained_models[pretrained_model]["urls"],
                "./artifacts",
                INITIALIZE_MODEL_ARTIFACTS
                if pretrained_models[pretrained_model]["mode"] == "init"
                else CONTINUE_TRAINING_ARTIFACTS,
            )
            for pretrained_model in pretrained_models
        }
        pretrained_model_training_artifact_mounts = next(
            pretrained_models_training_artifact_mounts.get(config.kind, iter((None,)))
        )
        if pretrained_model_training_artifact_mounts:
            mounts = job["worker"].get("mounts", [])
            mounts.extend(pretrained_model_training_artifact_mounts)
            job["worker"]["mounts"] = mounts
            job["dependencies"].pop("train-vocab")
            job["fetches"].pop("train-vocab")

            #TODO: fix, currenlty removes evaluation steps for student
            # if pretrained_models[config.kind]["mode"] == "use":
            #     # In use mode, no upstream dependencies of the training job are needed - the
            #     # task simply republishes the pretrained artifacts.
            #     job["dependencies"] = {}
            #     job["fetches"] = {}
            #
            #     # We also need to adjust the caching parameters. The only thing that should influence
            #     # the cache digest are the pretrained model parameters.
            #     job["attributes"]["cache"]["resources"] = []
            #     job["attributes"]["cache"]["from-parameters"] = {
            #         p: v
            #         for p, v in job["attributes"]["cache"]["from-parameters"].items()
            #         if p.startswith("pretrained")
            #     }

        yield job


evaluate_stage = TransformSequence()


@evaluate_stage.add
def skip_for_pretrained_models(config, jobs):
    # Find the types of pretrained models that are being used as is without finetuning and skip evaluation.
    # This make it easier to filter them out in the loop below.
    pretrained_models = [
        name.split("-")[-1].replace("backwards", "backward")
        for name, obj in config.params["training_config"]["experiment"]
        .get("pretrained-models", {})
        .items()
        if obj["mode"] == "use"
    ]

    for job in jobs:
        if any([pretrained in job["attributes"]["stage"] for pretrained in pretrained_models]):
            continue

        yield job
