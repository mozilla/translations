from typing import Any
import requests
from taskgraph.transforms.base import TransformSequence
from translations_taskgraph.util.mocked_downloads import get_mocked_downloads_file_path
from urllib.parse import urljoin
import os

CONTINUE_TRAINING_ARTIFACTS = [
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
    # + vocab*.spm artifacts which are determined dynamically
]

INITIALIZE_MODEL_ARTIFACTS = [
    "final.model.npz.best-chrf.npz",
    # + vocab*.spm artifacts which are determined dynamically
]


def location_exists(location: str) -> bool:
    if get_mocked_downloads_file_path(location):
        return True

    return requests.head(location, allow_redirects=True).ok


def get_artifact_url(url: str, artifact_name: str) -> str:
    normalized_url = f"{url}/" if not url.endswith("/") else url
    return urljoin(normalized_url, artifact_name)


def get_artifact_mounts(urls: list[str], directory: str, artifact_names: list[str]):
    url_mounts = []
    for url in urls:
        artifact_mounts = []
        for artifact_name in artifact_names:
            artifact_mounts.append(
                {
                    "content": {"url": get_artifact_url(url, artifact_name)},
                    "file": os.path.join(directory, artifact_name),
                }
            )
        url_mounts.append(artifact_mounts)
    return url_mounts


def get_models_mounts(pretrained_models: dict[str, Any], src: str, trg: str):
    mounts = {}
    for pretrained_model in pretrained_models:
        model_urls: list[str] = pretrained_models[pretrained_model].get("urls")
        if not model_urls:
            # Only teachers can be ensembles, all other models have a single URL.
            model_urls = [pretrained_models[pretrained_model]["url"]]

        model_artifacts = (
            INITIALIZE_MODEL_ARTIFACTS
            if pretrained_models[pretrained_model]["mode"] == "init"
            else CONTINUE_TRAINING_ARTIFACTS
        )
        joint_vocab_url = get_artifact_url(model_urls[0], "vocab.spm")
        src_vocab_url = get_artifact_url(model_urls[0], f"vocab.{src}.spm")
        if location_exists(joint_vocab_url):
            print("Using a joint vocab mount")
            vocab_artifacts = ["vocab.spm"]
        elif location_exists(src_vocab_url):
            print("Using separate vocabs mounts")
            vocab_artifacts = [f"vocab.{src}.spm", f"vocab.{trg}.spm"]
        else:
            raise ValueError("Could not find either a shared or split vocab.")
        model_artifacts += vocab_artifacts

        mount = get_artifact_mounts(model_urls, "./artifacts", model_artifacts)
        mounts[pretrained_model] = mount
    return mounts


transforms = TransformSequence()


@transforms.add
def add_pretrained_model_mounts(config, jobs):
    pretrained_models = config.params["training_config"].get("continuation", {}).get("models", {})
    src = config.params["training_config"]["experiment"]["src"]
    trg = config.params["training_config"]["experiment"]["trg"]
    pretrained_models_training_artifact_mounts = get_models_mounts(pretrained_models, src, trg)
    kind_models = [
        pretrained_model
        for pretrained_model in pretrained_models_training_artifact_mounts
        if pretrained_model in config.kind
    ]

    for job in jobs:
        if kind_models:
            pretrained_model = kind_models[0]
            pretrained_model_training_artifact_mounts = pretrained_models_training_artifact_mounts[
                pretrained_model
            ][0]

            if pretrained_model_training_artifact_mounts:
                mounts = job["worker"].get("mounts", [])
                mounts.extend(pretrained_model_training_artifact_mounts)
                job["worker"]["mounts"] = mounts
                job["dependencies"].pop("build-vocab")
                job["fetches"].pop("build-vocab")

                if pretrained_models[pretrained_model]["mode"] == "use":
                    # In use mode, no upstream dependencies of the training job are needed - the
                    # task simply republishes the pretrained artifacts.
                    job["dependencies"] = {}
                    job["fetches"] = {}
                    # We also need to adjust the caching parameters. The only thing that should influence
                    # the cache digest are the pretrained model parameters.
                    job["attributes"]["cache"]["resources"] = []
                    job["attributes"]["cache"]["from-parameters"] = {
                        p: v
                        for p, v in job["attributes"]["cache"]["from-parameters"].items()
                        if p.startswith("pretrained")
                    }

        yield job


evaluate_stage = TransformSequence()


@evaluate_stage.add
def skip_for_pretrained_models(config, jobs):
    # Find the types of pretrained models that are being used. This makes
    # it easier to filter them out in the loop below.
    pretrained_models = [
        pretrained.split("-")[-1].replace("backwards", "backward")
        for pretrained in config.params["training_config"]
        .get("continuation", {})
        .get("models", {})
        .keys()
    ]

    for job in jobs:
        if any([pretrained in job["attributes"]["stage"] for pretrained in pretrained_models]):
            continue

        yield job
