from taskgraph.transforms.base import TransformSequence
from urllib.parse import urljoin
import os

MODEL_TRAINING_ARTIFACT_NAMES = (
    "devset.out",
    "model.npz",
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-bleu-detok.npz.decoder.yml",
    "model.npz.best-ce-mean-words.npz",
    "model.npz.best-ce-mean-words.npz.decoder.yml",
    "model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz.decoder.yml",
    "model.npz.decoder.yml",
    "model.npz.optimizer.npz",
    "model.npz.progress.yml",
    "model.npz.yml",
    "train.log",
    "valid.log",
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
    ret = []
    for url in urls:
        artifact_mounts = []
        for artifact_name in artifact_names:
            artifact_mounts.append(get_artifact_mount(url, directory, artifact_name))
        ret.append(artifact_mounts)
    return ret


transforms = TransformSequence()


@transforms.add
def training_continuation(config, jobs):
    pretrained_models = config.params["training_config"]["experiment"].get("pretrained-models", {})
    if not any(pretrained_models):
        for job in jobs:
            yield job
    else:
        model_config = pretrained_models.get("teacher-base")
        teacher_ensemble = config.params["training_config"]["experiment"]["teacher-ensemble"]
        if len(model_config["urls"]) != teacher_ensemble:
            raise Exception(
                f"The experiment's 'teacher-ensemble' ({teacher_ensemble}) "
                f"does not match the number of provided model 'urls' ({len(model_config['urls'])}) "
                f"for the pretrained 'teacher-base' ensemble."
            )
        model_training_artifact_mounts = get_artifact_mounts(
            model_config["urls"], "./artifacts", MODEL_TRAINING_ARTIFACT_NAMES
        )
        for job, job_mounts in zip(jobs, model_training_artifact_mounts):
            job["task"]["payload"]["mounts"].extend(job_mounts)
            yield job
