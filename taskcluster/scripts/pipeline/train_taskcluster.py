#!/usr/bin/env python3

import logging
import os
import os.path
import requests
import subprocess
import sys

TRAINING_SCRIPT = os.path.join(os.path.dirname(__file__), "train-taskcluster.sh")
CONTINUATION_ARTIFACTS = {
    "config.opustrainer.yml",
    "config.opustrainer.yml.state",
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
    "opustrainer.log",
    "train.log",
    "valid.log",
    # + vocab*.spm artifacts which are determined dynamically
}

logger = logging.getLogger("train_taskcluster")
logger.setLevel(logging.INFO)

ARTIFACTS_URL = "{root_url}/api/queue/v1/task/{task_id}/runs/{run_id}/artifacts"
ARTIFACT_URL = "{root_url}/api/queue/v1/task/{task_id}/runs/{run_id}/artifacts/{artifact_name}"
# TODO: consolidate everything in train.py or at least do not rely on the argument names and the number of them in the Taskcluster part
# TODO: https://github.com/mozilla/translations/issues/607
# The argument number where pretrained model mode is expected.
# This is 1-indexed, not 0-indexed, so it should line up with the argument
# number this is fetched in in train-taskcluster.sh
PRETRAINED_MODEL_MODE_ARG_NUMBER = 14
# Nothing special about 17...just a number plucked out of thin air that
# should be distinct enough to retry on.
DOWNLOAD_ERROR_EXIT_CODE = 17


def main(args):
    script_args = list(args)
    src = args[2]
    trg = args[3]
    task_id = os.environ["TASK_ID"]
    run_id = int(os.environ["RUN_ID"])
    root_url = os.environ["TASKCLUSTER_ROOT_URL"]
    # Must line up with where model_dir is in `train-taskcluster.sh` while that script
    # still exists.
    model_dir = script_args[6]
    pretrained_model_mode = None
    if len(args) >= PRETRAINED_MODEL_MODE_ARG_NUMBER:
        pretrained_model_mode = script_args[PRETRAINED_MODEL_MODE_ARG_NUMBER - 1]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if run_id > 0:
        logger.info("run_id > 0, attempting to resume training from an earlier run...")
        prev_run_id = run_id - 1

        while prev_run_id >= 0:
            try:
                resp = requests.get(
                    ARTIFACTS_URL.format(root_url=root_url, task_id=task_id, run_id=prev_run_id)
                )
                resp.raise_for_status()
            except Exception:
                logger.exception("Caught exception, exiting with distinct code...")
                sys.exit(DOWNLOAD_ERROR_EXIT_CODE)

            run_artifacts = set([os.path.basename(a["name"]) for a in resp.json()["artifacts"]])

            resumable = True

            if run_artifacts.issuperset(
                CONTINUATION_ARTIFACTS.union({f"vocab.{src}.spm", f"vocab.{trg}.spm"})
            ) or run_artifacts.issuperset(CONTINUATION_ARTIFACTS.union({"vocab.spm"})):
                logger.info(
                    f"Run {prev_run_id} appears to have the artifacts we need! Downloading them..."
                )
            else:
                logger.info(f"Run {prev_run_id} is missing some necessary artifacts...")
                resumable = False

            if resumable:
                for artifact in resp.json()["artifacts"]:
                    # Skip Taskcluster logs - we only care about artifacts that the training tools create.
                    if artifact["name"].startswith("public/log"):
                        continue
                    out_name = os.path.basename(artifact["name"])
                    logger.info(f"Fetching {artifact['name']}...")

                    r = requests.get(
                        ARTIFACT_URL.format(
                            root_url=root_url,
                            task_id=task_id,
                            run_id=prev_run_id,
                            artifact_name=artifact["name"],
                        ),
                        stream=True,
                    )
                    if 400 <= r.status_code <= 500:
                        logger.exception(
                            f"Got 4xx error for {artifact['name']}, run {run_id} is not resumable..."
                        )
                        resumable = False
                        break
                    elif r.status_code >= 500:
                        logger.exception("Caught exception, exiting with distinct code...")
                        sys.exit(DOWNLOAD_ERROR_EXIT_CODE)

                    with open(os.path.join(model_dir, out_name), "wb+") as fd:
                        for chunk in r.iter_content(chunk_size=8192):
                            fd.write(chunk)

            if resumable:
                # We successfully downloaded all the artifacts from a previous run. Override
                # the pretrained model mode and we're done!
                pretrained_model_mode = "continue"
                break
            else:
                # We weren't able to get all of the necessary artifacts; try the next previous run
                prev_run_id -= 1

    if pretrained_model_mode:
        if len(script_args) < PRETRAINED_MODEL_MODE_ARG_NUMBER:
            script_args.append(pretrained_model_mode)
        else:
            script_args[PRETRAINED_MODEL_MODE_ARG_NUMBER - 1] = pretrained_model_mode
    result = subprocess.run([TRAINING_SCRIPT, *script_args], check=False)
    if result.returncode != 0:
        logger.exception(
            f"taskcluster/scripts/pipeline/train-taskcluster.sh failed with exit code {result.returncode}"
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
