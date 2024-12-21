"""
Downloads artifacts from a Taskcluster Task Group. This command supports the following modes:
 - logs
 - evals
 - model
"""

import os
import sys
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the pipeline is available on the path.
sys.path.append(os.path.join(CURRENT_DIR, ".."))

import argparse
import csv
import enum
import os
import re

import requests

import taskcluster
from pipeline.common.downloads import stream_download_to_file
from taskcluster.download import downloadArtifactToBuf

TC_MOZILLA = "https://firefox-ci-tc.services.mozilla.com"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data"))

# to parse evaluation task tag
# examples:
# evaluate-teacher-flores-flores_aug-title_devtest-lt-en-1_2
# evaluate-teacher-mtdata-mtdata_aug-mix_Neulab-tedtalks_test-1-eng-lit-lt-en-1_2
# evaluate-finetuned-student-sacrebleu-wmt19-lt-en
eval_regex = re.compile(
    r"evaluate-"
    r"(?P<model>finetuned-student|teacher-ensemble|teacher|student|quantized|backward)"
    r"-"
    r"(?P<importer>flores|mtdata|sacrebleu)"
    r"(?P<extra_importer>-flores|-mtdata|-sacrebleu)?"
    r"[_-]"
    r"(?P<aug>aug-[^_]+)?"
    r"_?"
    r"(?P<dataset>[-\w_]*?)"
    r"-"
    r"(?P<extra_lang>([a-z]{3}-[a-z]{3})?"
    r"-?"
    r"(?P<lang>[a-z]{2}-[a-z]{2}))"
    r"-?"
    r"(?P<suffix>[\d_]+)?"
)


class Mode(enum.Enum):
    logs = "logs"
    evals = "evals"
    model = "model"


def download_logs(group_id, output):
    options = {"rootUrl": TC_MOZILLA}
    queue = taskcluster.Queue(options=options)
    group: Any = queue.listTaskGroup(group_id)

    task_found = False
    for task in group["tasks"]:
        if task["status"]["state"] not in ("completed", "running"):
            continue

        label = task["task"]["tags"]["kind"]
        if ("train" not in label and "finetune" not in label) or "vocab" in label:
            continue

        task_found = True

        task_id = task["status"]["taskId"]

        task_obj: Any = queue.task(task_id)
        task["status"]["runs"][-1]["runId"]
        task_obj_label = task_obj["tags"]["label"].replace("/", "_")

        os.makedirs(output, exist_ok=True)
        output_path = os.path.join(output, f"{task_obj_label}.log")

        url = queue.buildUrl("getLatestArtifact", task_id, "public/logs/live.log")
        resp = requests.get(url, stream=True, timeout=5)
        print(f"Downloading {url}")

        log_lines = []
        start_writing = False
        try:
            for line in resp.iter_lines():
                line_str = line.decode()

                if "[marian]" in line_str:
                    start_writing = True

                if start_writing:
                    log_lines.append(re.sub(r"\[task .*Z\] ", "", line_str))
        except requests.exceptions.ConnectionError:
            pass

        print(f"Writing to {output_path}")
        with open(output_path, "w") as f:
            f.write("\n".join(log_lines))

    if not task_found:
        print(f"No logs were found for {group_id}")


def donwload_evals(group_id, output):
    options = {"rootUrl": ("%s" % TC_MOZILLA)}
    queue = taskcluster.Queue(options=options)
    group: Any = queue.listTaskGroup(group_id)

    results = []

    for task in group["tasks"]:
        if task["status"]["state"] != "completed":
            continue

        label = task["task"]["tags"]["kind"]
        if "evaluate" not in label:
            continue

        task_id = task["status"]["taskId"]

        task_obj: Any = queue.task(task_id)
        task["status"]["runs"][-1]["runId"]
        task_obj_label = task_obj["tags"]["label"].replace("/", "_")

        artifacts_response: Any = queue.listLatestArtifacts(task_id)
        artifacts = artifacts_response["artifacts"]
        artifact_name = [
            artifact["name"] for artifact in artifacts if artifact["name"].endswith(".metrics")
        ][0]

        print(f"Downloading {artifact_name} for {task_obj_label}")
        content, _ = downloadArtifactToBuf(
            taskId=task["status"]["taskId"],
            name=artifact_name,
            queueService=queue,
        )
        bleu, chrf, _ = content.tobytes().decode().split("\n")

        match = eval_regex.match(task_obj_label)
        if not match:
            print(f"Cannot match {task_obj_label}")
            raise ValueError(f"Cannot match {task_obj_label}")

        groups = match.groupdict()
        model = groups["model"]
        importer = groups["importer"]
        augmentation = groups.get("aug", "") or ""
        dataset = groups["dataset"]
        groups["lang"]
        suffix = groups.get("suffix", "") or ""

        result = (model + suffix, f"{importer}_{dataset}", augmentation, bleu, chrf)
        print(f"Result: {result}")
        results.append(result)

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, f"{group_id}-metrics.csv")
    with open(output_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Model", "Dataset", "Augmentation", "BLEU", "chrF"])
        for res in results:
            csv_writer.writerow(res)


def download_model(group_id: str, output: str):
    options = {"rootUrl": ("%s" % TC_MOZILLA)}
    queue = taskcluster.Queue(options=options)
    group: Any = queue.listTaskGroup(group_id)

    for task in group["tasks"]:
        if task["status"]["state"] != "completed":
            continue

        if task["task"]["tags"]["kind"] != "export":
            continue

        task_id = task["status"]["taskId"]
        task_name = task["task"]["metadata"]["name"]
        language_pair = task_name.replace("export-", "")

        artifacts_response: Any = queue.listLatestArtifacts(task_id)
        artifacts = artifacts_response["artifacts"]
        model_artifacts = [
            artifact for artifact in artifacts if not artifact["name"].endswith(".log")
        ]

        model_path = os.path.join(output, language_pair)

        os.makedirs(model_path, exist_ok=True)

        print(
            f'Downloading models from "{task_name}": '
            f"https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}"
        )

        for artifact in model_artifacts:
            url = queue.buildUrl("getLatestArtifact", task_id, artifact["name"])
            path = os.path.join(model_path, os.path.basename(artifact["name"]))
            stream_download_to_file(url, path)

        print(f"Model files are available at: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        metavar="OUTPUT",
        type=str,
        help="Output directory to save logs. Defaults to the data directory.",
    )
    parser.add_argument(
        "--task-group-id",
        metavar="TASK_GROUP_ID",
        required=True,
        type=str,
        help="ID of a Taskcluster task group",
    )
    parser.add_argument(
        "--mode",
        metavar="MODE",
        type=Mode,
        choices=Mode,
        required=True,
        help="What to download: " + ", ".join([m.name for m in Mode]),
    )

    args = parser.parse_args()
    group_id: str = args.task_group_id
    mode: Mode = args.mode

    output: str
    if args.output:
        output = args.output
    else:
        output = os.path.join(DATA_DIR, f"taskcluster-{mode.value}")

    if mode == Mode.logs:
        download_logs(group_id, output)
    elif mode == Mode.evals:
        donwload_evals(group_id, output)
    elif mode == Mode.model:
        download_model(group_id, output)


if __name__ == "__main__":
    main()
