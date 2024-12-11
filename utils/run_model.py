#!/usr/bin/env python3
"""
Run Marian models locally. When a --task_id is given, the model is downloaded to the ./data
directory, and immediately run. This process loads the marian-server binary in the background,
and communicates to it via a websocket.

Usage:

    task run-model -- --task_id fpJkLJRaRAqTxgG0ARwR1w
"""

import argparse
import atexit
import os
import re
import subprocess
import sys
import time
from typing import Any, Optional

from websocket import WebSocket, create_connection

import taskcluster
from pipeline.common.downloads import stream_download_to_file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TC_MOZILLA = "https://firefox-ci-tc.services.mozilla.com"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data"))


def get_language_pair_from_task_name(task_name):
    pattern = r"""
        \btrain-            # "train-"
        \w+-                # "backwards", "student", etc.
        (?P<language_pair>  # Start a capture group 'language_pair'
            \w{2}-\w{2}     # Match the actual language pair
        )
    """
    match = re.search(pattern, task_name, re.VERBOSE)
    return match.group("language_pair") if match else None


def download_model_from_task(output: str, task_id: str) -> str:
    """Downloads to the output directory in a subfolder {src}-{trg}-{task_id}"""
    options = {"rootUrl": ("%s" % TC_MOZILLA)}
    queue: Any = taskcluster.Queue(options=options)
    task = queue.task(task_id)
    status = queue.status(task_id)

    if status["status"]["state"] != "completed":
        raise Exception("The task was not completed")

    task_name = task["metadata"]["name"]
    language_pair = get_language_pair_from_task_name(task_name)
    if not language_pair:
        raise Exception(f'Could not find the language pair for the task "{task_name}"')

    print(f"Downloading from task {task_name}")

    artifacts = queue.listLatestArtifacts(task_id)["artifacts"]

    for artifact in artifacts:
        print(artifact["name"])

    decoder = next(
        a for a in artifacts if a["name"].endswith("/final.model.npz.best-chrf.npz.decoder.yml")
    )
    model = next(a for a in artifacts if a["name"].endswith("/final.model.npz.best-chrf.npz"))
    vocab = next(a for a in artifacts if a["name"].endswith("/vocab.spm"))

    if not decoder:
        raise Exception("Could not find the decoder in artifacts")
    if not model:
        raise Exception("Could not find the model in artifacts")
    if not vocab:
        raise Exception("Could not find the vocab in artifacts")

    model_path = os.path.join(output, f"{language_pair}-{task_id}")
    print(model_path)

    os.makedirs(model_path, exist_ok=True)

    print(
        f'Downloading models from "{task_name}": '
        f"https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}"
    )

    downloads = [
        (decoder, "decoder.yml"),
        (model, "model.npz"),
        (vocab, "vocab.spm"),
    ]

    for artifact, filename in downloads:
        stream_download_to_file(
            queue.buildUrl("getLatestArtifact", task_id, artifact["name"]),
            os.path.join(model_path, filename),
        )

    print(f"Model files are available at: {model_path}")
    return model_path


def find_model(output: str, task_id: str) -> Optional[str]:
    for dir_name in os.listdir(output):
        if f"-{task_id}" in dir_name:
            return os.path.join(output, dir_name)
    return None


def connect_to_ws(port: int) -> WebSocket:
    """Attempts to connect to a websocket with multiple attempts."""
    attempt = 0
    ws = None
    max_retries = 100
    retry_delay_sec = 1

    uri = f"ws://localhost:{port}/translate"
    print(f"Attempting to connect to {uri}", end="")

    # Retry multiple times to connect.
    while attempt < max_retries:
        try:
            ws = create_connection(uri)
            break
        except Exception:
            # Attempt to reconnect
            print(".", end="")
            time.sleep(retry_delay_sec)
            attempt += 1

    print()

    if ws is None:
        print("Failed to connect to the Marian server.")
        sys.exit(1)

    print("Connected to Marian server.\n")
    return ws


def translate_over_websocket(port: int):
    """
    Opens a websocket connection to the Marian server, and accepts translation input from stdin.
    """
    ws = connect_to_ws(port)

    try:
        while True:
            print("Enter text to translate:")
            line = input("> ")

            ws.send(line.encode("utf-8"))
            translation = ws.recv()

            print("\nTranslation:")
            print(">", translation)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error communicating with Marian server: {e}")

    ws.close()


def main() -> None:
    if not os.environ.get("IS_DOCKER"):
        # Re-run the command in docker if it wasn't started.
        args = sys.argv[1:]
        subprocess.check_call(["task", "docker-run", "--", "task", "run-model", "--", *args])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id",
        type=str,
        help="The task ID that contains model artifacts",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a local folder containing a model",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Where to save the models",
        default=os.path.abspath(os.path.join(DATA_DIR, "models")),
    )
    parser.add_argument(
        "--port", type=int, help="The port that the Marian server listens over", default=8886
    )
    parser.add_argument(
        "--output_marian", default=False, help="Include the output from the Marian server."
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Ensure the model is downloaded and we can get the model path. Re-use an existing model if
    # it is alreaady downloaded.
    model_path = None
    if args.model:
        model_path = args.model
    elif args.task_id:
        model_path = find_model(args.output, args.task_id)
        if model_path:
            print(f"Model with task ID {args.task_id} has been downloaded at {model_path}")
        else:
            model_path = download_model_from_task(args.output, args.task_id)
    else:
        raise Exception("Provide either a --task_id or a --model")

    decoder = os.path.join(model_path, "decoder.yml")
    model = os.path.join(model_path, "model.npz")
    vocab = os.path.join(model_path, "vocab.spm")

    if not os.path.exists(decoder):
        raise Exception(f"Decoder was not found: {decoder}")
    if not os.path.exists(model):
        raise Exception(f"Model was not found: {model}")
    if not os.path.exists(vocab):
        raise Exception(f"Vocab was not found: {vocab}")

    if args.output_marian:
        stdout = None
        stderr = None
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    command = (
        "/builds/worker/tools/marian-dev/build/marian-server "
        f"--config {decoder} "
        f"--models {model} "
        f"--vocabs {vocab} {vocab} "
        f"--port {args.port}"
    )

    marian_server = subprocess.Popen(command, shell=True, stdout=stdout, stderr=stderr)

    atexit.register(exit_handler, marian_server)
    translate_over_websocket(args.port)


def exit_handler(marian_server):
    marian_server.terminate()


if __name__ == "__main__":
    main()
