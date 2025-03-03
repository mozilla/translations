"""
Runs models for use within Firefox via local Remote Settings.

The artifacts will be saved to: ./data/artifacts

To run the local models:
 * Open Firefox (debug builds don't seem to work)
 * Run the following from the web console.

    ChromeUtils.importESModule("resource://services-settings/remote-settings.sys.mjs")
        .RemoteSettings("translations-wasm").verifySignature = false;
    ChromeUtils.importESModule("resource://services-settings/remote-settings.sys.mjs")
        .RemoteSettings("translations-models").verifySignature = false;

 * Install the Remote Settings Devtools xpi
   https://github.com/mozilla-extensions/remote-settings-devtools/releases
 * Click the addon's button
 * Change the Environment from Prod to Local
 * Click "Clear all"
 * Click "Sync"
 * Once you are done, make sure to switch the Environment back to Prod.

Usage:

- Load a model from a task group
  task local-remote-settings -- --taskgroup_ids I9uKJEPvQd-1zeItJK0cOQ aAVZJcsXQg-vfGIjHmcCTw

- Run a local mirror of the production models
  task local-remote-settings -- --sync_remote_settings
"""

import re
import gzip
import sys
import time
import json
import yaml
import shutil
import argparse
import requests
import threading
import subprocess
import taskcluster
from uuid import uuid4
from pathlib import Path
from kinto_http import Client
from typing import IO, Callable, Optional, Type, Union
from pipeline.common.logging import get_logger
from pipeline.common.downloads import stream_download_to_file
from utils.common.taskcluster_api import (
    ListArtifacts,
    ListDependentTasks,
    ListTaskGroup,
    TaskAndStatus,
)
from utils.common.remote_settings import (
    ModelRecord,
    ModelsResponse,
    WasmResponse,
    WasmRecord,
    models_collection,
    wasm_collection,
    get_prod_records_url,
)

logger = get_logger("util")
docker_logger = get_logger("docker")

mount_path = Path(__file__).parent / "mount"
data_path = (Path(__file__).parent / "../../data").resolve()

attachments_path = data_path / "remote-settings/attachments"
attachments_path.mkdir(parents=True, exist_ok=True)

models_path = data_path / "remote-settings/models"
models_path.mkdir(parents=True, exist_ok=True)

bucket = "main"


def sync_records(
    remote_settings: Client,
    collection: str,
    record_response_class: Union[Type[WasmResponse], Type[ModelsResponse]],
    record_class: Union[Type[WasmRecord], Type[ModelRecord]],
):
    """
    Sync records from the production Remote Settings with a local version of it.
    """
    logger.info(f'Syncing records for "{collection}"')
    url = get_prod_records_url(collection)
    response = requests.get(url)
    response.raise_for_status()
    records_response = record_response_class(**response.json())
    new_records = {data.id: data for data in records_response.data}
    existing_records = [
        record_class(**data)
        for data in remote_settings.get_records(bucket=bucket, collection=collection)
    ]
    for record in existing_records:
        id = record.id
        if new_records.get(id):
            logger.info(f"Record exists {record.name} {record.version}")
            # The new record already exists.
            del new_records[id]
        else:
            logger.info(f"Removing outdated record {record.name} {record.version}")
            remote_settings.delete_record(id=id, collection=collection, bucket=bucket)

    for record in new_records.values():
        logger.info(f"Creating record {record.name} {record.version}")

        attachment = record.attachment
        assert attachment

        if ".." in attachment.location:
            raise Exception(f"Attachment location changes directory {attachment.location}")

        # TODO - This needs an upstream fix.
        # remote_settings.create_record(
        #     id=record.id,
        #     collection=collection,
        #     bucket=bucket,
        #     data=json.loads(record.json()),
        # )

        record_name = Path(record.name)
        cache_dir = attachments_path / f"sync-{collection}"
        cache_dir.mkdir(exist_ok=True)
        attachment_file_path = (
            cache_dir / f"{record_name.stem}-v{record.version}{record_name.suffix}"
        )
        if attachment_file_path.exists():
            logger.info(f"✅ {attachment_file_path}")
        else:
            download_url = (
                f"https://firefox-settings-attachments.cdn.mozilla.net/{attachment.location}"
            )
            with attachment_file_path.open("wb") as attachment_file:
                try:
                    logger.info(
                        f"⬇️ Downloading {record.name} {record.version} from {download_url}"
                    )
                    response = requests.get(download_url, stream=True, allow_redirects=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        attachment_file.write(chunk)

                except Exception as e:
                    logger.info(
                        f"Error occurred while downloading attachment {record.name} {record.version}: {e}"
                    )

        record_data = json.loads(record.json())
        del record_data["attachment"]

        create_record_with_attachment(
            record.id,
            collection,
            attachment.mimetype,
            attachment_file_path,
            record_data,
        )

        # TODO - This needs an upstream fix.
        # remote_settings.add_attachment(
        #     id=record.id,
        #     filepath=str(attachment_file_path),
        #     collection=collection,
        #     bucket=bucket,
        #     mimetype=attachment.mimetype,
        # )

        logger.info(f"Attachment downloaded and added {record.name} {record.version}")


def create_remote_settings_environment(remote_settings: Client):
    logger.info("Ensuring the buckets and collections are created")
    remote_settings.create_bucket(id=bucket, if_not_exists=True)
    remote_settings.create_collection(id=wasm_collection, bucket=bucket, if_not_exists=True)
    remote_settings.create_collection(id=models_collection, bucket=bucket, if_not_exists=True)


class DockerContainerManager:
    def __init__(
        self,
        container_name: str,
        image: str,
        volumes: dict[str, str],
        env_vars: dict[str, str],
        ports: dict[int, int],
    ):
        """
        Initializes the Docker container manager.

        container_name: Name of the Docker container.
        image: Docker image to run.
        volumes : Dictionary of host-to-container volume mappings.
        env_vars : Dictionary of environment variables to pass to the container.
        ports : Dictionary of host-to-container port mappings.
        """
        self.container_name = container_name
        self.image = image
        self.volumes = volumes
        self.env_vars = env_vars
        self.ports = ports
        self.process = None

    def _build_docker_command(self):
        """Builds the Docker run command."""
        cmd = ["docker", "run", "--rm", "--name", self.container_name]

        # Add volumes
        for host_path, container_path in self.volumes.items():
            cmd += ["--volume", f"{host_path}:{container_path}"]

        # Add environment variables
        for key, value in self.env_vars.items():
            cmd += ["--env", f"{key}={value}"]

        # Add port mappings
        for host_port, container_port in self.ports.items():
            cmd += ["--publish", f"{host_port}:{container_port}"]

        cmd += [self.image]

        return cmd

    def stream_output(self, stream: IO[str], log: Callable):
        """
        Stream the output from a subprocess stream (stdout or stderr) in real time.

        Args:
            stream (io.TextIOWrapper): The stream to read from (e.g., stdout or stderr).
        """

        def stream_reader():
            try:
                for line in iter(stream.readline, ""):
                    log(f"{line.strip()}")
            except Exception as e:
                log(f"Error reading from stream: {e}")
            finally:
                stream.close()

        # Use a separate thread to avoid blocking
        thread = threading.Thread(target=stream_reader, daemon=True)
        thread.start()

    def stop_and_remove_docker(self):
        # Stop and remove any existing container with the same name
        logger.info("Stopping translations-remote-settings.")
        subprocess.run(
            ["docker", "stop", self.container_name],
            check=False,
        )

    def start(self):
        """Starts the Docker container."""
        self.stop_and_remove_docker()

        logger.info(f"Starting Docker container '{self.container_name}'...")
        docker_command = self._build_docker_command()

        logger.info(f"Running: {' '.join(docker_command)}")
        # Start the Docker container as a subprocess
        self.process = subprocess.Popen(
            docker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        # Stream Docker output to stdout
        logger.info("Docker container is running.")
        logger.info(f"Access it at: http://localhost:{list(self.ports.keys())[0]}/v1/admin")

        # Stream stdout and stderr asynchronously
        try:
            stdout = self.process.stdout
            assert stdout
            self.stream_output(stdout, docker_logger.info)
            stderr = self.process.stderr
            assert stderr
            self.stream_output(stderr, docker_logger.error)
        except Exception as e:
            docker_logger.error(f"Error streaming output: {e}")

    def wait(self):
        """Waits for the Docker container process to complete, handling interruptions."""
        if not self.process:
            logger.info("No Docker container process to wait for.")
            return

        logger.info("Press Ctrl-C to stop the container.")
        try:
            self.process.wait()
        except KeyboardInterrupt:
            logger.info("\nStopping Docker container...")
            self.process.terminate()
            self.stop_and_remove_docker()
            self.process.wait()
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            self.process.terminate()
            self.stop_and_remove_docker()
            self.process.wait()


def add_model_from_taskgroup_id(
    queue: taskcluster.Queue, remote_settings: Client, taskgroup_id: str
):
    logger.info(f"Looking up task group information for {taskgroup_id}")
    list_task_group = ListTaskGroup.call(queue, taskgroup_id)
    tasks = list_task_group.tasks

    export_task = next((t for t in tasks if t.task.metadata.name.startswith("export-")), None)

    if export_task:
        add_model_from_export_task(queue, remote_settings, export_task)
        return

    logger.info(f"Could not find export task for {taskgroup_id}, checking for Train actions next.")

    train_actions = [t for t in tasks if t.task.metadata.name == "Action: Train"]

    if not train_actions:
        logger.info("No train actions were found.")

    logger.info(f"Found {len(train_actions)} train action tasks")
    for train_action in train_actions:
        action_task_id = train_action.status.taskId
        dependents = ListDependentTasks.call(queue, action_task_id)
        # We only need to get the first task to identify its Task Group ID.

        export_task = next(
            (t for t in dependents.tasks if t.task.metadata.name.startswith("export-")),
            None,
        )

        if export_task:
            if export_task.status.state == "completed":
                logger.info(f"Found the action {action_task_id}'s export task")
                add_model_from_export_task(queue, remote_settings, export_task)
            else:
                logger.info(
                    f"Found the action {action_task_id}'s export task, but it was not completed"
                )
        else:
            logger.info(f"The action {action_task_id} didn't produce an export tasks.")
            for t in dependents.tasks:
                logger.debug(f" - {t.task.metadata.name}")


def add_model_from_export_task_id(queue: taskcluster.Queue, remote_settings: Client, task_id: str):
    add_model_from_export_task(queue, remote_settings, TaskAndStatus.call(queue, task_id))


def add_model_from_export_task(
    queue: taskcluster.Queue, remote_settings: Client, export_task: TaskAndStatus
):
    last_run = export_task.status.runs[-1]
    assert last_run, "A run was found in the export task"

    action_task_id = export_task.status.taskGroupId
    action_task = TaskAndStatus.call(queue, action_task_id)

    response = requests.get(
        queue.buildUrl("getLatestArtifact", action_task.status.taskId, "public/parameters.yml"),
        stream=True,
    )
    response.raise_for_status()
    parameters = yaml.safe_load(response.text)
    config = parameters["training_config"]
    experiment_name = config["experiment"]["name"]

    match = re.search(r"export-(?P<src>\w+)-(?P<trg>\w+)+", export_task.task.metadata.name)
    assert match
    src = match.group("src")
    trg = match.group("trg")

    logger.info(
        f"Looking up the artifacts for {export_task.task.metadata.name} ({export_task.status.taskId})"
    )
    list_artifacts = ListArtifacts.call(queue, export_task.status.taskId, last_run.runId)
    artifacts = list_artifacts.artifacts

    # public/build/lex.50.50.enlt.s2t.bin.gz
    lex = next((a for a in artifacts if "/lex." in a.name), None)
    # public/build/model.enlt.intgemm.alphas.bin.gz
    model = next((a for a in artifacts if "/model." in a.name), None)
    # public/build/vocab.enlt.spm.gz
    vocab = next((a for a in artifacts if "/vocab." in a.name), None)

    if not lex:
        raise Exception("Could not find the lexical shortlist in artifacts")
    if not model:
        raise Exception("Could not find the model in artifacts")
    if not vocab:
        raise Exception("Could not find the vocab in artifacts")

    model_path = models_path / f"{experiment_name}-{src}-{trg}-{export_task.status.taskId}"
    model_path.mkdir(exist_ok=True)
    config_path = model_path / "config.yml"

    logger.info("Ensuring the artifacts are downloaded")

    # Write out the config.
    if config_path.exists():
        logger.info(f"✅ {config_path}")
    else:
        config_text = yaml.safe_dump(config)
        with config_path.open("wt") as config_file:
            config_file.write(config_text)
            logger.info(f"Destination: {config_path}")

    downloads = [
        (lex, "lex.s2t.bin"),
        (model, "model.bin"),
        (vocab, "vocab.spm"),
    ]

    # 1. Cache the artifact files locally to `model_path`
    # 2. Add them as records to Remote Settings.
    for artifact, filename in downloads:
        # First cache it locally.
        destination = model_path / filename
        if destination.exists():
            logger.info(f"✅ {destination}")
        else:
            zipped_file = f"{destination}.gz"
            stream_download_to_file(
                queue.buildUrl("getLatestArtifact", export_task.status.taskId, artifact.name),
                zipped_file,
            )
            # Decompress the gzip.
            logger.info(f"Decompressing from {zipped_file} to {destination}")
            with gzip.open(zipped_file, "rb") as f_in:
                with open(destination, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Add it to Remote Settings, which retain it in-memory.
        record = ModelRecord(
            name=filename,
            schema=0,
            fromLang=src,
            toLang=trg,
            # Include the experiment name as a variant.
            variant=experiment_name,
            version="1.0",
            fileType=filename.split(".")[0],  # lex, model, vocab
            attachment=None,
            filter_expression="",
            id=str(uuid4()),
            last_modified=1728419357986,  # This is just a plausible static value.
        )
        remote_settings.create_record(
            id=record.id,
            collection=models_collection,
            bucket=bucket,
            data=json.loads(record.json()),
        )
        remote_settings.add_attachment(
            id=record.id,
            filepath=destination,
            collection=models_collection,
            bucket=bucket,
        )
        logger.info(f"Attachment added {record.name} {record.version}")


def wait_for_remote_settings():
    max_attempts = 500
    timeout = 0.5
    url = "http://localhost:8888/__heartbeat__"

    logger.info(f"Checking to see if Remote Settings is ready: {url}")

    for attempt in range(max_attempts):
        try:
            logger.info(f"Checking {url}")
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Remote Settings is ready after {attempt + 1} attempts.")
                return True
        except requests.RequestException:
            pass

        time.sleep(timeout)

    raise Exception("Remote Settings is not ready after maximum attempts.")


def create_record_with_attachment(
    record_id: str,
    collection: str,
    mime_type: str,
    attachment_path: Path,
    record_data: dict,
) -> None:
    url = f"http://localhost:8888/buckets/{bucket}/collections/{collection}/records/{record_id}/attachment"

    logger.info(f"Posting record to {url}")
    with open(attachment_path, "rb") as attachment:
        files = {"attachment": (attachment_path.name, attachment, mime_type)}
        form_data = {"data": json.dumps(record_data)}
        response: Optional[requests.Response] = None
        exception: Optional[Exception] = None
        for _ in range(10):
            exception = None
            response = None
            try:
                response = requests.post(url, files=files, data=form_data)
                if response.ok:
                    return
            except Exception as e:
                logger.warning(f"An exception occurred while creating a record: {e}")
                exception = e

            if response:
                logger.warning(f"Response was not ok, code: {response.status_code}")

        if response:
            response.raise_for_status()
        elif exception:
            raise exception


def log_records(remote_settings: Client) -> None:
    wasm_records = remote_settings.get_records(collection=wasm_collection, bucket=bucket)
    model_records = remote_settings.get_records(collection=models_collection, bucket=bucket)

    logger.info("Wasm records:")
    for record_json in wasm_records:
        record = WasmRecord(**record_json)
        logger.info(f" - {record.name} {record.version}")

    logger.info("Model records:")
    for record_json in model_records:
        record = ModelRecord(**record_json)
        logger.info(f" - {record.name} {record.fromLang}-{record.toLang}")

    logger.info("Remote Settings is ready: http://localhost:8888/v1/admin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--sync_remote_settings",
        action="store_true",
        help="Mirror the production Remote Settings models",
    )
    parser.add_argument(
        "--taskgroup_ids",
        type=str,
        help="Task groups that contain an export- task, or train actions.",
        nargs="*",
    )
    parser.add_argument(
        "--export_task_id",
        type=str,
        help="The export-{src}-{trg} task id to use for models",
        nargs="*",
    )

    args = parser.parse_args()
    taskgroup_ids: list[str] = args.taskgroup_ids or []
    export_task_ids: list[str] = args.export_task_id or []
    sync_remote_settings: bool = args.sync_remote_settings

    if not sync_remote_settings and not export_task_ids and not taskgroup_ids:
        # Print the help when nothing is provided.
        print("--sync_remote_settings, --export_task_ids, or --taskgroup_ids must be provided\n")
        parser.print_help()
        sys.exit(0)

    if sync_remote_settings and (export_task_ids or taskgroup_ids):
        print("--sync_remote_settings can not be used with --export_task_ids or --taskgroup_ids")
        sys.exit(1)

    docker = DockerContainerManager(
        container_name="translations-remote-settings",
        image="mozilla/remote-settings",
        volumes={
            str(attachments_path): "/tmp/attachments",
            str(mount_path): "/app/mount",
        },
        env_vars={"KINTO_INI": "mount/translations.ini"},
        ports={8888: 8888},
    )

    logger.info("Starting remote settings")
    docker.start()

    # Initialize Remote Settings.
    wait_for_remote_settings()
    remote_settings = Client(server_url="http://localhost:8888/v1")
    create_remote_settings_environment(remote_settings)

    # The Wasm will be used from the production Remote Settings.
    sync_records(remote_settings, wasm_collection, WasmResponse, WasmRecord)

    if taskgroup_ids or export_task_ids:
        # Pull specific models from Taskcluster.
        queue = taskcluster.Queue({"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})

        for taskgroup_id in taskgroup_ids:
            add_model_from_taskgroup_id(queue, remote_settings, taskgroup_id)

        for task_id in export_task_ids:
            add_model_from_export_task_id(queue, remote_settings, task_id)
    else:
        # Sync records from the production Remote Settings.
        sync_records(remote_settings, models_collection, ModelsResponse, ModelRecord)

    log_records(remote_settings)

    docker.wait()


if __name__ == "__main__":
    main()
