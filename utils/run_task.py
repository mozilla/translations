import argparse
import gzip
import logging
import os
import shlex
import shutil
import subprocess
import sys
import json
from pathlib import Path
import tarfile
import textwrap
from typing import Any, Optional, Union
import taskcluster
import re
import platform
import hashlib
from pipeline.common.downloads import stream_download_to_file

from zstandard import ZstdDecompressor

# Taskcluster commands can either be a single list of commands, or a nested list.
Commands = Union[list[str], list[list[str]]]

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

ROOT_PATH = (Path(__file__).parent / "..").resolve()
DATA_PATH = ROOT_PATH / "data"
RUN_TASK = DATA_PATH / "run_task"
RUN_TASK.mkdir(exist_ok=True, parents=True)

SKIP_ARTIFACTS = {
    # 3.5GB - And GPU only.
    "cuda-toolkit.tar.zst"
}

queue = taskcluster.Queue({"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})


def log_multiline(string: str):
    for line in string.split("\n"):
        logger.info(line)


def fail_on_error(result: subprocess.CompletedProcess[bytes]):
    """When a process fails, surface the stderr."""
    if not result.returncode == 0:
        logger.error(f"❌ Task failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def split_on_ampersands_operator(command_parts: list[str]) -> list[list[str]]:
    """Splits a command with the bash && operator into multiple lists of commands."""
    multiple_command_parts: list[list[str]] = []
    sublist: list[str] = []
    for part in command_parts:
        if part.strip().startswith("&&"):
            command_part = part.replace("&&", "").strip()
            if len(command_part):
                sublist.append(command_part)
            multiple_command_parts.append(sublist)
            sublist = []
        else:
            sublist.append(part)
    multiple_command_parts.append(sublist)
    return multiple_command_parts


def extract_file_archive(path: Path, to_dir: Path):
    to_dir.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".zst":
        with open(path, "rb") as compressed:
            dctx = ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    members = list(tar)
                    if all((to_dir / m.name).exists() for m in members if m.isfile()):
                        logger.info(f"✅ extracted files all exist: {path}")
                        return

        # Rewind and extract since tar was exhausted during check
        logger.info(f"📦 Extracting archive {path.name} -> {to_dir}")
        with open(path, "rb") as compressed:
            dctx = ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=to_dir)

    else:
        logger.error(f"⚠️ Unsupported extract format for {path.name}")


def extract_single_file(path: Path, to_file: Path):
    logger.info(f"📦 Extracting file {path.name} -> {to_file}")

    if path.suffix == ".zst":
        with open(path, "rb") as src, open(to_file, "wb") as outfile:
            dctx = ZstdDecompressor()
            dctx.copy_stream(src, outfile)

    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as gz_in:
            with open(to_file, "wb") as ooutfilet:
                shutil.copyfileobj(gz_in, ooutfilet)

    else:
        logger.error(f"⚠️ Unsupported extract format for {path.name}")


def download_fetches(env: dict, task_fetches_dir: Path):
    if "MOZ_FETCHES" not in env:
        logger.info("[fetches] No fetches")
        return
    fetches = json.loads(env["MOZ_FETCHES"])
    cache_dir = RUN_TASK / "fetches-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[fetches] Fetches from task:")
    for line in json.dumps(fetches, indent=2).split("\n"):
        logger.info(f"[fetches] {line}")

    for entry in fetches:
        task_id: str = entry["task"]
        artifact_path: str = entry["artifact"]
        extract = entry.get("extract", False)
        artifact_name = Path(artifact_path).name
        if artifact_name in SKIP_ARTIFACTS:
            logger.info("[fetches] ⏩ skipping: " + artifact_name)
            continue

        cached_path = cache_dir / f"{task_id}--{artifact_name}"
        local_path = task_fetches_dir / artifact_name

        # Download it if it is missing in the cache.
        if not cached_path.exists():
            logger.info(f"[fetches] ⬇️ Fetching {artifact_name} from task {task_id}...")
            url = queue.buildUrl("getLatestArtifact", task_id, artifact_path)
            stream_download_to_file(url, cached_path)
            logger.info(f"[fetches] 📁 cached: {cached_path.relative_to(ROOT_PATH)}")

        # Skip copying if extractable compressed file
        if extract and str(cached_path).endswith(".tar.zst"):
            # Extract all of the files into the task's fetches directory.
            logger.info(
                f"[fetches] 📤 extracting: {artifact_name} to {task_fetches_dir.relative_to(ROOT_PATH)}"
            )
            # We have no way to now if this has been extracted or not, so always run it.
            extract_file_archive(cached_path, task_fetches_dir)

        elif extract and cached_path.suffix in [".zst", ".gz"]:
            # For instance fetches/corpus.aln.zst becomes fetches/corpus.aln
            extract_target = (task_fetches_dir / artifact_name).with_suffix("")
            if extract_target.exists():
                logger.info(f"[fetches] ✅ exists: {extract_target.relative_to(ROOT_PATH)}")
            else:
                logger.info(
                    f"[fetches] 📤 extracting: {artifact_name} to {extract_target.relative_to(ROOT_PATH)}"
                )
                extract_single_file(cached_path, extract_target)
        elif local_path.exists():
            logger.info(f"[fetches] ✅ exists: {local_path.relative_to(ROOT_PATH)}")
        else:
            shutil.copy2(cached_path, local_path)
            logger.info(f"[fetches] 📤 copying: {local_path.relative_to(ROOT_PATH)}")


def get_command(commands: Commands) -> str:
    if isinstance(commands[-1], str):
        # Non-nested command, get the last string.
        return commands[-1]

    if isinstance(commands[-1][-1], str):
        # Nested command, get the last string of the last command.
        return commands[-1][-1]

    logger.info(commands)
    raise Exception("Unable to find a string in the nested command.")


def find_pipeline_script(commands: Commands) -> str:
    """
    Extract the pipeline script and arguments from a command list.

    Commands take the form:
    [
       ['chmod', '+x', 'run-task'],
       ['./run-task', '--translations-checkout=./checkouts/vcs/', '--', 'bash', '-c', "full command"]
    ]

    or

    [
          "/usr/local/bin/run-task",
          "--translations-checkout=/builds/worker/checkouts/vcs/",
          "--", "bash", "-c",
          "full command"
    ]
    """
    command = get_command(commands)

    # Match a pipeline script like:
    #   pipeline/data/dataset_importer.py
    #   $VCS_PATH/taskcluster/scripts/pipeline/train-taskcluster.sh
    #   $VCS_PATH/pipeline/alignment/generate-alignment-and-shortlist.sh
    match = re.search(
        r"""
        # Script group:
        (?P<script>
            (?:python3?[ ])?   # Allow the script to be preceded by "python3 " or "python ".
            \$VCS_PATH         # "$VCS_PATH"
            [\w\/-]*           # Match any directories before "/pipeline/"
            \/pipeline\/       # "/pipeline/"
            [\w\/-]+           # Match any directories after "/pipeline/"
            \.(?:py|sh)        # Match the .sh, or .py extension
        )
        """,
        command,
        re.X,
    )

    if not match:
        raise Exception(f"Could not find a pipeline script in the command: {command}")

    script = match.group("script")

    # Split the parts of the command.
    command_parts = command.split(script)

    if len(command_parts) < 2:
        raise Exception(f"Could not find {script} in: {command}")

    # Remove the preamble to the script, which is should be the pip install.
    command_parts[0] = ""

    # Join the command parts back together to reassemble the command.
    return script.join(command_parts).strip()


def find_requirements(commands: Commands) -> Optional[str]:
    command = get_command(commands)

    # Match the following:
    # pip install -r $VCS_PATH/pipeline/eval/requirements/eval.txt && ...
    #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    match = re.search(
        r"""
        pip3?\ install\ -r\ \$VCS_PATH\/  # Find the pip install.
        (?P<requirements>                 # Capture as "requirements"
            [\w\/\-\.]+                   # Match the path
        )
        """,
        command,
        re.X,
    )

    if match:
        return match.groupdict()["requirements"]

    return None


def get_task_command_and_env(
    task: dict[str, Any],
) -> tuple[list[str], Optional[str], dict[str, str]]:
    env = task["payload"]["env"]

    commands = task["payload"]["command"]
    pipeline_script = find_pipeline_script(commands)
    requirements = find_requirements(commands)

    logger.info(f'Running: "{task["metadata"]["name"]}":')
    log_multiline("\nCommands: " + json.dumps(commands, indent=2))
    log_multiline("\nRunning:" + json.dumps(pipeline_script, indent=2))
    log_multiline("\nEnv:" + json.dumps(env, indent=2))
    log_multiline("\nRequirements: " + requirements if requirements else "None")
    logger.info("")

    command_parts = [
        part
        for part in shlex.split(pipeline_script)
        # subprocess.run doesn't understand how to redirect stderr to stdout, so ignore this
        # if it's part of the command.
        if part != "2>&1"
    ]

    # The python binary will be picked by the run_task abstraction.
    if requirements and (command_parts[0] == "python" or command_parts[0] == "python3"):
        command_parts = command_parts[1:]

    # Return the full command.
    return command_parts, requirements, env


def assert_clean_git():
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    if result.stdout.strip():
        logger.error("❌ Git working directory is not clean. Please commit or stash changes.")
        sys.exit(1)


def get_current_repo_url():
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.error("❌ Failed to get git remote URL.")
        sys.exit(1)
    return result.stdout.strip()


def assert_remote_matches(task_repo_url: str):
    current_url = get_current_repo_url()

    # normalize to https format for GitHub
    current_url = re.sub(r"^git@github.com:(.+)\.git$", r"https://github.com/\1", current_url)

    if not task_repo_url.startswith(current_url):
        logger.error(
            f"❌ Task repo {task_repo_url} is not an upstream of current repo {current_url}"
        )
        sys.exit(1)


def hash_file(hash: Any, path: str):
    """
    Hash the contents of a file.
    """
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            hash.update(chunk)


def get_python_dirs(requirements: str) -> tuple[str, str]:
    """
    Creates a virtual environment for each requirements file that a task needs. The virtual
    environment is hashed based on the requirements file contents, and the system details. This
    way a virtual environment will be re-used between docker environments.
    """

    system_details = "-".join(
        [
            platform.system(),  # Linux
            platform.machine(),  # aarch64
            platform.release(),  # 5.15.49-linuxkit-pr
        ]
    )

    # Create a hash based on files and contents that would invalidate the python library.
    md5 = hashlib.md5()
    hash_file(md5, requirements)
    md5.update(system_details.encode("utf-8"))
    if os.environ.get("IS_DOCKER"):
        hash_file(md5, os.path.join(ROOT_PATH, "docker/Dockerfile"))
    hash = md5.hexdigest()

    requirements_stem = Path(requirements).stem
    environment = "docker" if os.environ.get("IS_DOCKER") else "native"
    venv_dir = os.path.abspath(
        os.path.join(DATA_PATH, "task-venvs", f"{environment}-{requirements_stem}-{hash}")
    )
    python_bin_dir = os.path.join(venv_dir, "bin")
    python_bin = os.path.join(python_bin_dir, "python")

    # Create the venv only if it doesn't exist.
    if not os.path.exists(venv_dir):
        try:
            logger.info(f"Creating virtual environment: {venv_dir}")
            subprocess.check_call(
                # Give the virtual environment access to the system site packages, as these
                # are installed via docker.
                ["python", "-m", "venv", "--system-site-packages", venv_dir],
            )

            logger.info(f"Installing setuptools: {requirements}")
            subprocess.check_call(
                [python_bin, "-m", "pip", "install", "--upgrade", "setuptools", "pip"],
            )

            logger.info(f"Installing: {requirements}")
            subprocess.check_call(
                [python_bin, "-m", "pip", "install", "-r", requirements],
            )
        except Exception as exception:
            logger.info("Removing the venv due to an error in its creation.")
            shutil.rmtree(venv_dir)
            raise exception
    logger.info(f"Using virtual environment {venv_dir}")

    return python_bin_dir, venv_dir


def run_task(
    task_id: str,
    task: dict[str, Any],
) -> Path:
    """
    Runs a task from the taskgraph. See artifacts/full-task-graph.json after running a
    test for the full list of task names
    """

    command_parts, requirements, task_env = get_task_command_and_env(task)

    # There are some non-string environment variables that involve taskcluster references
    # Remove these.
    for key in task_env:
        if not isinstance(task_env[key], str):
            task_env[key] = ""

    task_name = task["metadata"]["name"]
    work_dir = RUN_TASK / f"{task_name}-{task_id}"
    fetches_dir = work_dir / "fetches"
    artifacts_dir = work_dir / "artifacts"
    temp_fetches = RUN_TASK / "fetches-tmp"

    if fetches_dir.exists():
        logger.info("[fetches] The fetches exist from a previous run, preserving them.")
        if temp_fetches.exists():
            shutil.rmtree(temp_fetches)
        fetches_dir.rename(temp_fetches)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    artifacts_dir.mkdir()
    fetches_dir.mkdir(exist_ok=True)
    if temp_fetches.exists():
        temp_fetches.rename(fetches_dir)

    # shutil.copytree(ROOT_PATH / "pipeline", work_dir / "pipeline")
    download_fetches(task_env, fetches_dir)

    logger.info(f"Writing out: {work_dir / 'task.json'}")
    with (work_dir / "task.json").open("wt") as file:
        json.dump(task, file)

    for command_parts_split in split_on_ampersands_operator(command_parts):
        task_env = {
            # The following are set by the Taskcluster server.
            "TASK_ID": "fake_id",
            "RUN_ID": "0",
            "TASKCLUSTER_ROOT_URL": "https://some.cluster",
            **os.environ,
            **task_env,
            "TASK_WORKDIR": work_dir,
            "MOZ_FETCHES_DIR": fetches_dir,
            "BMT_MARIAN": fetches_dir,
            "VCS_PATH": ROOT_PATH,
        }

        # Expand out environment variables in environment, for instance MARIAN=$MOZ_FETCHES_DIR
        # and FETCHES=./fetches will be expanded to MARIAN=./fetches
        for key, value in task_env.items():
            if not isinstance(value, str):
                continue
            expanded_value = task_env.get(value[1:])
            if value and value[0] == "$" and expanded_value:
                task_env[key] = expanded_value

        # Ensure the environment variables are sorted so that the longer variables get replaced first.
        sorted_env = sorted(task_env.items(), key=lambda kv: kv[0])
        sorted_env.reverse()

        for index, p in enumerate(command_parts_split):
            part = (
                p.replace("$TASK_WORKDIR/$VCS_PATH", str(ROOT_PATH))
                .replace("$VCS_PATH", str(ROOT_PATH))
                .replace("$TASK_WORKDIR", str(work_dir))
                .replace("$MOZ_FETCHES_DIR", str(fetches_dir))
            )

            # Apply the task environment.
            for key, value in sorted_env:
                env_var = f"${key}"
                if env_var in part:
                    part = part.replace(env_var, value)

            command_parts_split[index] = part

        # If using a venv, prepend the binary directory to the path so it is used.
        if requirements:
            python_bin_dir, venv_dir = get_python_dirs(requirements)
            if python_bin_dir:
                task_env = {
                    **task_env,
                    "PATH": f'{python_bin_dir}:{os.environ.get("PATH", "")}',
                }
                if command_parts_split[0].endswith(".py"):
                    # This script is relying on a shebang, add the python3 from the executable instead.
                    command_parts_split.insert(0, os.path.join(python_bin_dir, "python3"))
            elif command_parts_split[0].endswith(".py"):
                # This script does not require a virtual environment.
                command_parts_split.insert(0, "python3")

            # We have to set the path to the C++ lib before the process is started
            # https://github.com/Helsinki-NLP/opus-fast-mosestokenizer/issues/6
            with open(requirements) as f:
                reqs_txt = f.read()
            if venv_dir and "opus-fast-mosestokenizer" in reqs_txt:
                lib_path = os.path.join(
                    venv_dir, "lib/python3.10/site-packages/mosestokenizer/lib"
                )
                logger.info(f"Setting LD_LIBRARY_PATH to {lib_path}")
                task_env["LD_LIBRARY_PATH"] = lib_path

        # There might be a better way to do this by passing the in the correct VCS_PATH.
        # command_parts_split = [  # noqa: PLW2901
        #     re.sub(r"^\/builds\/worker\/pipeline\/", "pipeline/", part)
        #     for part in command_parts_split
        # ]
        logger.info("┌──────────────────────────────────────────────────────────")
        logger.info("│ run_task: " + " ".join(command_parts_split))
        logger.info("└──────────────────────────────────────────────────────────")

        result = subprocess.run(
            command_parts_split,
            env=task_env,
            cwd=work_dir,
            check=False,
        )

        fail_on_error(result)

    return work_dir


def git_reset(commit_sha: str):
    logger.info(f"🔁 Switching to commit {commit_sha}")
    result = subprocess.run(["git", "switch", "--detach", commit_sha], check=False)
    if result.returncode != 0:
        logger.info("❌ Git switch failed.")
        sys.exit(1)


def print_tree(path: Path):
    """
    Print a tree view of the task directory.
    """
    span_len = 90
    span = "─" * span_len
    logger.info(f"┌{span}┐")

    for root, dirs, files in os.walk(path):
        level = root.replace(str(path), "").count(os.sep)
        indent = " " * 4 * (level)
        if level == 0:
            # For the root level, display the relative path to the data directory.
            folder_text = root.replace(f"{ROOT_PATH}/", "")
            folder_text = f"│ {folder_text}"
        else:
            folder_text = f"│ {indent}{os.path.basename(root)}/"
        logger.info(f"{folder_text.ljust(span_len)} │")
        subindent = " " * 4 * (level + 1)

        if len(files) == 0 and len(dirs) == 0:
            empty_text = f"│ {subindent} <empty folder>"
            logger.info(f"{empty_text.ljust(span_len)} │")
        for file in files:
            file_text = f"│ {subindent}{file}"
            bytes = f"{os.stat(os.path.join(root, file)).st_size} bytes"

            logger.info(f"{file_text.ljust(span_len - len(bytes))}{bytes} │")

    logger.info(f"└{span}┘")


def main():
    parser = argparse.ArgumentParser(description="Fetch Taskcluster task and its kind.yml")
    parser.add_argument("--task-id", required=True, help="Taskcluster taskId to fetch")
    parser.add_argument(
        "--checkout", action="store_true", help="Reset current repo to match task commit"
    )

    args = parser.parse_args()
    task_id: str = args.task_id
    task: Any = queue.task(args.task_id)
    metadata = task["metadata"]

    logger.info("")
    logger.info("Task:")
    logger.info(" name: " + task["metadata"]["name"])
    logger.info(f" url: https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}")
    logger.info(" description:")
    formatted = textwrap.fill(task["metadata"]["description"], width=100)
    for line in formatted.splitlines():
        logger.info(f"    > {line}")
    logger.info("")

    if args.checkout:
        logger.info("\n🔍 Checking repo state...")
        assert_clean_git()
        assert_remote_matches(metadata["source"])
        commit_sha = metadata["source"].split("/")[-2]
        git_reset(commit_sha)

    logger.info("")
    work_dir = run_task(task_id, task)
    print_tree(work_dir)


if __name__ == "__main__":
    main()
