from dataclasses import dataclass
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
import yaml
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import zstandard as zstd

from pipeline.common.downloads import read_lines
from utils.preflight_check import get_taskgraph_parameters, run_taskgraph

FIXTURES_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FIXTURES_PATH, "../.."))
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, "data"))
TESTS_DATA = os.path.join(DATA_PATH, "tests_data")


en_sample = """The little girl, seeing she had lost one of her pretty shoes, grew angry, and said to the Witch, “Give me back my shoe!”
“I will not,” retorted the Witch, “for it is now my shoe, and not yours.”
“You are a wicked creature!” cried Dorothy. “You have no right to take my shoe from me.”
“I shall keep it, just the same,” said the Witch, laughing at her, “and someday I shall get the other one from you, too.”
This made Dorothy so very angry that she picked up the bucket of water that stood near and dashed it over the Witch, wetting her from head to foot.
Instantly the wicked woman gave a loud cry of fear, and then, as Dorothy looked at her in wonder, the Witch began to shrink and fall away.
“See what you have done!” she screamed. “In a minute I shall melt away.”
“I’m very sorry, indeed,” said Dorothy, who was truly frightened to see the Witch actually melting away like brown sugar before her very eyes.
"""

ru_sample = """Маленькая девочка, увидев, что потеряла одну из своих красивых туфелек, рассердилась и сказала Ведьме: «Верни мне мою туфельку!»
«Я не буду, — парировала Ведьма, — потому что теперь это моя туфля, а не твоя».
«Ты злое существо!» - воскликнула Дороти. «Ты не имеешь права забирать у меня туфлю».
«Я все равно сохраню его, — сказала Ведьма, смеясь над ней, — и когда-нибудь я получу от тебя и другой».
Это так разозлило Дороти, что она взяла стоявшее рядом ведро с водой и облила им Ведьму, обмочив ее с головы до ног.
Мгновенно злая женщина громко вскрикнула от страха, а затем, когда Дороти с удивлением посмотрела на нее, Ведьма начала сжиматься и падать.
«Посмотри, что ты наделал!» она закричала. «Через минуту я растаю».
«Мне действительно очень жаль», — сказала Дороти, которая была по-настоящему напугана, увидев, что Ведьма тает, как коричневый сахар, у нее на глазах.
"""

zh_sample = """小女孩看到自己丢了一只漂亮的鞋子，生气了，对女巫说：“把我的鞋子还给我！”
“我不会的，”女巫反驳道，“因为现在是我的鞋子，不是你的。”
“你是个坏女人！”多萝西喊道。“你无权夺走我的鞋子。”
“我会把它留着的，”女巫笑着说，“总有一天我也会从你那里得到另一只。”
这让多萝西非常生气，她拿起旁边的一桶水，泼在女巫身上，把她从头到脚都淋湿了。
恶毒的女人立刻发出一声恐惧的尖叫，然后，当多萝西惊奇地看着她时，女巫开始缩小并倒下。
“看看你做了什么！”她尖叫道。“我马上就会融化。”
“我真的很抱歉，”多萝西说，她真的很害怕看到女巫真的像红糖一样在她眼前融化。
"""


class DataDir:
    """
    Creates a persistent data directory in data/tests_data/{dir_name} that will be
    cleaned out before a test run. This should help in persisting artifacts between test
    runs to manually verify the results.

    Taskcluster tasks can be run directly using the data dir via the run_task method.
    """

    def __init__(self, dir_name: str) -> None:
        self.path = os.path.join(TESTS_DATA, dir_name)

        # Ensure the base /data directory exists.
        os.makedirs(TESTS_DATA, exist_ok=True)

        # Clean up a previous run if this exists.
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

        os.makedirs(self.path)
        print("Tests are using the subdirectory:", self.path)

    def join(self, *paths: str):
        """Create a folder or file name by joining it to the test directory."""
        return os.path.join(self.path, *paths)

    def read_text(self, name: str):
        """Load the text from a file. It can be a txt file or a compressed file."""
        text = ""
        with read_lines(self.join(name)) as lines:
            for line in lines:
                text += line
        return text

    def create_zst(self, name: str, contents: str) -> str:
        """
        Creates a compressed zst file and returns the path to it.
        """
        zst_path = os.path.join(self.path, name)
        if not os.path.exists(self.path):
            raise Exception(f"Directory for the compressed file does not exist: {self.path}")
        if os.path.exists(zst_path):
            raise Exception(f"A file already exists and would be overwritten: {zst_path}")

        # Create the compressed file.
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(contents.encode("utf-8"))

        print("Writing a compressed file to: ", zst_path)
        with open(zst_path, "wb") as file:
            file.write(compressed_data)

        return zst_path

    def create_file(self, name: str, contents: Union[str, Iterable[str]]) -> str:
        """
        Creates a text file and returns the path to it.
        """
        if not isinstance(contents, str):
            contents = "\n".join(contents) + "\n"

        text_path = os.path.join(self.path, name)
        if not os.path.exists(self.path):
            raise Exception(f"Directory for the text file does not exist: {self.path}")
        if os.path.exists(text_path):
            raise Exception(f"A file already exists and would be overwritten: {text_path}")

        print("Writing a text file to: ", text_path)
        with open(text_path, "w") as file:
            file.write(contents)

        return text_path

    def mkdir(self, name) -> str:
        path = self.join(name)
        os.makedirs(path, exist_ok=True)
        return path

    def run_task(
        self,
        task_name: str,
        work_dir: Optional[str] = None,
        fetches_dir: Optional[str] = None,
        env: dict[str, str] = {},
        extra_flags: List[str] = None,
        extra_args: List[str] = None,
        replace_args: List[Tuple[str, str]] = None,
        config: Optional[str] = None,
    ):
        """
        Runs a task from the taskgraph. See artifacts/full-task-graph.json after running a
        test for the full list of task names

        Arguments:

        task_name - The full task name like "distillation-mono-src-chunk-en"
            or "evaluate-backward-sacrebleu-wmt09-en-ru".

        data_dir - The test's DataDir

        work_dir - This is the TASK_WORKDIR, in tests generally the test's DataDir.

        fetches_dir - The MOZ_FETCHES_DIR, generally set as the test's DataDir.

        env - Any environment variable overrides.

        extra_flags - Place extra flags in the command before the "--" that is used to apply
            marian args.

        extra_args - Place extra arguments at the end of the command.

        replace_args - A list of Tuples where an argument is replaced by word match.

        config - A path to a Taskcluster config file
        """

        command_parts, requirements, task_env = get_task_command_and_env(task_name, config=config)

        # There are some non-string environment variables that involve taskcluster references
        # Remove these.
        for key in task_env:
            if not isinstance(task_env[key], str):
                task_env[key] = ""

        current_folder = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.abspath(os.path.join(current_folder, "../.."))
        env.setdefault("PYTHONPATH", root_path)

        if not work_dir:
            work_dir = self.path
        if not fetches_dir:
            fetches_dir = self.path

        for command_parts_split in split_on_ampersands_operator(command_parts):
            if extra_flags:
                index = command_parts_split.index("--")
                command_parts_split = [  # noqa: PLW2901
                    *command_parts_split[:index],
                    *extra_flags,
                    *command_parts_split[index:],
                ]
            if extra_args:
                command_parts_split.extend(extra_args)

            if replace_args:
                for arg_from, arg_to in replace_args:
                    for index, arg in enumerate(command_parts_split):
                        if arg == arg_from:
                            command_parts_split[index] = arg_to

            final_env = {
                # The following are set by the Taskcluster server.
                "TASK_ID": "fake_id",
                "RUN_ID": "0",
                "TASKCLUSTER_ROOT_URL": "https://some.cluster",
                **os.environ,
                **task_env,
                "TASK_WORKDIR": work_dir,
                "MOZ_FETCHES_DIR": fetches_dir,
                "VCS_PATH": root_path,
                **env,
            }

            # Expand out environment variables in environment, for instance MARIAN=$MOZ_FETCHES_DIR
            # and FETCHES=./fetches will be expanded to MARIAN=./fetches
            for key, value in final_env.items():
                if not isinstance(value, str):
                    continue
                expanded_value = final_env.get(value[1:])
                if value and value[0] == "$" and expanded_value:
                    final_env[key] = expanded_value

            # Ensure the environment variables are sorted so that the longer variables get replaced first.
            sorted_env = sorted(final_env.items(), key=lambda kv: kv[0])
            sorted_env.reverse()

            for index, p in enumerate(command_parts_split):
                part = (
                    p.replace("$TASK_WORKDIR/$VCS_PATH", root_path)
                    .replace("$VCS_PATH", root_path)
                    .replace("$TASK_WORKDIR", work_dir)
                    .replace("$MOZ_FETCHES_DIR", fetches_dir)
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
                    final_env = {
                        **final_env,
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
                    print(f"Setting LD_LIBRARY_PATH to {lib_path}")
                    final_env["LD_LIBRARY_PATH"] = lib_path

            print("┌──────────────────────────────────────────────────────────")
            print("│ run_task:", " ".join(command_parts_split))
            print("└──────────────────────────────────────────────────────────")

            result = subprocess.run(
                command_parts_split,
                env=final_env,
                cwd=root_path,
                check=False,
            )

            fail_on_error(result)

    def rewrite_ci_config(self, rewrite_config: Callable[[dict[str, Any]], None]) -> str:
        """
        Take the config.ci.yml file, and rewrite the deserialized python values for a
        specific test. Returns the path to the config.
        """
        config_path = (Path(__file__).parent / "../../taskcluster/configs/config.ci.yml").resolve()
        with config_path.open() as file:
            config = yaml.safe_load(file)
        rewrite_config(config)
        return self.create_file("config.yml", yaml.dump(config))

    def print_tree(self):
        """
        Print a tree view of the data directory, which is useful for debugging test failures.
        """
        span_len = 90
        span = "─" * span_len
        print(f"┌{span}┐")

        for root, dirs, files in os.walk(self.path):
            level = root.replace(self.path, "").count(os.sep)
            indent = " " * 4 * (level)
            if level == 0:
                # For the root level, display the relative path to the data directory.
                folder_text = root.replace(f"{ROOT_PATH}/", "")
                folder_text = f"│ {folder_text}"
            else:
                folder_text = f"│ {indent}{os.path.basename(root)}/"
            print(f"{folder_text.ljust(span_len)} │")
            subindent = " " * 4 * (level + 1)

            if len(files) == 0 and len(dirs) == 0:
                empty_text = f"│ {subindent} <empty folder>"
                print(f"{empty_text.ljust(span_len)} │")
            for file in files:
                file_text = f"│ {subindent}{file}"
                bytes = f"{os.stat(os.path.join(root, file)).st_size} bytes"

                print(f"{file_text.ljust(span_len - len(bytes))}{bytes} │")

        print(f"└{span}┘")

    def assert_files(self, expected: list[str]) -> None:
        """
        Assert that the data_dir contains exactly the files listed in `expected`,
        including files in subdirectories, relative to the data_dir.
        """
        expected_set = set(os.path.normpath(p) for p in expected)

        actual_set: set[str] = set()
        for root, _, files in os.walk(self.path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, self.path)
                actual_set.add(os.path.normpath(relative_path))

        if expected_set != actual_set:
            missing = expected_set - actual_set
            unexpected = actual_set - expected_set
            messages = []
            if missing:
                messages.append(f"Missing files: {sorted(missing)}")
            if unexpected:
                messages.append(f"Unexpected files: {sorted(unexpected)}")

            raise AssertionError("\n".join(messages))


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


def fail_on_error(result: CompletedProcess[bytes]):
    """When a process fails, surface the stderr."""
    if not result.returncode == 0:
        raise Exception(f"{result.args[0]} exited with a status code: {result.returncode}")


@dataclass
class TaskgraphFiles:
    # Task label to task description. This is the full task graph for all of the kinds.
    # artifacts/full-task-graph.json
    # { "corpus-merge-parallel-ru-en": TaskDescription, ... }
    full: dict[str, dict[str, Any]]

    # Task id to task description. These are just the tasks that are resolved.
    # artifacts/task-graph.json
    # { "AnmIFVMrT0OoS7UdB4mQGQ": TaskDescription, ... }
    resolved: dict[str, dict[str, Any]]

    @staticmethod
    def from_config(config: str) -> "TaskgraphFiles":
        start = time.time()
        artifacts = Path(__file__).parent / "../../artifacts"

        if os.environ.get("SKIP_TASKGRAPH"):
            print("Using existing taskgraph generation.")
        else:
            print(
                f"Generating the full taskgraph with config {config}, this can take a second. Set SKIP_TASKGRAPH=1 to skip this step."
            )
            run_taskgraph(config, get_taskgraph_parameters())

        with (artifacts / "full-task-graph.json").open() as file:
            full = json.load(file)
        with (artifacts / "task-graph.json").open() as file:
            resolved = json.load(file)

        elapsed_sec = time.time() - start
        print(f"Taskgraph generated in {elapsed_sec:.2f} seconds.")
        return TaskgraphFiles(full=full, resolved=resolved)


# Cache the taskgraphs as it's quite slow to generate them.
_full_taskgraph_cache: dict[str, TaskgraphFiles] = {}


def get_taskgraph_files(config: Optional[str] = None) -> TaskgraphFiles:
    """
    Generates the full taskgraph and stores it for re-use. It uses the config.pytest.yml
    in this directory.

    config - A path to a Taskcluster config
    """
    if not config:
        config = str((Path(__file__).parent / "config.pytest.yml").resolve())

    if config in _full_taskgraph_cache:
        return _full_taskgraph_cache[config]

    taskgraph_files = TaskgraphFiles.from_config(config)
    _full_taskgraph_cache[config] = taskgraph_files
    return taskgraph_files


# Taskcluster commands can either be a single list of commands, or a nested list.
Commands = Union[list[str], list[list[str]]]


def get_command(commands: Commands) -> str:
    if isinstance(commands[-1], str):
        # Non-nested command, get the last string.
        return commands[-1]

    if isinstance(commands[-1][-1], str):
        # Nested command, get the last string of the last command.
        return commands[-1][-1]

    print(commands)
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
    #   pipeline/data/parallel_importer.py
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
    task_name: str, config: Optional[str]
) -> tuple[list[str], Optional[str], dict[str, str]]:
    """
    Extracts a task's command from the full taskgraph. This allows for testing
    the full taskcluster pipeline and the scripts that it generates.
    See artifacts/full-task-graph.json for the full list of what is generated.

    task_name - The full task name like "distillation-mono-src-chunk-en"
        or "evaluate-backward-sacrebleu-wmt09-en-ru".

    config - A path to a Taskcluster config
    """
    full_taskgraph = get_taskgraph_files(config).full
    task = full_taskgraph.get(task_name)
    if not task:
        print("Available tasks:")
        for key in full_taskgraph.keys():
            print(f' - "{key}"')
        raise Exception(f"Could not find the task {task_name}")

    env = task["task"]["payload"]["env"]

    commands = task["task"]["payload"]["command"]
    pipeline_script = find_pipeline_script(commands)
    requirements = find_requirements(commands)

    print(f'Running: "{task_name}":')
    print(" > Commands:", commands)
    print(" > Running:", pipeline_script)
    print(" > Env:", env)
    print(" > Requirements:", requirements)

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


def get_mocked_downloads() -> str:
    corpus_samples = os.path.abspath(os.path.join(FIXTURES_PATH, "../data/corpus_samples"))

    def get_path(name: str):
        return os.path.join(corpus_samples, name)

    return json.dumps(
        {
            "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz":
                get_path("flores101_dataset.tar.gz"),
            "https://object.pouta.csc.fi/OPUS-ELRC-3075-wikipedia_health/v1/moses/en-ru.txt.zip":
                get_path("en-ru.txt.zip"),
            "https://object.pouta.csc.fi/OPUS-ELRC_2922/v1/moses/en-ru.txt.zip":
                get_path("en-ru2.txt.zip"),
            "https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/en-zh.txt.zip":
                get_path("en-zh.txt.zip"),
            "https://object.pouta.csc.fi/OPUS-ELRC-3075-wikipedia_health/v1/moses/ru-en.txt.zip":
                "404",
            "http://data.statmt.org/news-crawl/en/news.2021.en.shuffled.deduped.gz":
                get_path("pytest-dataset.en.gz"),
            "http://data.statmt.org/news-crawl/ru/news.2021.ru.shuffled.deduped.gz":
                get_path("pytest-dataset.ru.gz"),
            "http://data.statmt.org/news-crawl/zh/news.2021.zh.shuffled.deduped.gz":
                get_path("pytest-dataset.zh.gz"),
            "https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.en.zst":
                get_path("pytest-dataset.en.zst"),
            "https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.ru.zst":
                get_path("pytest-dataset.ru.zst"),
            "https://data.hplt-project.org/one/monotext/cleaned/ru/ru_10.jsonl.zst":
                get_path("hplt-ru_10.jsonl.zst"),
            "https://data.hplt-project.org/one/monotext/cleaned/ru/ru_11.jsonl.zst":
                get_path("hplt-ru_11.jsonl.zst"),
            "https://data.hplt-project.org/one/monotext/cleaned/en/en_100.jsonl.zst":
                get_path("hplt-en_100.jsonl.zst"),
            "https://data.hplt-project.org/one/monotext/cleaned/en/en_101.jsonl.zst":
                get_path("hplt-en_101.jsonl.zst"),
            "https://data.hplt-project.org/two/cleaned/rus_Cyrl_map.txt":
                get_path("hplt-rus_Cyrl.txt"),
            "https://data.hplt-project.org/two/cleaned/eng_Latn_map.txt":
                get_path("hplt-eng_Latn_map.txt"),
        }
    )  # fmt: skip


def get_python_dirs(requirements: str) -> Tuple[str, str]:
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
            print("Creating virtual environment")
            subprocess.check_call(
                # Give the virtual environment access to the system site packages, as these
                # are installed via docker.
                ["python", "-m", "venv", "--system-site-packages", venv_dir],
            )

            print("Installing setuptools", requirements)
            subprocess.check_call(
                [python_bin, "-m", "pip", "install", "--upgrade", "setuptools", "pip"],
            )

            print("Installing", requirements)
            subprocess.check_call(
                [python_bin, "-m", "pip", "install", "-r", requirements],
            )
        except Exception as exception:
            print("Removing the venv due to an error in its creation.")
            shutil.rmtree(venv_dir)
            raise exception
    print(f"Using virtual environment {venv_dir}")

    return python_bin_dir, venv_dir


def hash_file(hash: any, path: str):
    """
    Hash the contents of a file.
    """
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            hash.update(chunk)
