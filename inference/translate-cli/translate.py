"""
Run models using a local translator built as similar to the Wasm as possible.
"""

from packaging.version import parse
from pathlib import Path
from pipeline.common.logging import get_logger
from typing import TextIO
from utils.common.remote_settings import ModelsResponse, ModelRecord, get_prod_records_url
import argparse
import json
import requests
import subprocess
import sys
import yaml

ROOT_PATH = Path(__file__).parent.parent.parent.resolve()
MODEL_FILES = ROOT_PATH / "data/models"

logger = get_logger(__file__)


def request_model_records(langpair_arg: str | None) -> list[ModelRecord] | Path:
    url = get_prod_records_url("translations-models")
    response = requests.get(url)
    response.raise_for_status()
    models = ModelsResponse(**response.json())

    records = models.get_max_versions()

    if langpair_arg:
        model_records = list(filter(lambda record: record.langpair() == langpair_arg, records))
        if model_records:
            return model_records

    # Print a list of all of the models, and cache if the configs were already downloaded for it.
    models = list(filter(lambda model: model.fileType == "model", records))
    configs_by_langpair: dict[str, Path] = {}

    for model in models:
        model_dir = MODEL_FILES / f"{model.langpair()}-{model.version}"
        config = model_dir / "config.yml"
        if config.exists():
            configs_by_langpair[model.langpair()] = config
            print(" -", model.langpair(), "[downloaded]")
        else:
            print(" -", model.langpair())

    if langpair_arg:
        # Since a langpair was provided via the CLI, this is now an error if no model was found.
        logger.error("\nCould not find the language pair: " + langpair_arg)
        sys.exit(1)

    # No language pair was provided, so prompt for one.
    while True:
        langpair = input("Choose a language pair:")
        model_records = list(filter(lambda record: record.langpair() == langpair, records))
        if model_records:
            # Either return the records, or the path to the downloaded config.
            return configs_by_langpair.get(langpair, model_records)

        logger.info("That language pair was not, try again.")


def get_downloaded_model_config(langpair: str) -> Path | None:
    """
    Look for a config that has already been downloaded.
    """
    folders = list(MODEL_FILES.glob(f"{langpair}-*"))
    max_version = None
    model_folder = None
    for folder in folders:
        version = parse(folder.name.split("-")[-1])
        if not max_version or version > max_version:
            max_version = version
            model_folder = folder

    if not model_folder:
        return None

    config_yml = model_folder / "config.yml"

    if config_yml.exists():
        return config_yml

    return None


def download_records(records: list[ModelRecord]) -> Path:
    model = next(filter(lambda record: record.fileType == "model", records), None)
    lex = next(filter(lambda record: record.fileType == "lex", records), None)
    vocabs = list(filter(lambda record: record.fileType == "vocab", records))

    assert model, "The model file was not found."
    assert lex, "The lex file was not found."
    assert len(vocabs) in {1, 2}, f"Expected 1 or 2 vocab files, instead there were {len(vocabs)}"

    model_dir = MODEL_FILES / f"{model.langpair()}-{model.version}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Download all of the attachments.
    for record in records:
        attachment = record.attachment
        assert attachment, "Attachment was not found."
        assert "/" not in record.name, f"Record name changes locations: {record.name}"

        attachment_file_path = model_dir / f"{record.name}"
        if not attachment_file_path.exists():
            download_url = (
                f"https://firefox-settings-attachments.cdn.mozilla.net/{attachment.location}"
            )
            with attachment_file_path.open("wb") as attachment_file:
                logger.info(f"⬇️ Downloading {record.name} {record.version} from {download_url}")
                response = requests.get(download_url, stream=True, allow_redirects=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    attachment_file.write(chunk)

    # The records.json isn't strictly necessary to write out, but it's useful to know the
    # provenance of the models.
    with (model_dir / "records.json").open("wt") as outfile:
        data = {
            "model": model.json(),
            "lex": lex.json(),
            "vocabs": [vocab.json() for vocab in vocabs],
        }
        json.dump(data, outfile, indent=2, sort_keys=True)

    config_vocabs = [vocab.name for vocab in vocabs]
    if len(config_vocabs) == 1:
        # Two vocabs are still required when it's shared.
        config_vocabs.append(config_vocabs[0])

    # The config is what actually is used for translating.
    # https://marian-nmt.github.io/docs/cmd/marian-decoder/
    config = {
        # Paths are relative to this config.
        "relative-paths": True,
        "models": [model.name],
        "vocabs": config_vocabs,
        "beam-size": 1,
        "normalize": 1.0,
        "word-penalty": 0,
        "max-length-break": 128,
        "mini-batch-words": 1024,
        "workspace": 128,
        "max-length-factor": 2.0,
        "skip-cost": True,
        "cpu-threads": 0,
        "quiet": False,
        "quiet-translation": False,
        "gemm-precision": "int8shiftAlphaAll",
        "alignment": "soft",
    }

    config_path = model_dir / "config.yml"
    with config_path.open("w") as outfile:
        yaml.safe_dump(config, outfile)

    return config_path


class Translator:
    """
    Manages a subprocess for streaming translations via stdin/stdout.
    Use as a context manager.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.proc: subprocess.Popen | None = None
        self.stdin: TextIO | None = None
        self.stdout: TextIO | None = None

    def __enter__(self) -> "Translator":
        binary = ROOT_PATH / "inference/build/translate"
        assert binary.exists(), (
            "The translate binary did not exist. It needs to be built with:\n"
            "task inference-build -- --build_cli"
        )

        self.proc = subprocess.Popen(
            [
                str(binary),
                "--model-config-paths",
                str(self.config_path),
                "--log-level",
                "trace",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Output marian logging.
            text=True,
            bufsize=1,  # line-buffered
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.stdin = self.proc.stdin  # type: ignore
        self.stdout = self.proc.stdout  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.stdin:
            self.stdin.close()
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def translate(self, sentence: str) -> str:
        if not self.stdin or not self.stdout:
            raise RuntimeError("Translator process is not running")

        self.stdin.write(sentence + "\n")
        self.stdin.flush()
        return self.stdout.readline().strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--langpair",
        type=str,
        help="The language pair to translate. If none is provided, the user will be queried for one.",
    )

    args = parser.parse_args()
    langpair: str | None = args.langpair
    config_yml: Path | None = None

    if langpair:
        config_yml = get_downloaded_model_config(langpair)

    if not config_yml:
        request_response = request_model_records(langpair)
        if isinstance(request_response, Path):
            config_yml = request_response
        else:
            config_yml = download_records(request_response)

    logger.info(f"Using config: {config_yml.relative_to(ROOT_PATH)}")

    with Translator(config_yml) as translator:
        while True:
            output = translator.translate(input("Type in text to translate:"))
            print("Translation:", output)


if __name__ == "__main__":
    main()
