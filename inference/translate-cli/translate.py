"""
Run models using a local translator built as similar to the Wasm as possible.
"""

from packaging.version import parse
from pathlib import Path
from pipeline.common.logging import get_logger
from pipeline.common.command_runner import run_command_pipeline
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


def get_records():
    url = get_prod_records_url("translations-models")
    response = requests.get(url)
    response.raise_for_status()
    models = ModelsResponse(**response.json())
    records = models.get_max_versions()
    models = list(filter(lambda model: model.fileType == "model", records))
    return records


def get_model(langpair: str) -> Path:
    """
    Get a model from a specific langpair. It will look for it locally on disk first, then
    look in remote settings and download it.
    """
    config_yml = get_downloaded_model_config(langpair)
    if config_yml:
        return config_yml

    all_records = get_records()
    langpair_records = list(filter(lambda record: record.langpair() == langpair, all_records))

    if not langpair_records:
        logger.error("\nCould not find the language pair: " + langpair)
        sys.exit(1)

    return download_records(langpair_records)


def prompt_for_model() -> Path:
    """
    Get the model records from Remote Settings and prompt for a language pair. If the model
    is already downloaded it will be used, if not the model will be downloaded.
    """
    records = get_records()

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

    # Prompt for a language pair:
    while True:
        langpair = input("Choose a language pair:")
        langpair_records = list(filter(lambda record: record.langpair() == langpair, records))
        if langpair_records:
            config_yml = configs_by_langpair.get(langpair)
            if config_yml:
                return config_yml
            return download_records(langpair_records)

        logger.info("That language pair was not in the list, try again.")


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
    Naively runs the translator by passing a single string. This reloads the translation
    instance for every translation.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def translate(self, sentence: str) -> str:
        binary = ROOT_PATH / "inference/build/translate"
        assert binary.exists(), (
            "The translate binary did not exist. It needs to be built with:\n"
            "task inference-build -- --build_cli"
        )

        return (
            run_command_pipeline(
                [
                    ["echo", sentence],
                    [
                        str(binary),
                        "--model-config-paths",
                        str(self.config_path),
                        "--log-level",
                        "error",
                    ],
                ],
                capture=True,
            )
            or ""
        )


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

    if langpair:
        config_yml = get_model(langpair)
    else:
        config_yml = prompt_for_model()

    logger.info(f"Using config: {config_yml.relative_to(ROOT_PATH)}")

    translator = Translator(config_yml)
    while True:
        output = translator.translate(input("Type in text to translate: "))
        print("Translation:", output)


if __name__ == "__main__":
    main()
