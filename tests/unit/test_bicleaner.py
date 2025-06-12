import os
import tarfile
from subprocess import CompletedProcess

import pytest
import yaml
from tests.fixtures import DataDir
from pytest import fixture

from pipeline.bicleaner import download_pack
from pipeline.bicleaner.download_pack import main as download_model
from pipeline.common.datasets import decompress


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_bicleaner")


@fixture(scope="session")
def init():
    def _fake_download(src, trg, dir):
        pair = f"{src}-{trg}"
        if pair not in ["en-pt", "en-xx"]:
            return CompletedProcess(
                [], returncode=1, stderr=b"Error: language pack does not exist."
            )

        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, "metadata.yaml"), "w") as f:
            f.writelines([f"source_lang: {src}", "\n", f"target_lang: {trg}"])

        return CompletedProcess([], returncode=0)

    download_pack._run_download = _fake_download


def decompress_tar(path):
    tar_path = decompress(path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(os.path.dirname(path))


@pytest.mark.parametrize(
    "src,trg,model_src,model_trg",
    [
        ("en", "pt", "en", "pt"),
        ("pt", "en", "en", "pt"),
        ("ru", "en", "en", "xx"),
        ("en", "ru", "en", "xx"),
    ],
)
def test_model_download(src, trg, model_src, model_trg, init, data_dir):
    target_path = data_dir.join(f"bicleaner-ai-{src}-{trg}.tar.zst")
    decompressed_path = data_dir.join(f"bicleaner-ai-{src}-{trg}")
    meta_path = os.path.join(decompressed_path, "metadata.yaml")

    download_model([f"--src={src}", f"--trg={trg}", target_path])

    assert os.path.isfile(target_path)
    decompress_tar(target_path)
    assert os.path.isdir(decompressed_path)
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    assert metadata["source_lang"] == model_src
    assert metadata["target_lang"] == model_trg
