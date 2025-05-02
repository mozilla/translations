import shutil
import subprocess
import tarfile
from enum import Enum
from pathlib import Path
import zipfile


from pipeline.common.command_runner import run_command
from pipeline.common.downloads import stream_download_to_file, compress_file, DownloadException
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class Importer(Enum):
    opus = "opus"
    mtdata = "mtdata"
    sacrebleu = "sacrebleu"
    flores = "flores"
    url = "url"


def opus(src: str, trg: str, dataset: str, output_prefix: Path):
    logger.info("###### Downloading opus corpus")
    logger.info("###### Downloading opus corpus")

    name = dataset.split("/")[0]
    name_and_version = "".join(c if c.isalnum() or c in "-_ " else "_" for c in dataset)

    tmp_dir = output_prefix.parent / "opus" / name_and_version
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / f"{name}.txt.zip"

    def download_opus(pair):
        url = f"https://object.pouta.csc.fi/OPUS-{dataset}/moses/{pair}.txt.zip"
        logger.info(f"Downloading corpus for {pair} {url} to {archive_path}")
        stream_download_to_file(url, archive_path)

    try:
        pair = f"{src}-{trg}"
        download_opus(pair)
    except DownloadException:
        logger.info("Downloading error, trying opposite direction")
        pair = f"{trg}-{src}"
        download_opus(pair)

    logger.info("Extracting directory")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    logger.info("Compressing output files")
    for lang in (src, trg):
        file_path = tmp_dir / f"{name}.{pair}.{lang}"
        compressed_path = compress_file(file_path, keep_original=False, compression="zst")
        output_path = output_prefix.with_suffix(f".{lang}.zst")
        compressed_path.rename(output_path)

    shutil.rmtree(tmp_dir)
    logger.info("###### Done: Downloading opus corpus")


def mtdata(src: str, trg: str, dataset: str, output_prefix: Path):
    logger.info("###### Downloading mtdata corpus")

    from mtdata.iso import iso3_code

    tmp_dir = output_prefix.parent / "mtdata" / dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_command(["mtdata", "get", "-l", f"{src}-{trg}", "-tr", dataset, "-o", str(tmp_dir)])

    for file in tmp_dir.rglob("*"):
        logger.info(file)

    for lang in (src, trg):
        iso = iso3_code(lang, fail_error=True)
        file = tmp_dir / "train-parts" / f"{dataset}.{iso}"
        compressed_path = compress_file(file, keep_original=False, compression="zst")
        compressed_path.rename(output_prefix.with_suffix(f".{lang}.zst"))

    shutil.rmtree(tmp_dir)
    logger.info("###### Done: Downloading mtdata corpus")


def url(src: str, trg: str, url: str, output_prefix: Path):
    logger.info("###### Downloading corpus from a url")
    for lang in (src, trg):
        file = url.replace("[LANG]", lang)
        dest = output_prefix.with_suffix(f".{lang}.zst")
        logger.info(f"{lang} destination:      {dest}")
        stream_download_to_file(file, dest)
    logger.info("###### Done: Downloading corpus from a url")


def sacrebleu(src: str, trg: str, dataset: str, output_prefix: Path):
    logger.info("###### Downloading sacrebleu corpus")

    def try_download(src_lang, trg_lang):
        try:
            for lang, target in ((src, "src"), (trg, "ref")):
                output = run_command(
                    [
                        "sacrebleu",
                        "--test-set",
                        dataset,
                        "--language-pair",
                        f"{src_lang}-{trg_lang}",
                        "--echo",
                        target,
                    ],
                    capture=True,
                )
                output_file = output_prefix.with_suffix(f".{lang}")
                with open(output_file, "w") as f:
                    f.write(output)
                compress_file(output_file, keep_original=False, compression="zst")

            return True
        except subprocess.CalledProcessError:
            return False

    # Try original direction
    success = try_download(src, trg)

    if not success:
        logger.info("The first import failed, try again by switching the language pair direction.")
        # Try reversed direction
        if not try_download(trg, src):
            raise RuntimeError("Both attempts to download the dataset failed.")

    logger.info("###### Done: Downloading sacrebleu corpus")


def flores(src: str, trg: str, dataset: str, output_prefix: Path):
    def flores_code(lang_code):
        if lang_code in ["zh", "zh-Hans"]:
            return "zho_simpl"
        elif lang_code == "zh-Hant":
            return "zho_trad"
        else:
            # Import and resolve ISO3 code using mtdata
            from mtdata.iso import iso3_code

            return iso3_code(lang_code, fail_error=True)

    logger.info("###### Downloading flores corpus")
    tmp_dir = output_prefix.parent / "flores" / dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / "flores101_dataset.tar.gz"
    dataset_url = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"

    stream_download_to_file(dataset_url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)

    for lang in (src, trg):
        code = flores_code(lang)
        file = tmp_dir / "flores101_dataset" / dataset / f"{code}.{dataset}"
        compressed_path = compress_file(file, keep_original=False, compression="zst")
        compressed_path.rename(output_prefix.with_suffix(f".{lang}.zst"))

    shutil.rmtree(tmp_dir)
    logger.info("###### Done: Downloading flores corpus")


mapping = {
    Importer.opus: opus,
    Importer.sacrebleu: sacrebleu,
    Importer.flores: flores,
    Importer.url: url,
    Importer.mtdata: mtdata,
}


def download(importer: Importer, src: str, trg: str, dataset: str, output_prefix: Path):
    logger.info(f"importer:      {importer}")
    logger.info(f"src:           {src}")
    logger.info(f"trg:           {trg}")
    logger.info(f"dataset:       {dataset}")
    logger.info(f"output_prefix: {output_prefix}")

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    mapping[importer](src, trg, dataset, output_prefix)
