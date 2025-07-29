"""
Parallel (bilingual) translation dataset downloaders for various external resources like OPUS, mtdata etc.
"""
import shutil
import subprocess
import tarfile
import time
from enum import Enum
from pathlib import Path
import zipfile


from pipeline.common.command_runner import run_command
from pipeline.common.downloads import stream_download_to_file, compress_file, DownloadException
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class Downloader(Enum):
    opus = "opus"
    mtdata = "mtdata"
    sacrebleu = "sacrebleu"
    flores = "flores"
    url = "url"
    tmx = "tmx"


def opus(src: str, trg: str, dataset: str, output_prefix: Path):
    """
    Download a dataset from OPUS

    https://opus.nlpl.eu/
    """
    logger.info("Downloading opus corpus")

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
    logger.info("Done: Downloading opus corpus")


def mtdata(src: str, trg: str, dataset: str, output_prefix: Path):
    """
    Download a dataset using MTData

    https://github.com/thammegowda/mtdata
    """
    logger.info("Downloading mtdata corpus")

    from mtdata.iso import iso3_code

    tmp_dir = output_prefix.parent / "mtdata" / dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n = 3
    while True:
        try:
            run_command(
                ["mtdata", "get", "-l", f"{src}-{trg}", "-tr", dataset, "-o", str(tmp_dir)]
            )
            break
        except Exception as ex:
            logger.warn(f"Error while downloading mtdata corpus: {ex}")
            if n == 1:
                logger.error("Exceeded the number of retries, downloading failed")
                raise
            n -= 1
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)
            continue

    for file in tmp_dir.rglob("*"):
        logger.info(file)

    # some dataset names include BCP-47 country codes, e.g. OPUS-gnome-v1-eng-zho_CN
    src_suffix = None
    trg_suffix = None
    iso_src = iso3_code(src, fail_error=True)
    iso_trg = iso3_code(trg, fail_error=True)
    parts = dataset.split("-")
    code1, code2 = parts[-1], parts[-2]
    # make sure iso369 code matches the beginning of the mtdata langauge code (e.g. zho and zho_CN)
    if code1.startswith(iso_src) and code2.startswith(iso_trg):
        src_suffix = code1
        trg_suffix = code2
    elif code2.startswith(iso_src) and code1.startswith(iso_trg):
        src_suffix = code2
        trg_suffix = code1
    else:
        ValueError(f"Languages codes {code1}-{code2} do not match {iso_src}-{iso_trg}")

    for lang, suffix in ((src, src_suffix), (trg, trg_suffix)):
        file = tmp_dir / "train-parts" / f"{dataset}.{suffix}"
        compressed_path = compress_file(file, keep_original=False, compression="zst")
        compressed_path.rename(output_prefix.with_suffix(f".{lang}.zst"))

    shutil.rmtree(tmp_dir)
    logger.info("Done: Downloading mtdata corpus")


def url(src: str, trg: str, url: str, output_prefix: Path):
    """
    Download a dataset using http url
    """
    logger.info("Downloading corpus from a url")
    for lang in (src, trg):
        file = url.replace("[LANG]", lang)
        dest = output_prefix.with_suffix(f".{lang}.zst")
        logger.info(f"{lang} destination:      {dest}")
        stream_download_to_file(file, dest)
    logger.info("Done: Downloading corpus from a url")


def sacrebleu(src: str, trg: str, dataset: str, output_prefix: Path):
    """
    Download an evaluation dataset using SacreBLEU

    https://github.com/mjpost/sacrebleu
    """
    logger.info("Downloading sacrebleu corpus")

    def try_download(src_lang, trg_lang):
        try:
            for lang, target in ((src, "src"), (trg, "ref")):
                output = str(
                    run_command(
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

    logger.info("Done: Downloading sacrebleu corpus")


def pontoon_handle_bcp(lang):
    if lang == "sv":
        return "sv-SE"
    if lang == "gu":
        return "gu-IN"
    if lang == "pa":
        return "pa-IN"
    if lang == "nn":
        return "nn-NO"
    if lang == "nb":
        return "nb-NO"
    if lang == "no":
        return "nb-NO"
    if lang == "ne":
        return "ne-NP"
    if lang == "hi":
        return "hi-IN"
    if lang == "hy":
        return "hy-AM"
    if lang == "ga":
        return "ga-IE"
    if lang == "bn":
        return "bn-IN"
    if lang == "zh":
        return "zh-CN"
    return lang

def tmx(src: str, trg: str, dataset: str, output_prefix: Path):
    """
    Download and extract TMX from a predefined URL
    """
    logger.info(f"Downloading and extracting TMX from a url")

    if dataset == "pontoon":
        if src == 'en':
            lang = trg
        elif trg == 'en':
            lang = src
        else:
            raise ValueError(f"One of the languages must be 'en', src: {src} trg: {trg}")
        # for an iso639-1 lang code, select one of BCP codes from pontoon
        lang = pontoon_handle_bcp(lang)
        dataset_url = f"https://pontoon.mozilla.org/translation-memory/{lang}.all-projects.tmx"
    else:
        raise ValueError(f"Dataset {dataset} url is not defined")

    tmx_path = output_prefix.parent / f"{src}-{trg}.tmx"
    src_path = output_prefix.with_suffix(f".{src}")
    trg_path = output_prefix.with_suffix(f".{trg}")
    # set a large timeout because pontoon takes time
    stream_download_to_file(dataset_url, tmx_path, timeout_sec=240.0)

    from mtdata.tmx import read_tmx
    with open(src_path, 'w') as src_file, open(trg_path, 'w') as trg_file:
        for src_seg, trg_seg in read_tmx(tmx_path, langs=(src,trg)):
            print(src_seg, file=src_file)
            print(trg_seg, file=trg_file)

    compress_file(src_path, keep_original=False, compression='zst')
    compress_file(trg_path, keep_original=False, compression='zst')
    tmx_path.unlink()
    logger.info(f"Done: Downloading and extracting TMX from a url")


def flores(src: str, trg: str, dataset: str, output_prefix: Path):
    """
    Download Flores 101 evaluation dataset

    https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md
    """

    def flores_code(lang_code):
        if lang_code in ["zh", "zh-Hans"]:
            return "zho_simpl"
        elif lang_code == "zh-Hant":
            return "zho_trad"
        else:
            # Import and resolve ISO3 code using mtdata
            from mtdata.iso import iso3_code

            return iso3_code(lang_code, fail_error=True)

    logger.info("Downloading flores corpus")
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
    logger.info("Done: Downloading flores corpus")


mapping = {
    Downloader.opus: opus,
    Downloader.sacrebleu: sacrebleu,
    Downloader.flores: flores,
    Downloader.url: url,
    Downloader.mtdata: mtdata,
    Downloader.tmx: tmx,
}


def download(
    downloader: Downloader, src: str, trg: str, dataset: str, output_prefix: Path
) -> None:
    """
    Download a parallel dataset using :downloader

    :param downloader: downloader type (opus, mtdata etc.)
    :param src: source language code
    :param trg: target language code
    :param dataset: unsanitized dataset name e.g. wikimedia/v20230407 (for OPUS)
    :param output_prefix: output files prefix.

    Outputs two compressed files <output_prefix>.<src|trg>.zst

    """
    logger.info(f"importer:      {downloader}")
    logger.info(f"src:           {src}")
    logger.info(f"trg:           {trg}")
    logger.info(f"dataset:       {dataset}")
    logger.info(f"output_prefix: {output_prefix}")

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    mapping[downloader](src, trg, dataset, output_prefix)
