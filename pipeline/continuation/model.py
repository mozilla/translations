"""
Continue the training pipeline with an existing model. This produces the `continue-model`
task, which is dynamically used (at taskgraph generation time) as a dependency rather
than a `train`

TODO(#1151) - This only support backwards models with the "use" mode. We may be able
to unify everything here.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Optional
from pipeline.common.downloads import stream_download_to_file, location_exists
from pipeline.common.logging import get_logger
from pipeline.common import arg_utils

logger = get_logger(__file__)

potential_models = [
    "final.model.npz.best-chrf.npz",
    "model.npz.best-chrf.npz",
    "model.npz",
    "model.npz.best-bleu-detok.npz",
    "model.npz.best-ce-mean-words.npz",
    "model.npz.optimizer.npz",
]

potential_decoders = [
    "final.model.npz.best-chrf.npz.decoder.yml",
    "model.npz.best-chrf.npz.decoder.yml",
    "model.npz.best-bleu-detok.npz.decoder.yml",
    "model.npz.best-ce-mean-words.npz.decoder.yml",
    "model.npz.decoder.yml",
    "model.npz.progress.yml",
    "model.npz.yml",
]


def get_model_urls(url_prefix: str) -> tuple[str, str]:
    model: Optional[str] = None
    for potential_model in potential_models:
        url = url_prefix + potential_model
        logger.info(f"Checking to see if a model exists: {url}")
        if location_exists(url):
            model = url
            break
    assert model, "Could not find the model"

    decoder: Optional[str] = None
    for potential_decoder in potential_decoders:
        url = url_prefix + potential_decoder
        logger.info(f"Checking to see if a decoder.yml exists: {url}")
        if location_exists(url):
            decoder = url
            break
    assert decoder, "Could not find the decoder.yml"

    return model, decoder


def get_vocab_urls(
    url_prefix: str,
    src_locale: str,
    trg_locale: str,
    config_src_vocab_url: Optional[str],
    config_trg_vocab_url: Optional[str],
) -> tuple[str, str]:
    """
    Get the vocab URLs. Prefer the vocab near the model. Normalize the handling of
    shared and split vocabs so that there are is always a "vocab.{src_locale}.spm"
    and "vocab.{trg_locale}.spm" file. These files will be different with a split vocab
    and the same for shared vocabs.
    """
    shared_vocab_url = url_prefix + "vocab.spm"
    src_vocab_url = url_prefix + f"vocab.{src_locale}.spm"
    trg_vocab_url = url_prefix + f"vocab.{trg_locale}.spm"

    if location_exists(src_vocab_url) and location_exists(trg_vocab_url):
        # Prefer the vocabs next to the model, these are split.
        logger.info(f"A split vocab was found: {url_prefix}vocab.[locale].spm")
    elif location_exists(shared_vocab_url):
        # Prefer the vocabs next to the model, these are a shared vocab.
        logger.info(f"A single vocab was found: {url_prefix}vocab.spm")
        src_vocab_url = shared_vocab_url
        trg_vocab_url = shared_vocab_url
    else:
        # Fallback to the vocab provided by the config.
        assert config_src_vocab_url, "A src vocab was not provided."
        assert config_trg_vocab_url, "A trg vocab was not provided."
        assert location_exists(config_src_vocab_url), "The config's src vocab could not be found."
        assert location_exists(config_trg_vocab_url), "The config's trg vocab could not be found."

        src_vocab_url = config_src_vocab_url
        trg_vocab_url = config_trg_vocab_url

    return src_vocab_url, trg_vocab_url


def sha256sum(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src_locale", type=str, required=True, help="The source language for this model"
    )
    parser.add_argument(
        "--trg_locale", type=str, required=True, help="The target language for this model"
    )
    parser.add_argument("--url_prefix", type=str, required=True, help="The prefix for the URLs")
    parser.add_argument("--vocab_src", type=str, help="The source vocab file")
    parser.add_argument(
        "--vocab_trg", type=str, help="The target vocab file, potentially the same as the source"
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="The metric used to determine the best model, e.g. chrf, bleu, etc.",
    )
    parser.add_argument(
        "--artifacts", type=Path, help="The location where the models will be saved"
    )

    args = parser.parse_args()

    src_locale = arg_utils.ensure_string("--src_locale", args.src_locale)
    trg_locale = arg_utils.ensure_string("--trg_locale", args.trg_locale)
    url_prefix = arg_utils.ensure_string("--url_prefix", args.url_prefix)
    best_model = arg_utils.ensure_string("--best_model", args.best_model)
    config_src_vocab_url = arg_utils.handle_none_value(args.vocab_src)
    config_trg_vocab_url = arg_utils.handle_none_value(args.vocab_trg)
    artifacts: Path = args.artifacts

    assert artifacts
    artifacts.mkdir(exist_ok=True)

    if not url_prefix.endswith("/"):
        url_prefix = f"{url_prefix}/"

    # Download the model and decoder.yml:

    model_url, decoder_url = get_model_urls(url_prefix)

    model_out = artifacts / f"final.model.npz.best-{best_model}.npz"
    logger.info(f"Downloading model to: {model_out}")
    stream_download_to_file(model_url, model_out)

    decoder_out = artifacts / f"final.model.npz.best-{best_model}.npz.decoder.yml"
    logger.info(f"Downloading decoder to: {decoder_out}")
    stream_download_to_file(decoder_url, decoder_out)

    src_vocab_url, trg_vocab_url = get_vocab_urls(
        url_prefix=url_prefix,
        src_locale=src_locale,
        trg_locale=trg_locale,
        config_src_vocab_url=config_src_vocab_url,
        config_trg_vocab_url=config_trg_vocab_url,
    )

    # Download the vocab:

    src_vocab = artifacts / f"vocab.{src_locale}.spm"
    trg_vocab = artifacts / f"vocab.{trg_locale}.spm"
    stream_download_to_file(src_vocab_url, src_vocab)
    stream_download_to_file(trg_vocab_url, trg_vocab)

    if (
        config_src_vocab_url
        and config_trg_vocab_url
        and (src_vocab_url != config_src_vocab_url or trg_vocab_url != config_trg_vocab_url)
    ):
        # Double check that the vocab near the model is the same as the config's vocab.
        config_src_vocab = artifacts / f"config-vocab.{src_locale}.spm"
        config_trg_vocab = artifacts / f"config-vocab.{trg_locale}.spm"

        stream_download_to_file(config_src_vocab_url, config_src_vocab)
        stream_download_to_file(config_trg_vocab_url, config_trg_vocab)

        assert sha256sum(src_vocab) == sha256sum(
            config_src_vocab
        ), "The vocab src must match between the model directory and the config"
        assert sha256sum(trg_vocab) == sha256sum(
            config_trg_vocab
        ), "The vocab src must match between the model directory and the config"


if __name__ == "__main__":
    main()
