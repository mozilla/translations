"""
Continue the training pipeline with existing corpora.
"""
import argparse
from enum import Enum
from pathlib import Path
from pipeline.common.downloads import stream_download_to_file
from pipeline.common.logging import get_logger
from pipeline.common import arg_utils

logger = get_logger(__file__)


class Corpus(Enum):
    backtranslations = "backtranslations"
    parallel = "parallel"
    distillation = "distillation"

    def __str__(self):
        # Support for argparse choices.
        return self.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src_locale", type=str, required=True, help="The source language for this corpus"
    )
    parser.add_argument(
        "--trg_locale", type=str, required=True, help="The target language for this file"
    )
    parser.add_argument("--corpus", type=Corpus, required=True, choices=list(Corpus))
    parser.add_argument(
        "--src_url", type=str, required=True, help="The source URL for this corpus"
    )
    parser.add_argument(
        "--trg_url", type=str, required=True, help="The target URL for this corpus"
    )
    parser.add_argument(
        "--tok_src_url",
        type=str,
        default="",
        help="The (optional) source URL for the tokenized corpus",
    )
    parser.add_argument(
        "--tok_trg_url",
        type=str,
        default="",
        help="The (optional) target URL for the tokenized corpus",
    )
    parser.add_argument(
        "--alignments_url",
        type=str,
        default="",
        help="The (optional) alignments file URL for this file",
    )
    parser.add_argument(
        "--artifacts", type=Path, help="The location where the dataset will be saved"
    )

    args = parser.parse_args()

    src_locale = arg_utils.ensure_string("--src_locale", args.src_locale)
    trg_locale = arg_utils.ensure_string("--trg_locale", args.trg_locale)
    corpus: Corpus = args.corpus
    src_url = arg_utils.ensure_string("--src_url", args.src_url)
    trg_url = arg_utils.ensure_string("--trg_url", args.trg_url)
    tok_src_url = arg_utils.handle_none_value(args.tok_src_url)
    tok_trg_url = arg_utils.handle_none_value(args.tok_trg_url)
    alignments_url = arg_utils.handle_none_value(args.alignments_url)

    artifacts: Path = args.artifacts

    if corpus == Corpus.backtranslations:
        file_name_part = "mono"
    elif corpus == Corpus.parallel:
        file_name_part = "corpus"
    elif corpus == Corpus.distillation:
        file_name_part = "corpus"
    else:
        raise ValueError(f'Unexpected corpus name: "{corpus}"')

    artifacts.mkdir(exist_ok=True)
    src_destination = artifacts / f"{file_name_part}.{src_locale}.zst"
    trg_destination = artifacts / f"{file_name_part}.{trg_locale}.zst"

    stream_download_to_file(src_url, src_destination)
    stream_download_to_file(trg_url, trg_destination)

    if tok_src_url or tok_trg_url or alignments_url:
        assert (
            tok_src_url and tok_trg_url and alignments_url
        ), "All three URLs must be provided for the tokenized corpus."

        alignments_destination = artifacts / f"{file_name_part}.aln.zst"
        tok_src_destination = artifacts / f"{file_name_part}.tok-icu.{src_locale}.zst"
        tok_trg_destination = artifacts / f"{file_name_part}.tok-icu.{trg_locale}.zst"
        stream_download_to_file(alignments_url, alignments_destination)
        stream_download_to_file(tok_src_url, tok_src_destination)
        stream_download_to_file(tok_trg_url, tok_trg_destination)


if __name__ == "__main__":
    main()
