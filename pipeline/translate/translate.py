"""
Translate a corpus using either Marian or CTranslate2.
"""

import argparse
from enum import Enum
from glob import glob
import os
from pathlib import Path
import re
import tempfile

from pipeline.common.command_runner import apply_command_args, run_command
from pipeline.common.datasets import compress, decompress
from pipeline.common.downloads import count_lines, is_file_empty, write_lines
from pipeline.common.logging import (
    get_logger,
    start_gpu_logging,
    start_byte_count_logger,
    stop_gpu_logging,
    stop_byte_count_logger,
)
from pipeline.common.marian import get_combined_config
from pipeline.translate.translate_ctranslate2 import translate_with_ctranslate2
from pipeline.common.marian import assert_gpus_available

logger = get_logger(__file__)

DECODER_CONFIG_PATH = Path(__file__).parent / "decoder.yml"


class Decoder(Enum):
    marian = "marian"
    ctranslate2 = "ctranslate2"


class Device(Enum):
    cpu = "cpu"
    gpu = "gpu"


def get_beam_size(extra_marian_args: list[str]):
    return get_combined_config(DECODER_CONFIG_PATH, extra_marian_args)["beam-size"]


def strip_vec_args(extra_marian_args: list[str]):
    """
    Strip vector delimiters that come from expanding yaml config into cli arguments
    e.g. --output-sampling [topk, 10] -> --output-sampling topk 10
    """
    strip_vec = re.compile(r"^[,\[]?([\w+\-]+)[,\]]$")
    return [strip_vec.sub(r"\1", i) for i in extra_marian_args]


def run_marian(
    marian_dir: Path,
    models: list[Path],
    vocabs: tuple[str, str],
    input: Path,
    output: Path,
    gpus: list[str],
    workspace: int,
    is_nbest: bool,
    extra_args: list[str],
):
    config = Path(__file__).parent / "decoder.yml"
    marian_bin = str(marian_dir / "marian-decoder")
    log = input.parent / f"{input.name}.log"
    if is_nbest:
        extra_args = ["--n-best", *extra_args]

    logger.info("Starting Marian to translate")

    run_command(
        [
            marian_bin,
            *apply_command_args(
                {
                    "config": config,
                    "models": models,
                    "vocabs": vocabs,
                    "input": input,
                    "output": output,
                    "log": log,
                    "devices": gpus,
                    "workspace": workspace,
                }
            ),
            *extra_args,
        ],
        logger=logger,
        env={**os.environ},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input", type=Path, required=True, help="The path to the text to translate."
    )
    parser.add_argument(
        "--models_glob",
        type=str,
        required=True,
        nargs="+",
        help="A glob pattern to the Marian model(s)",
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="Output path to the artifacts."
    )
    parser.add_argument("--nbest", action="store_true", help="Whether to use the nbest")
    parser.add_argument(
        "--marian_dir", type=Path, required=True, help="The path the Marian binaries"
    )
    parser.add_argument("--vocab_src", type=Path, help="Path to src vocab file")
    parser.add_argument("--vocab_trg", type=Path, help="Path to trg vocab file")
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='The indexes of the GPUs to use on a system, e.g. --gpus "0 1 2 3"',
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="The amount of Marian memory (in MB) to preallocate",
    )
    parser.add_argument(
        "--decoder",
        type=Decoder,
        default=Decoder.marian,
        help="Either use the normal marian decoder, or opt for CTranslate2.",
    )
    parser.add_argument(
        "--device",
        type=Device,
        default=Device.gpu,
        help="Either use the normal marian decoder, or opt for CTranslate2.",
    )
    parser.add_argument(
        "extra_marian_args",
        nargs=argparse.REMAINDER,
        help="Additional parameters for the training script",
    )

    args = parser.parse_args()

    # Provide the types for the arguments.
    marian_dir: Path = args.marian_dir
    input_zst: Path = args.input
    artifacts: Path = args.artifacts
    models_globs: list[str] = args.models_glob
    models: list[Path] = []
    for models_glob in models_globs:
        for path in glob(models_glob):
            models.append(Path(path))
    postfix = "nbest" if args.nbest else "out"
    output_zst = artifacts / f"{input_zst.stem}.{postfix}.zst"
    vocab_src: Path = args.vocab_src
    vocab_trg: Path = args.vocab_trg
    gpus: list[str] = args.gpus.split(" ")
    extra_marian_args: list[str] = strip_vec_args(args.extra_marian_args)
    decoder: Decoder = args.decoder
    is_nbest: bool = args.nbest
    device: Device = args.device

    logger.info(extra_marian_args)
    logger.info(args.extra_marian_args)

    # Do some light validation of the arguments.
    assert input_zst.exists(), f"The input file exists: {input_zst}"
    assert vocab_src.exists(), f"The vocab src file exists: {vocab_src}"
    assert vocab_trg.exists(), f"The vocab trg file exists: {vocab_trg}"
    if not artifacts.exists():
        artifacts.mkdir()
    for gpu_index in gpus:
        assert gpu_index.isdigit(), f'GPUs must be list of numbers: "{gpu_index}"'
    assert models, "There must be at least one model"
    for model in models:
        assert model.exists(), f"The model file exists {model}"
    if extra_marian_args and extra_marian_args[0] != "--":
        logger.error(" ".join(extra_marian_args))
        raise Exception("Expected the extra marian args to be after a --")
    if get_beam_size(extra_marian_args) != "1" and "--output-sampling" in extra_marian_args:
        raise Exception("Output sampling and beam search are incompatible, set beam to 1")

    logger.info(f"Input file: {input_zst}")
    logger.info(f"Output file: {output_zst}")

    # Taskcluster can produce empty input files when chunking out translation for
    # parallelization. In this case skip translating, and write out an empty file.
    if is_file_empty(input_zst):
        logger.info(f"The input is empty, create a blank output: {output_zst}")
        with write_lines(output_zst) as _outfile:
            # Nothing to write, just create the file.
            pass
        return

    assert_gpus_available(logger)

    if decoder == Decoder.ctranslate2:
        translate_with_ctranslate2(
            input_zst=input_zst,
            artifacts=artifacts,
            extra_marian_args=extra_marian_args,
            models_globs=models_globs,
            is_nbest=is_nbest,
            vocab=[str(vocab_src), str(vocab_trg)],
            device=device.value,
            device_index=[int(n) for n in gpus],
        )
        return

    # The device flag is for use with CTranslate, but add some assertions here so that
    # we can be consistent in usage.
    if device == Device.cpu:
        assert (
            "--cpu-threads" in extra_marian_args
        ), "Marian's cpu should be controlled with the flag --cpu-threads"
    else:
        assert (
            "--cpu-threads" not in extra_marian_args
        ), "Requested a GPU device, but --cpu-threads was provided"

    # Run the training.
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_txt = temp_dir / input_zst.stem
        output_txt = temp_dir / output_zst.stem

        decompress(input_zst, destination=input_txt, remove=True, logger=logger)

        five_minutes = 300
        if device == Device.gpu:
            start_gpu_logging(logger, five_minutes)
        start_byte_count_logger(logger, five_minutes, output_txt)

        run_marian(
            marian_dir=marian_dir,
            models=models,
            vocabs=(str(vocab_src), str(vocab_trg)),
            input=input_txt,
            output=output_txt,
            gpus=gpus,
            workspace=args.workspace,
            is_nbest=is_nbest,
            # Take off the initial "--"
            extra_args=extra_marian_args[1:],
        )

        stop_gpu_logging()
        stop_byte_count_logger()

        compress(output_txt, destination=output_zst, remove=True, logger=logger)

        input_count = count_lines(input_txt)
        output_count = count_lines(output_zst)
        if is_nbest:
            beam_size = get_beam_size(extra_marian_args)
            expected_output = input_count * beam_size
            assert (
                expected_output == output_count
            ), f"The nbest output had {beam_size}x as many lines ({expected_output} vs {output_count})"
        else:
            assert (
                input_count == output_count
            ), f"The input ({input_count} and output ({output_count}) had the same number of lines"


if __name__ == "__main__":
    main()
