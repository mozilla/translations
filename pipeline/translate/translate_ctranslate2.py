"""
Translate a corpus with a teacher model (transformer-based) using CTranslate2. This is useful
to quickly synthesize training data for student distillation as CTranslate2 is ~2 times faster
than Marian. For a more detailed analysis see: https://github.com/mozilla/translations/issues/931

https://github.com/OpenNMT/CTranslate2/
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, TextIO
from enum import Enum
from glob import glob
from pathlib import Path
import re

import ctranslate2
import sentencepiece as spm
from ctranslate2.converters.marian import MarianConverter

from pipeline.common.downloads import read_lines, write_lines
from pipeline.common.logging import (
    get_logger,
    start_gpu_logging,
    start_byte_count_logger,
    stop_gpu_logging,
    stop_byte_count_logger,
)
from pipeline.common.marian import get_combined_config
from pipeline.translate.decoder import Decoder


def load_vocab(path: str):
    logger.info("Loading vocab:")
    logger.info(path)
    sp = spm.SentencePieceProcessor(path)

    return [sp.id_to_piece(i) for i in range(sp.vocab_size())]


# The vocab expects a .yml file. Instead directly load the vocab .spm file via a monkey patch.
if not ctranslate2.converters.marian.load_vocab:
    raise Exception("Expected to be able to monkey patch the load_vocab function")
ctranslate2.converters.marian.load_vocab = load_vocab

logger = get_logger(__file__)
ctranslate2.set_random_seed(42)


class Device(Enum):
    gpu = "gpu"
    cpu = "cpu"


class MaxiBatchSort(Enum):
    src = "src"
    none = "none"


def get_model(models_globs: list[str]) -> Path:
    models: list[Path] = []
    for models_glob in models_globs:
        for path in glob(models_glob):
            models.append(Path(path))
    if not models:
        raise ValueError(f'No model was found with the glob "{models_glob}"')
    if len(models) != 1:
        logger.info(f"Found models {models}")
        raise ValueError("Ensemble training is not supported in CTranslate2")
    return Path(models[0])


class DecoderConfig:
    def __init__(self, extra_marian_args: list[str]) -> None:
        super().__init__()
        # Combine the two configs.
        self.config = get_combined_config(Path(__file__).parent / "decoder.yml", extra_marian_args)

        self.mini_batch_words: int = self.get_from_config("mini-batch-words", int)
        self.beam_size: int = self.get_from_config("beam-size", int)
        self.precision = self.get_from_config("precision", str, "float32")
        if self.get_from_config("fp16", bool, False):
            self.precision = "float16"
        self.sampling_topk, self.sampling_temperature = 1, 1.0
        if "output-sampling" in self.config and self.get_from_config("output-sampling", list):
            self.sampling_topk, self.sampling_temperature = self.parse_sampling()

        if self.beam_size > 1 and self.sampling_topk > 1:
            raise ValueError("Beam size has to be 1 if sampling is enabled")

    def get_from_config(self, key: str, type: any, default=None):
        value = self.config.get(key, default)
        if value is None:
            raise ValueError(f'"{key}" could not be found in the decoder.yml config')
        if isinstance(value, type):
            return value
        if type != str and isinstance(value, str):
            return type(value)
        raise ValueError(f'Expected "{key}" to be of a type "{type}" in the decoder.yml config')

    def parse_sampling(self):
        """
        Expected output-sampling param format to be the same as marian-decoder param
        """
        if len(self.config["output-sampling"]) < 2:
            raise ValueError(
                "output-sampling mus specify at least two values <method> <num_topk> [temp]"
            )
        mode = self.config["output-sampling"][0]
        if mode != "topk":
            raise ValueError(f"Only output-sampling topk is supported, received {mode}")
        if len(self.config["output-sampling"]) == 2:
            return int(self.config["output-sampling"][1]), 1.0
        if len(self.config["output-sampling"]) == 3:
            # Replace Marian float format '.f' by '.0'
            temp = re.sub(r"^(\d+)\.f$", r"\1.0", self.config["output-sampling"][2])
            return int(self.config["output-sampling"][1]), float(temp)


class Translator(ABC):
    def write_translation(index: int, is_nbest: bool, hypotheses: List[str], outfile: TextIO):
        """
        Match Marian's way of writing out nbest translations. For example, with a beam-size of 2 and
        collection nbest translations:

        0 ||| Translation attempt
        0 ||| An attempt at translation
        1 ||| The quick brown fox jumped
        1 ||| The brown fox quickly jumped
        ...

        If no nbest candidates are provided, write a single line without formatting.
        """
        if is_nbest:
            for hypothesis in hypotheses:
                outfile.write(f"{index} ||| {hypothesis}\n")
        else:
            outfile.write(hypotheses[0])
            outfile.write("\n")

    @abstractmethod
    def translate_iterable(
        self,
        is_nbest,
        source: Iterable[str],  # Iterable of untokenized strings, not the same as Ctranslate
        **kwargs,
    ) -> Iterable[List[str]]:
        """
        Wrap around Ctranslate2 translate_iterable to unify interfaces
        so that input and output are untokenized strings
        """


class TranslatorCtranslate2(Translator):
    def __init__(
        self,
        models_globs: list[str],
        vocab: list[str],
        precision: str,
        device: str,
        device_index: list[int],
    ):
        self.model = get_model(models_globs)
        self.tokenizer_src = spm.SentencePieceProcessor(vocab[0])
        if len(vocab) == 1:
            self.tokenizer_trg = self.tokenizer_src
        else:
            self.tokenizer_trg = spm.SentencePieceProcessor(vocab[1])

        ctranslate2_model_dir = self.model.parent / f"{Path(self.model).stem}"
        logger.info("Converting the Marian model to Ctranslate2:")
        logger.info(self.model)
        logger.info("Outputing model to:")
        logger.info(ctranslate2_model_dir)

        converter = MarianConverter(self.model, vocab)
        converter.convert(ctranslate2_model_dir, quantization=precision)

        if device == "gpu":
            self.translator = ctranslate2.Translator(
                str(ctranslate2_model_dir), device="cuda", device_index=device_index
            )
        else:
            self.translator = ctranslate2.Translator(str(ctranslate2_model_dir), device="cpu")

        logger.info("Loading model")
        self.translator.load_model()
        logger.info("Model loaded")

    def tokenize(self, line):
        return self.tokenizer_src.Encode(line, out_type=str)

    def translate_iterable(
        self,
        source: Iterable[str],
        is_nbest: bool,
        **kwargs,
    ) -> Iterable[List[str]]:
        """
        Adapter from the common interface to Ctranslate2 interface
        it tokenizes with sentencepiece before sending it to ct2
        and detokenizes results
        """
        for result in self.translator.translate_iterable(map(self.tokenize, source), **kwargs):
            if is_nbest:
                yield [self.tokenizer_trg.decode(h) for h in result.hypotheses]
            else:
                yield [self.tokenizer_trg.decode(result.hypotheses[0])]


class TranslatorIndicTrans2(Translator):
    def __init__(
        self,
        src_locale: str,
        trg_locale: str,
        mini_batch_size: int,
        beam_size: int,
        device: str,
        device_index: list[int],
    ):
        from indictrans2_ct2_inference.translate import Translator as IndicTrans2Inference

        self.maxi_batch_size = 10000
        self.model = IndicTrans2Inference(
            src_locale,
            trg_locale,
            device=device,
            device_index=device_index if device == "gpu" else 0,
            beam_size=beam_size,
            mini_batch_size=mini_batch_size,
        )

    def translate_iterable(
        self,
        source: Iterable[str],
        is_nbest: bool,
        **kwargs,
    ) -> Iterable[List[str]]:
        """
        Adapter from the common interface to IndicTrans2 interface
        which does not support translate_iterable, just batched
        so this just caches from the iterable source and translates in batches
        then yields as if it was an iterator

        TODO: n-best generation is not supported, so the output is always the same: [hyp_0]
        """

        def batched(stream):
            batch = []
            for i in stream:
                batch.append(i.strip())
                if len(batch) > self.maxi_batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        num_hypotheses = self.beam_size if is_nbest else 1
        for batch in batched(source):
            result = self.model.batch_translate(batch, num_hypotheses)
            assert len(result) == len(batch)
            for i in result:
                yield i


def translate_with_ctranslate2(
    input_zst: Path,
    artifacts: Path,
    extra_marian_args: list[str],
    models_globs: list[str],
    decoder_type: Decoder,
    is_nbest: bool,
    src_locale: str,
    trg_locale: str,
    vocab: list[str],
    device: str,
    device_index: list[int],
) -> None:
    postfix = "nbest" if is_nbest else "out"

    if extra_marian_args and extra_marian_args[0] != "--":
        logger.error(" ".join(extra_marian_args))
        raise Exception("Expected the extra marian args to be after a --")

    decoder_config = DecoderConfig(extra_marian_args[1:])
    translator = None
    if decoder_type == Decoder.ctranslate2:
        translator = TranslatorCtranslate2(
            models_globs, vocab, decoder_config.precision, device, device_index
        )
    elif decoder_type == Decoder.indictrans2:
        translator = TranslatorIndicTrans2(
            src_locale=src_locale,
            trg_locale=trg_locale,
            mini_batch_size=decoder_config.mini_batch_words,
            beam_size=decoder_config.beam_size,
            device=device,
            device_index=device_index,
        )
    else:
        raise ValueError("Decoder cannot be {decoder_type}")

    output_zst = artifacts / f"{input_zst.stem}.{postfix}.zst"

    five_minutes = 300
    if device == "gpu":
        start_gpu_logging(logger, five_minutes)
    start_byte_count_logger(logger, five_minutes, output_zst)

    index = 0
    with write_lines(output_zst) as outfile, read_lines(input_zst) as lines:
        for result in translator.translate_iterable(
            lines,
            is_nbest,
            # Options for "translate_iterable":
            # https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_iterable
            max_batch_size=decoder_config.mini_batch_words,
            batch_type="tokens",
            # Options for "translate_batch":
            # https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch
            beam_size=decoder_config.beam_size,
            return_scores=False,
            num_hypotheses=1 if is_nbest else decoder_config.beam_size,
            sampling_topk=decoder_config.sampling_topk,
            sampling_temperature=decoder_config.sampling_temperature,
        ):
            Translator.write_translation(index, is_nbest, result, outfile)
            index += 1

    stop_gpu_logging()
    stop_byte_count_logger()
