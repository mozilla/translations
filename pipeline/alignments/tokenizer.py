"""
Tokenizes a text file with line separated sentences using Moses tokenizer.

Example:
  python pipeline/alignments/tokenizer.py --input_path=data/datasets/news.2023.en.shuffled.deduped \
    --output_path=data/datasets/news.2023.en.shuffled.deduped.tok-icu --lang=en --chunk_size=500000 --tokenizer=icu

Using C++ opus-fast-mosestokenizer sometimes requires specifying LD_LIBRARY_PATH before starting the Python process
see https://github.com/Helsinki-NLP/opus-fast-mosestokenizer/issues/6
export LD_LIBRARY_PATH=.../<you-python-env>/lib/python3.10/site-packages/mosestokenizer/lib

Using ICU tokenizer requires installing it with `apt-get install python3-icu`,
see more installation instructions here: https://pypi.org/project/PyICU/

Whitespaces are ignored by Moses based tokenizers and preserved and replaced with a special token "▁" by ICU tokenizer
which allows lossless reconstruction of the original text on detokenization.

"""
import argparse
import multiprocessing
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import re
from typing import List

from tqdm import tqdm

from pipeline.common.logging import get_logger

logger = get_logger("tokenizer")


class TokenizerType(Enum):
    fast_moses = "fast_moses"
    sacre_moses = "sacre_moses"
    icu = "icu"


class Tokenizer(ABC):
    def __init__(self, lang: str):
        self.lang = lang

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        pass


class FastMosesTokenizer(Tokenizer):
    """
    Uses Moses tokenizer https://github.com/Helsinki-NLP/opus-fast-mosestokenizer
    """

    def __init__(self, lang):
        super().__init__(lang)
        from mosestokenizer import MosesTokenizer

        try:
            self.tokenizer = MosesTokenizer(lang)
        except RuntimeError as err:
            msg = str(err)
            if "No known abbreviations for language" in msg:
                # Fall-back to English if the language is not found
                self.tokenizer = MosesTokenizer("en")
            else:
                raise err

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self.tokenizer.detokenize(tokens)


class SacreMosesTokenizer(Tokenizer):
    """
    Uses Moses tokenizer https://github.com/hplt-project/sacremoses
    """

    def __init__(self, lang):
        super().__init__(lang)
        import sacremoses

        self.tokenizer = sacremoses.MosesTokenizer(lang)
        self.detokenizer = sacremoses.MosesDetokenizer(lang)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self.detokenizer.detokenize(tokens)


class IcuTokenizer(Tokenizer):
    """
    Uses ICU based word segmenter https://pypi.org/project/PyICU/
    Preserves whitespaces as tokens by replacing them with a special character "▁".
    Allows lossless reconstruction of the original text on detokenization.
    """

    # Same character is used by SentencePiece
    SPACE_TOKEN = "▁"
    RE_SPACES = re.compile("^ +$")

    def tokenize(self, text: str, nospace=False) -> List[str]:
        from icu import BreakIterator, Locale

        bi = BreakIterator.createWordInstance(Locale(self.lang))
        bi.setText(text)

        tokens = []
        start = bi.first()
        for end in bi:
            token = text[start:end]
            start = end
            if not token or token == "\n":  # exclude empty tokens
                continue
            if nospace:  # nospace behaves as old tokenizers, discards all spaces
                if self.RE_SPACES.match(token):
                    continue
                tokens.append(token)
            else:  # keep whitespaces and replace them with a special token
                tokens.append(token.replace(" ", self.SPACE_TOKEN))
        return tokens

    def tokenize_nospace(self, text: str) -> List[str]:
        return self.tokenize(text, nospace=True)

    def detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens).replace(self.SPACE_TOKEN, " ")


def _read_file_in_chunks(file_path: Path, chunk_size: int):
    with open(file_path, "r", encoding="utf-8") as file:
        while True:
            lines = file.readlines(chunk_size)
            if not lines:
                break
            yield [line.rstrip() for line in lines]


def _tokenize_lines(params) -> List[str]:
    lines, lang, tok_type = params

    if tok_type == TokenizerType.fast_moses:
        tokenizer = FastMosesTokenizer(lang)
    elif tok_type == TokenizerType.sacre_moses:
        tokenizer = SacreMosesTokenizer(lang)
    elif tok_type == TokenizerType.icu:
        tokenizer = IcuTokenizer(lang)
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")

    tokenized = []
    for line in lines:
        tokens = tokenizer.tokenize(line)
        tokenized.append(" ".join(tokens))
    return tokenized


def tokenize(
    input_path: Path,
    output_path: Path,
    lang: str,
    tokenizer: TokenizerType,
    sentences_per_chunk: int = 100000,
) -> None:
    logger.info(f"Tokenizing {input_path} with Moses tokenizer")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with open(output_path, "w") as output_file:
            chunks = _read_file_in_chunks(input_path, chunk_size=sentences_per_chunk)

            pbar = tqdm(mininterval=10)
            # ~100K sentences per second on a single core
            for tokenized_chunk in pool.imap(
                _tokenize_lines,
                ((ch, lang, tokenizer) for ch in chunks),
            ):
                output_file.write("\n".join(tokenized_chunk) + "\n")
                pbar.update(len(tokenized_chunk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output_path",
        metavar="OUTPUT_PATH",
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--input_path",
        metavar="INPUT_PATH",
        type=str,
        default=None,
        help="Input file",
    )
    parser.add_argument(
        "--lang",
        metavar="LANG",
        type=str,
        default=None,
        help="Language",
    )
    parser.add_argument(
        "--chunk_size",
        metavar="CHUNK_SIZE",
        type=int,
        default=None,
        help="Number of lines to process per chunk",
    )
    parser.add_argument(
        "--tokenizer",
        metavar="TOKENIZER",
        type=TokenizerType,
        choices=TokenizerType,
        default=TokenizerType.icu,
        help="Tokenization method",
    )
    args = parser.parse_args()
    tokenize(
        input_path=args.input_path,
        output_path=args.output_path,
        lang=args.lang,
        sentences_per_chunk=args.chunk_size,
        tokenizer=args.tokenizer,
    )
