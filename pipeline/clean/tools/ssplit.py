from icu import BreakIterator, Locale
import argparse
import re
import sys

from pipeline.langs.codes import LangCode
from pipeline.clean.tools.clean_mono import MAX_LENGTH

RE_SPACES = re.compile("\s+")

def get_sentences(text, locale=None):
    """
    Split a string into sentences using ICU's sentence break iterator.

    Args:
        text: The input string.
        locale: Optional ICU Locale or locale string (e.g., 'ur', 'en_US').

    Returns:
        A generator of sentence strings.
    """
    if locale is None:
        locale = Locale.getDefault()
    elif isinstance(locale, str):
        locale = Locale(locale)

    bi = BreakIterator.createSentenceInstance(locale)
    bi.setText(text)

    start = bi.first()
    end = bi.nextBoundary()
    while end != BreakIterator.DONE:
        sentence = str(text[start:end]).strip()
        if not sentence or RE_SPACES.fullmatch(sentence):
            continue # skip empty lines
        yield sentence.strip()
        start = end
        end = bi.nextBoundary()

def main():
    args = parse_user_args()
    lang = LangCode(args.lang)
    max_length = int(MAX_LENGTH * 0.9)

    for line in sys.stdin:
        line_num_toks = len(line.split()) if not lang.is_cjk() else len(line)

        if line_num_toks < MAX_LENGTH:
            print(line, end='')
            continue

        for sent in get_sentences(line, lang):
            print(sent)

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", default="en")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
