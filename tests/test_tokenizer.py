from pathlib import Path

import pytest

from pipeline.alignments.tokenizer import IcuTokenizer, TokenizerType, tokenize
from fixtures import zh_sample, en_sample, ru_sample, DataDir

tokenized_first_lines = {
    "en": "The ▁ little ▁ girl , ▁ seeing ▁ she ▁ had ▁ lost ▁ one ▁ of ▁ her ▁ pretty ▁ shoes , ▁ grew ▁ angry , ▁ and ▁ said ▁ to ▁ the ▁ Witch , ▁ “ Give ▁ me ▁ back ▁ my ▁ shoe ! ”",
    "ru": "Маленькая ▁ девочка , ▁ увидев , ▁ что ▁ потеряла ▁ одну ▁ из ▁ своих ▁ красивых ▁ туфелек , ▁ рассердилась ▁ и ▁ сказала ▁ Ведьме : ▁ « Верни ▁ мне ▁ мою ▁ туфельку ! »",
    "zh": "小 女孩 看到 自己 丢 了 一只 漂亮 的 鞋子 ， 生气 了 ， 对 女巫 说 ： “ 把 我的 鞋子 还给 我 ！ ”",
}

tokenized_nospace_first_lines = {
    "en": "The little girl , seeing she had lost one of her pretty shoes , grew angry , and said to the Witch , “ Give me back my shoe ! ”",
    "ru": "Маленькая девочка , увидев , что потеряла одну из своих красивых туфелек , рассердилась и сказала Ведьме : « Верни мне мою туфельку ! »",
    "zh": "小 女孩 看到 自己 丢 了 一只 漂亮 的 鞋子 ， 生气 了 ， 对 女巫 说 ： “ 把 我的 鞋子 还给 我 ！ ”",
}


@pytest.mark.parametrize(
    "lang,sample,first_line",
    [
        ("en", en_sample, tokenized_first_lines["en"]),
        ("ru", ru_sample, tokenized_first_lines["ru"]),
        ("zh", zh_sample, tokenized_first_lines["zh"]),
        ("zh", "这是一个简单的测试语句 🤣 。", "这 是 一个 简单 的 测试 语 句 ▁ 🤣▁ 。"),
    ],
    ids=["en", "ru", "zh", "zh2"],
)
def test_icu_tokenize_detokenize(lang, sample, first_line):
    lines = sample.splitlines()
    tokenizer = IcuTokenizer
    icu_tokenizer = tokenizer(lang)
    tok_lines = []
    detok_lines = []

    for line in lines:
        tokens = icu_tokenizer.tokenize(line)
        detokenized = icu_tokenizer.detokenize(tokens)
        tok_lines.append(" ".join(tokens))
        detok_lines.append(detokenized)

    assert lines == detok_lines
    assert tok_lines[0] == first_line


@pytest.mark.parametrize(
    "lang,sample,first_line",
    [
        ("en", en_sample, tokenized_nospace_first_lines["en"]),
        ("ru", ru_sample, tokenized_nospace_first_lines["ru"]),
        ("zh", zh_sample, tokenized_nospace_first_lines["zh"]),
        ("zh", "这是一个简单的测试语句 🤣 。", "这 是 一个 简单 的 测试 语 句 🤣  。"),
    ],
    ids=["en", "ru", "zh", "zh2"],
)
def test_icu_tokenize_nospace(lang, sample, first_line):
    lines = sample.splitlines()
    tokenizer = IcuTokenizer
    icu_tokenizer = tokenizer(lang)
    tok_lines = []

    for line in lines:
        tokens = icu_tokenizer.tokenize_nospace(line)
        tok_lines.append(" ".join(tokens))

    assert tok_lines[0] == first_line


@pytest.mark.parametrize(
    "lang,sample",
    [
        (
            "en",
            en_sample,
        ),
        (
            "ru",
            ru_sample,
        ),
        ("zh", zh_sample),
    ],
    ids=["en", "ru", "zh"],
)
def test_tokenizer(lang, sample):
    data_dir = DataDir("test_tokenizer")
    input_path = Path(data_dir.create_file(f"input.{lang}.txt", sample))
    output_path = Path(data_dir.join(f"output.{lang}.txt"))

    tokenize(
        input_path=input_path,
        output_path=output_path,
        lang=lang,
        tokenizer=TokenizerType.icu,
        sentences_per_chunk=3,
    )

    with open(output_path) as f:
        lines = f.read().splitlines()

    assert len(lines) == len(sample.splitlines())
    assert lines[0] == tokenized_first_lines[lang]


@pytest.mark.parametrize(
    "lang,text,expected_tokenized",
    [
        (
            "en",
            "This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf",
            "This , ▁ is ▁ a ▁ sentence ▁ with ▁ weird » ▁ symbols … ▁ appearing ▁ everywhere ¿",
        ),
        ("en", "abc def.", "abc ▁ def ."),
        ("en", "2016, pp.", "2016 , ▁ pp ."),
        (
            "en",
            "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?",
            "This ▁ ain't ▁ funny . ▁ It's ▁ actually ▁ hillarious , ▁ yet ▁ double ▁ Ls . ▁ | ▁ [ ] ▁ < ▁ > ▁ [ ▁ ] ▁ & ▁ You're ▁ gonna ▁ shake ▁ it ▁ off ? ▁ Don't ?",
        ),
        ("en", "this 'is' the thing", "this ▁ ' is ' ▁ the ▁ thing"),
        (
            "en",
            "By the mid 1990s a version of the game became a Latvian television series (with a parliamentary setting, and played by Latvian celebrities).",
            "By ▁ the ▁ mid ▁ 1990s ▁ a ▁ version ▁ of ▁ the ▁ game ▁ became ▁ a ▁ Latvian ▁ television ▁ series ▁ ( with ▁ a ▁ parliamentary ▁ setting , ▁ and ▁ played ▁ by ▁ Latvian ▁ celebrities ) .",
        ),
        (
            "en",
            "The meeting will take place at 11:00 a.m. Tuesday.",
            "The ▁ meeting ▁ will ▁ take ▁ place ▁ at ▁ 11 : 00 ▁ a.m . ▁ Tuesday .",
        ),
        ("en", "'Hello.'", "' Hello . '"),
        ("en", "'So am I.", "' So ▁ am ▁ I ."),
        (
            "fr",
            "Des gens admirent une œuvre d'art.",
            "Des ▁ gens ▁ admirent ▁ une ▁ œuvre ▁ d'art .",
        ),
        ("de", "...schwer wie ein iPhone 5.", ". . . schwer ▁ wie ▁ ein ▁ iPhone ▁ 5 ."),
        ("cz", "Dvě děti, které běží bez bot.", "Dvě ▁ děti , ▁ které ▁ běží ▁ bez ▁ bot ."),
        (
            "en",
            "this is a webpage https://stackoverflow.com/questions/6181381/how-to-print-variables-in-perl that kicks ass",
            "this ▁ is ▁ a ▁ webpage ▁ https : / / stackoverflow.com / questions / 6181381 / how - to - print - variables - in - perl ▁ that ▁ kicks ▁ ass",
        ),
        (
            "en",
            "What about a this,type,of-s-thingy?",
            "What ▁ about ▁ a ▁ this , type , of - s - thingy ?",
        ),
        (
            "de",
            "Sie sollten vor dem Upgrade eine Sicherung dieser Daten erstellen (wie unter Abschnitt 4.1.1, „Sichern aller Daten und Konfigurationsinformationen“ beschrieben). ",
            "Sie ▁ sollten ▁ vor ▁ dem ▁ Upgrade ▁ eine ▁ Sicherung ▁ dieser ▁ Daten ▁ erstellen ▁ ( wie ▁ unter ▁ Abschnitt ▁ 4.1.1 , ▁ „ Sichern ▁ aller ▁ Daten ▁ und ▁ Konfigurationsinformationen “ ▁ beschrieben ) . ▁",
        ),
        (
            "fr",
            "L'amitié nous a fait forts d'esprit",
            "L'amitié ▁ nous ▁ a ▁ fait ▁ forts ▁ d'esprit",
        ),
        ("zh", "记者 应谦 美国", "记者 ▁ 应 谦 ▁ 美国"),
        ("ko", "세계 에서 가장 강력한.", "세계 ▁ 에서 ▁ 가장 ▁ 강력한 ."),
        ("ja", "電話でんわの邪魔じゃまをしないでください", "電話 でんわ の 邪魔 じゃ ま を しない で くだ さい"),
        ("ja", "Japan is 日本 in Japanese.", "Japan ▁ is ▁ 日本 ▁ in ▁ Japanese ."),
    ],
    ids=[
        "en_weird_symbols",
        "en_fullstop",
        "en_numeric_prefix",
        "en_braces",
        "en_apostrophe",
        "en_opening_brackets",
        "en_dot_splitting",
        "en_trailing_dot_apostrophe",
        "en_one_apostrophe",
        "fr",
        "de",
        "cz",
        "en_pattern1",
        "en_pattern2",
        "de_final_comma_split_after_number",
        "fr_apostrophes",
        "zh",
        "ko",
        "ja",
        "cjk_mix",
    ],
)
def test_icu_tokens(lang, text, expected_tokenized):
    """
    Tests tokens produced by ICU tokenizer.

    The use cases were copied from https://github.com/hplt-project/sacremoses/blob/master/sacremoses/test/test_tokenizer.py as is.
    However, this test is mostly to show how the tokenizer works rather than fixing it because it relies on the underlying ICU tokenizer.
    The expected values were just copied from the test run.
    Having some mistakes in tokenization is ok because it's used only for the purposes of inline noise augmentation and to produce word alignments.
    """
    icu_tokenizer = IcuTokenizer(lang)

    tokens = icu_tokenizer.tokenize(text)
    tokenized = " ".join(tokens)

    assert expected_tokenized == tokenized
