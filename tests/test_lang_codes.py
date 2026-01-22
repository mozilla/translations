import pytest

from pipeline.langs.codes import (
    to_iso6391,
    to_iso6393,
    to_iso6393_individual_and_script,
    to_locale,
    iso6393_and_script_to_lang_id,
    icu_normalize,
    generate_all,
)
from pipeline.langs.scripts import get_script_info


@pytest.mark.parametrize(
    "source,expected",
    [
        ("en", "en"),
        ("ru", "ru"),
        ("zh", "zh"),
        ("zh_hant", "zh"),
        ("zh_hans", "zh"),
        ("sr_cyrl", "sr"),
        ("pt_br", "pt"),
    ],
)
def test_to_iso6391(source: str, expected: str):
    assert to_iso6391(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("en", "eng"),
        ("ru", "rus"),
        ("zh", "zho"),
        ("zh_hant", "zho"),
        ("ko", "kor"),
        ("pt", "por"),
        ("pt_br", "por"),
    ],
)
def test_to_iso6393(source: str, expected: str):
    assert to_iso6393(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("en", "eng_Latn"),
        ("ru", "rus_Cyrl"),
        ("zh", "cmn_Hans"),
        ("zh_hant", "cmn_Hant"),
        ("ko", "kor_Hang"),
        ("pt", "por_Latn"),
        ("pt_pt", "por_Latn"),
        ("ja", "jpn_Jpan"),
    ],
)
def test_to_iso6393_individual_and_script(source: str, expected: str):
    assert to_iso6393_individual_and_script(source) == expected


@pytest.mark.parametrize(
    "expected,source",
    [
        ("en", "eng_Latn"),
        ("ru", "rus_Cyrl"),
        ("zh", "cmn_Hans"),
        ("zh_hant", "cmn_Hant"),
        ("pt", "por_Latn"),
        ("ko", "kor_Hang"),
        ("pt", "por_Latn"),
        ("ja", "jpn_Jpan"),
    ],
)
def test_iso6393_and_script_to_lang_id(expected: str, source: str):
    assert iso6393_and_script_to_lang_id(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("zh_hant", "zh_TW"),
        ("zh", "zh_CN"),
        ("pt_br", "pt_BR"),
        ("pt", "pt_BR"),
        ("en", "en_US"),
        ("ru", "ru_RU"),
    ],
)
def test_to_locale(source: str, expected: str):
    assert to_locale(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("zh_hant", "Han (Traditional)"),
        ("zh", "Han (Simplified)"),
        ("en", "Latin"),
        ("ru", "Cyrillic"),
        ("sr_cyrl", "Cyrillic"),
    ],
)
def test_script_info(source: str, expected: str):
    script = get_script_info(source)
    assert script is not None
    assert script["name"] == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("zh_hant", "zh_Hant"),
        ("zh", "zh"),
        ("ru", "ru"),
        ("en", "en"),
        ("sr_cyrl", "sr_Cyrl"),
        ("pt_br", "pt_BR"),
    ],
)
def test_icu_normalize(source: str, expected: str):
    assert icu_normalize(source) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        (
            "zh",
            {
                "name": "Chinese",
                "script": "Han (Simplified)",
                "opus": "zh",
                "mtdata": "zho",
                "sacrebleu": "zh",
                "flores101": "zho_simpl",
                "pontoon": "zh-CN",
                "hplt": "cmn_Hans",
                "newscrawl": "zh",
                "opuscleaner": "zh",
                "bicleaner": "zh",
                "monocleaner": "zh",
                "fasttext": "cmn_Hans",
                "comet22": "zh",
                "metricx24": "zh",
                "flores200-plus": "cmn_Hans",
                "bouquet": "cmn_Hans",
                "wmt24pp": "zh_CN",
                "nllb": "zho_Hans",
                "google": "zh",
                "microsoft": "zh-Hans",
            },
        ),
        (
            "zh_hant",
            {
                "name": "Chinese (Traditional)",
                "script": "Han (Traditional)",
                "opus": "zh",
                "mtdata": "zho",
                "sacrebleu": "zh",
                "flores101": "zho_trad",
                "pontoon": "zh-TW",
                "hplt": "cmn_Hant",
                "newscrawl": "zh",
                "opuscleaner": "zh_Hant",
                "bicleaner": "zh",
                "monocleaner": "zh",
                "fasttext": "cmn_Hant",
                "comet22": "zh",
                "metricx24": "zh",
                "flores200-plus": "cmn_Hant",
                "bouquet": "not supported",
                "wmt24pp": "zh_TW",
                "nllb": "zho_Hant",
                "google": "zh-TW",
                "microsoft": "zh-Hant",
            },
        ),
        (
            "tl",
            {
                "name": "Tagalog",
                "script": "Latin",
                "opus": "tl",
                "mtdata": "tgl",
                "sacrebleu": "tl",
                "flores101": "tgl",
                "pontoon": "tl",
                "hplt": "tgl_Latn",
                "newscrawl": "tl",
                "opuscleaner": "tl",
                "bicleaner": "tl",
                "monocleaner": "tl",
                "fasttext": "tgl_Latn",
                "comet22": "fil",
                "metricx24": "fil",
                "flores200-plus": "fil_Latn",
                "bouquet": "tgl_Latn",
                "wmt24pp": "fil_PH",
                "nllb": "tgl_Latn",
                "google": "tl",
                "microsoft": "fil",
            },
        ),
        (
            "sr",
            {
                "name": "Serbian",
                "script": "Cyrillic",
                "opus": "sr",
                "mtdata": "srp",
                "sacrebleu": "sr",
                "flores101": "srp",
                "pontoon": "sr",
                "hplt": "srp_Cyrl",
                "newscrawl": "sr",
                "opuscleaner": "sr",
                "bicleaner": "hbs",
                "monocleaner": "sr",
                "fasttext": "srp_Cyrl",
                "comet22": "sr",
                "metricx24": "sr",
                "flores200-plus": "srp_Cyrl",
                "bouquet": "not supported",
                "wmt24pp": "sr_RS",
                "nllb": "srp_Cyrl",
                "google": "sr",
                "microsoft": "sr-Cyrl",
            },
        ),
    ],
)
def test_lang_code(code, expected):
    all = generate_all()

    assert all[code] == expected


def test_all_lang_codes_support_flores200():
    """
    Do not train languages without flores support as there is not data to evaluate
    """
    all = generate_all()

    for lang, data in all.items():
        if data["flores200-plus"] == "not supported":
            print(lang, data)
            assert False
