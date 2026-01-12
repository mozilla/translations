import pytest

from pipeline.langs.codes import (
    to_iso6391,
    to_iso6393,
    to_iso6393_individual_and_script,
    to_locale,
)


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
        ("ja", "jpn_Jpan"),
    ],
)
def test_to_iso6393_individual_and_script(source: str, expected: str):
    assert to_iso6393_individual_and_script(source) == expected


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
