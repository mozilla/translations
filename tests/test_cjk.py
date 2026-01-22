import json
from pathlib import Path

import pytest

from pipeline.data.cjk import ChineseType, handle_chinese_mono, handle_chinese_parallel
from fixtures import DataDir
from pipeline.langs.codes import LangCode

traditional = "中文簡繁轉換開源項目，支持詞彙級別的轉換"
simplified = "中文简繁转换开源项目，支持词汇级别的转换"
non_chinese = "hello world"


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_cjk")


@pytest.mark.parametrize(
    "text,expected,type",
    [
        (simplified, traditional, ChineseType.traditional),
        (traditional, simplified, ChineseType.simplified),
        (
            traditional + non_chinese,
            simplified + non_chinese,
            ChineseType.simplified,
        ),
        (
            simplified + non_chinese,
            traditional + non_chinese,
            ChineseType.traditional,
        ),
        (
            traditional + simplified,
            traditional + traditional,
            ChineseType.traditional,
        ),
        (
            traditional + simplified,
            simplified + simplified,
            ChineseType.simplified,
        ),
        (non_chinese, non_chinese, ChineseType.traditional),
        (non_chinese, non_chinese, ChineseType.simplified),
    ],
    ids=[
        "s2t",
        "t2s",
        "t2s_with_english",
        "s2t_with_english",
        "s2t_mixed",
        "t2s_mixed",
        "s2t_english",
        "t2s_english",
    ],
)
def test_convert_file(text: str, expected: str, type: ChineseType, data_dir: DataDir):
    all_text = text + "\n" + text + "\n" + text
    path = data_dir.create_zst("cjk_test.txt.zst", all_text)

    handle_chinese_mono(Path(path), is_src=True, language_code=LangCode("zh"), variant=type)

    out_text = data_dir.read_text(path)
    stats = json.loads(data_dir.read_text(data_dir.join("cjk_test.txt.converted.stats.json")))
    assert out_text
    assert stats["script_conversion"]["visited"] == 3
    assert stats["script_conversion"]["converted"] == (3 if all_text != out_text else 0)
    out_texts = out_text.split("\n")
    assert len(out_texts) == 3
    assert out_texts[0] == out_texts[1] == out_texts[2]
    assert out_texts[0] == expected


@pytest.mark.parametrize(
    "text,expected,type",
    [
        (simplified, simplified, ChineseType.simplified),
        (traditional, traditional, ChineseType.traditional),
        (simplified, "", ChineseType.traditional),
        (traditional, "", ChineseType.simplified),
        (
            traditional + non_chinese,
            "",
            ChineseType.simplified,
        ),
        (
            simplified + non_chinese,
            "",
            ChineseType.traditional,
        ),
        (
            traditional + simplified,
            "",
            ChineseType.traditional,
        ),
        (
            traditional + simplified,
            "",
            ChineseType.simplified,
        ),
        (non_chinese, "", ChineseType.traditional),
        (non_chinese, "", ChineseType.simplified),
    ],
    ids=[
        "s2s",
        "t2t",
        "s2t",
        "t2s",
        "t2s_with_english",
        "s2t_with_english",
        "s2t_mixed",
        "t2s_mixed",
        "s2t_english",
        "t2s_english",
    ],
)
def test_filter_file_variants(text: str, expected: str, type: ChineseType, data_dir: DataDir):
    all_text = text + "\n" + text + "\n" + text
    path = data_dir.create_zst("cjk_test.txt.zst", all_text)

    handle_chinese_mono(Path(path), is_src=False, language_code=LangCode("zh"), variant=type)

    out_text = data_dir.read_text(path)
    stats = json.loads(data_dir.read_text(data_dir.join("cjk_test.txt.converted.stats.json")))
    assert stats["script_conversion"]["visited"] == 3
    assert stats["script_conversion"]["converted"] == 0
    out_texts = out_text.split("\n")
    assert out_texts[0] == expected
    if len(out_texts) > 1:
        assert out_texts[0] == out_texts[1] == out_texts[2]
        assert stats["script_conversion"]["filtered"] == 0
    else:
        assert stats["script_conversion"]["filtered"] == 3


def test_filter_file_mixed(data_dir: DataDir):
    text = "\n".join([simplified, traditional, simplified, traditional, simplified])
    path = data_dir.create_zst("cjk_test.txt.zst", text)

    handle_chinese_mono(
        Path(path), is_src=False, language_code=LangCode("zh"), variant=ChineseType.simplified
    )

    out_text = data_dir.read_text(path)
    stats = json.loads(data_dir.read_text(data_dir.join("cjk_test.txt.converted.stats.json")))
    assert stats["script_conversion"]["visited"] == 5
    assert stats["script_conversion"]["converted"] == 0
    assert stats["script_conversion"]["filtered"] == 2
    out_texts = out_text.strip().split("\n")
    assert len(out_texts) == 3
    assert out_texts[0] == out_texts[1] == simplified


def test_filter_parallel_trg(data_dir: DataDir):
    """
    Test filtering parallel corpus when Chinese is a target language
    """
    text_zh = "\n".join([simplified, traditional, simplified, traditional, simplified])
    texts_en = [non_chinese + str(i) for i in range(5)]
    text_en = "\n".join(texts_en)
    path_zh = data_dir.create_zst("cjk_test.zh.zst", text_zh)
    path_en = data_dir.create_zst("cjk_test.en.zst", text_en)

    handle_chinese_parallel(
        output_prefix=data_dir.join("cjk_test"),
        src=LangCode("en"),
        trg=LangCode("zh"),
        variant=ChineseType.simplified,
    )

    out_text_zh = data_dir.read_text(path_zh)
    out_text_en = data_dir.read_text(path_en)
    stats = json.loads(data_dir.read_text(data_dir.join("cjk_test.filtered.zh.stats.json")))
    assert stats["script_conversion"]["visited"] == 5
    assert stats["script_conversion"]["converted"] == 0
    assert stats["script_conversion"]["filtered"] == 2
    out_texts_zh = out_text_zh.strip().split("\n")
    assert len(out_texts_zh) == 3
    assert all(txt == simplified for txt in out_texts_zh)
    out_texts_en = out_text_en.strip().split("\n")
    assert len(out_texts_en) == 3
    assert out_texts_en[0] == texts_en[0]
    assert out_texts_en[1] == texts_en[2]
    assert out_texts_en[2] == texts_en[4]


def test_filter_parallel_src(data_dir: DataDir):
    """
    Test converting a parallel corpus when Chinese is a source language
    """
    text_zh = "\n".join([simplified, traditional, simplified, traditional, simplified])
    texts_en = [non_chinese + str(i) for i in range(5)]
    text_en = "\n".join(texts_en)
    path_zh = data_dir.create_zst("cjk_test.zh.zst", text_zh)
    path_en = data_dir.create_zst("cjk_test.en.zst", text_en)

    handle_chinese_parallel(
        output_prefix=data_dir.join("cjk_test"),
        src=LangCode("zh"),
        trg=LangCode("en"),
        variant=ChineseType.simplified,
    )

    out_text_zh = data_dir.read_text(path_zh)
    out_text_en = data_dir.read_text(path_en)
    stats = json.loads(data_dir.read_text(data_dir.join("cjk_test.converted.zh.stats.json")))
    assert stats["script_conversion"]["visited"] == 5
    assert stats["script_conversion"]["converted"] == 2
    assert stats["script_conversion"]["filtered"] == 0
    out_texts_zh = out_text_zh.strip().split("\n")
    assert len(out_texts_zh) == 5
    assert all(txt == simplified for txt in out_texts_zh)
    out_texts_en = out_text_en.strip().split("\n")
    assert len(out_texts_en) == 5
    assert texts_en == out_texts_en
