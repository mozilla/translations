import pytest

from pipeline.data.cjk import ChineseType, ChineseConverter
from fixtures import DataDir

chinese_traditional = "中文簡繁轉換開源項目，支持詞彙級別的轉換"
chinese_simplified = "中文简繁转换开源项目，支持词汇级别的转换"
non_chinese = " hello world "


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_cjk")


@pytest.mark.parametrize(
    "text,expected,type",
    [
        (chinese_simplified, chinese_traditional, ChineseType.traditional),
        (chinese_traditional, chinese_simplified, ChineseType.simplified),
        (
            chinese_traditional + non_chinese,
            chinese_simplified + non_chinese,
            ChineseType.simplified,
        ),
        (
            chinese_simplified + non_chinese,
            chinese_traditional + non_chinese,
            ChineseType.traditional,
        ),
        (
            chinese_traditional + chinese_simplified,
            chinese_traditional + chinese_traditional,
            ChineseType.traditional,
        ),
        (
            chinese_traditional + chinese_simplified,
            chinese_simplified + chinese_simplified,
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
    in_path = data_dir.create_file("cjk_test_in.txt", text)
    all_text = text + "\n" + text + "\n" + text
    with open(in_path, "w") as f:
        f.write(all_text)
    out_path = data_dir.join("cjk_test_out.txt")
    converter = ChineseConverter()

    stats = converter.convert_file(in_path, out_path, type)

    with open(out_path, "r") as f:
        out_text = f.read()
    assert out_text
    assert stats.script_conversion.visited == 3
    assert stats.script_conversion.converted == (3 if all_text != out_text else 0)
    out_texts = out_text.split("\n")
    assert len(out_texts) == 3
    assert out_texts[0] == out_texts[1] == out_texts[2]
    assert out_texts[0] == expected


@pytest.mark.parametrize(
    "text,expected,type",
    [
        (chinese_simplified, chinese_simplified, ChineseType.simplified),
        (chinese_traditional, chinese_traditional, ChineseType.traditional),
        (chinese_simplified, "", ChineseType.traditional),
        (chinese_traditional, "", ChineseType.simplified),
        (
            chinese_traditional + non_chinese,
            "",
            ChineseType.simplified,
        ),
        (
            chinese_simplified + non_chinese,
            "",
            ChineseType.traditional,
        ),
        (
            chinese_traditional + chinese_simplified,
            "",
            ChineseType.traditional,
        ),
        (
            chinese_traditional + chinese_simplified,
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
def test_filter_file(text: str, expected: str, type: ChineseType, data_dir: DataDir):
    in_path = data_dir.create_file("cjk_test_in.txt", text)
    all_text = text + "\n" + text + "\n" + text
    with open(in_path, "w") as f:
        f.write(all_text)
    out_path = data_dir.join("cjk_test_out.txt")
    converter = ChineseConverter()

    stats = converter.filter_file(in_path, out_path, type)

    with open(out_path, "r") as f:
        out_text = f.read()
    assert stats.script_conversion.visited == 3
    assert stats.script_conversion.converted == 0
    out_texts = out_text.split("\n")
    assert out_texts[0] == expected
    if len(out_texts) > 1:
        assert out_texts[0] == out_texts[1] == out_texts[2]
        assert stats.script_conversion.filtered == 0
    else:
        assert stats.script_conversion.filtered == 3
