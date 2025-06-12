import os

from tests.fixtures import DataDir, en_sample, ru_sample


def test_analyze_mono():
    data_dir = DataDir("test_analyze_mono")

    data_dir.mkdir("artifacts")
    data_dir.create_zst("news_2020.en.zst", en_sample)

    data_dir.run_task(
        "corpus-analyze-mono-news-crawl-en-news_2020",
    )

    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts/news_2020.en.distribution-words.png"))
    assert os.path.isfile(data_dir.join("artifacts/news_2020.en.distribution-codepoints.png"))


def test_analyze_corpus():
    data_dir = DataDir("test_analyze_corpus")

    data_dir.mkdir("artifacts")
    data_dir.create_zst("Books_v1.en.zst", en_sample)
    data_dir.create_zst("Books_v1.ru.zst", ru_sample)

    data_dir.run_task("corpus-analyze-parallel-opus-Books_v1-en-ru")

    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts/Books_v1.en.distribution-words.png"))
    assert os.path.isfile(data_dir.join("artifacts/Books_v1.en.distribution-codepoints.png"))
    assert os.path.isfile(data_dir.join("artifacts/Books_v1.ru.distribution-words.png"))
    assert os.path.isfile(data_dir.join("artifacts/Books_v1.ru.distribution-codepoints.png"))
