import json

import pytest
from fixtures import DataDir

from pipeline.common.downloads import read_lines

corpus_sample = """CORPUS 1
CORPUS 2
CORPUS 3
CORPUS_SHARED 1
CORPUS_SHARED 2
CORPUS 4
CORPUS 5
"""

news_2014_sample = """NEWS_2014 1
NEWS_2014 2
MONO_SHARED 1
MONO_SHARED 2
NEWS_2014 3
CORPUS_SHARED 1
NEWS_2014 4
"""

nllb_sample = """NLLB 1
NLLB 2
MONO_SHARED 1
MONO_SHARED 2
NLLB 3
CORPUS_SHARED 2
NLLB 4
"""


@pytest.mark.parametrize("task", ["src-en", "trg-ru"])
def test_merge_mono(task: str):
    side, locale = task.split("-")
    sample_size = 5
    data_dir = DataDir("test_merge_mono")
    data_dir.mkdir("corpus")
    data_dir.create_zst(f"corpus/corpus.{locale}.zst", corpus_sample)
    data_dir.create_zst(f"news_2014.{locale}.zst", news_2014_sample)
    data_dir.create_zst(f"nllb.{locale}.zst", nllb_sample)
    data_dir.run_task(
        f"corpus-merge-mono-{side}-{locale}",
        env={"TEST_ARTIFACTS": data_dir.path},
        extra_args=["--sample_size", f"{sample_size}"],
    )

    data_dir.print_tree()

    assert json.loads(data_dir.read_text(f"artifacts/mono.{locale}.stats.json")) == {
        "final_truncated_monolingual_lines": {
            "description": "After truncation via the config's `experiment.mono-max-sentences-src.total`, how many lines are left.",
            "value": 10,
        },
        "final_truncated_monolingual_codepoints": {
            "description": "The amount of codepoints in the final monolingual corpus.",
            "value": 104,
        },
        "parallel_corpus_lines": {
            "description": "The size of the merged parallel corpus before truncation.",
            "value": 7,
        },
        "duplicates_of_parallel_corpus": {
            "description": "How much of the monolingual data was duplicated in the merged parallel corpus.",
            "value": 2,
        },
        "duplicates_of_monolingual_corpus": {
            "description": "How much of the monolingual data was duplicated across the monolingual datasets.",
            "value": 2,
        },
        "deduplicated_size": {
            "description": "What was the size of the monolingual data and how much was deduplicated. This doesn't count the truncation of datasets at the datasets gathering time.",
            "filtered": 4,
            "kept": 10,
            "visited": 14,
        },
        "deduplicated_monolingual_lines": {
            "description": "After deduplication, how much monolingual data is left.",
            "value": 10,
        },
    }

    with read_lines(data_dir.join(f"artifacts/mono.{locale}.zst")) as lines_iter:
        mono_lines = list(lines_iter)
        mono_lines_sorted = list(mono_lines)
        mono_lines_sorted.sort()

        assert mono_lines_sorted == [
            "MONO_SHARED 1\n",
            "MONO_SHARED 2\n",
            "NEWS_2014 1\n",
            "NEWS_2014 2\n",
            "NEWS_2014 3\n",
            "NEWS_2014 4\n",
            "NLLB 1\n",
            "NLLB 2\n",
            "NLLB 3\n",
            "NLLB 4\n",
        ]

        assert mono_lines != mono_lines_sorted, "The results are shuffled."

    with read_lines(
        data_dir.join(f"artifacts/mono.{locale}.sample.txt"), encoding="utf-8-sig"
    ) as lines_iter:
        samples = list(lines_iter)
        assert len(samples) == sample_size, "There are the expected number of samples"
        assert len(set(samples)) == sample_size, "All of the samples are are unique."
        for sample in samples:
            assert sample in mono_lines, "The sample is in the merged mono corpus"

        assert mono_lines != mono_lines_sorted, "The results are shuffled."
