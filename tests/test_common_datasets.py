import io
import logging
from pathlib import Path
from typing import Iterator

import pytest
from fixtures import DataDir

from pipeline.common.logging import get_logger
from pipeline.common.datasets import (
    WeakStringSet,
    WeakStringDict,
    compress,
    decompress,
    shuffle_in_temp_files,
    shuffle_with_max_lines,
)
from pipeline.common.downloads import read_lines, write_lines

ITEMS = 100_000
# ITEMS = 1_000
PERCENTAGE = 0.2
MAX_LINES = int(ITEMS * PERCENTAGE)

line_fixtures = [
    "line 1\n",
    "line 2\n",
    "line 3\n",
    "line 4\n",
    "line 5\n",
]
line_fixtures_bytes = "".join(line_fixtures).encode("utf-8")


def write_test_content(output_path: str) -> str:
    with write_lines(output_path) as outfile:
        for line in line_fixtures:
            outfile.write(line)
    return output_path


def get_total_byte_size(lines: list[str]) -> int:
    total_byte_size = 0
    for line in lines:
        total_byte_size = total_byte_size + len(line.encode())
    return total_byte_size


def compute_distribution(lines: Iterator[str], items=ITEMS, max_lines=MAX_LINES) -> list[float]:
    """
    Computes a histogram (list of 10 items) with a percentage value of 0.0 - 100.0 for each item.
    """
    histogram = [0] * 10
    for line in lines:
        # This assumes the content will be a tab separated list, with the first item to be the
        # initial sorted order in the list.
        key = int(int(line.split("\t")[0]) * 10 / items)
        histogram[key] = histogram[key] + (1 / max_lines)

    # Lower the precision of the ints.
    return [round(value * 1000) / 1000 for value in histogram]


# Test the distributions of the different types of datasets. This shuffler estimates the content
# size as it iterates through the line stream.
shuffle_params = [
    (
        # Each line is the same bytes as the next line. This should create an even distribution.
        # [
        #     "000000000 000000000 000000000 ... 000000000",
        #     "000000001 000000001 000000001 ... 000000001",
        #     ...
        # ]
        "even-distribution",
        [f"{line:09d}\t" * 10 for line in range(ITEMS)],
        [0.102, 0.101, 0.099, 0.1, 0.1, 0.102, 0.099, 0.097, 0.1, 0.1],
    ),
    (
        # The initial lines are low in byte count, and gradually increase. In this case there
        # will be a bias to over-sample the the initial items, but it will eventually even out as
        # more bytes are read in and the average spreads out.
        # [
        #     "0 0 0 ... 0",
        #     "1 1 1 ... 1",
        #     ...
        #     "99997 99997 99997 ... 99997",
        #     "99998 99998 99998 ... 99998",
        #     "99999 99999 99999 ... 99999",
        # ]
        "small-content-at-start",
        [f"{line}\t" * 10 for line in range(ITEMS)],
        [0.114, 0.116, 0.092, 0.095, 0.096, 0.099, 0.097, 0.095, 0.098, 0.098],
        # |     |     |
        # |     |     ^ Lower sample rate.
        # ^^^^^^^ Higher sampling rate.
    ),
    (
        # [
        #     "99999 99999 99999 ... 99999",
        #     "99998 99998 99998 ... 99998",
        #     "99997 99997 99997 ... 99997",
        #     ...
        #     "1 1 1 ... 1",
        #     "0 0 0 ... 0",
        # ]
        "large-content-at-start",
        [f"{line}\t" * 10 for line in range(ITEMS)][::-1],
        [0.101, 0.102, 0.099, 0.102, 0.103, 0.102, 0.101, 0.102, 0.102, 0.086],
        #                                             lower sample rate ^^^^^
    ),
]


@pytest.mark.parametrize("params", shuffle_params, ids=[d[0] for d in shuffle_params])
def test_shuffle_with_max_lines(params):
    description, line_stream, histograph = params
    # [
    #     "0000 0000 0000 ... 0000",
    #     "0001 0001 0001 ... 0001",
    #     "0002 0002 0002 ... 0002",
    #     ...
    # ]

    output = shuffle_with_max_lines(
        line_stream,
        seed="test",
        max_lines=MAX_LINES,
        total_byte_size=get_total_byte_size(line_stream),
    )

    assert compute_distribution(output) == histograph, description


def test_shuffle_in_temp_files():
    # [
    #     "0000 0000 0000 ... 0000",
    #     "0001 0001 0001 ... 0001",
    #     "0002 0002 0002 ... 0002",
    #     ...
    # ]
    line_stream = [f"{line:09d}\t" * 10 for line in range(ITEMS)]

    # Total byte size is ~10_000_000
    chunk_bytes = 100_000
    bucket_bytes = 2_000_000
    data_dir = DataDir("test_common_datasets")

    with io.StringIO() as output:
        shuffle_in_temp_files(
            line_stream,
            output=output,
            seed="test",
            chunk_bytes=chunk_bytes,
            bucket_bytes=bucket_bytes,
            chunk_dir=data_dir.path,
            keep_chunks=True,
        )

        data_dir.print_tree()

        output.seek(0)
        text = output.read()
        lines = [*text.splitlines()]
        sample = lines[:MAX_LINES]

        output.seek(0)
        with open(data_dir.join("shuffle.txt"), "w") as file:
            print(output.getvalue(), file=file)

        assert len(lines) == ITEMS
        assert compute_distribution(sample) == [
            0.149,
            0.258,
            0.04,
            0.1,
            0.001,  # The distribution is not perfect with this strategy.
            0.052,
            0.05,
            0.1,
            0.101,
            0.15,
        ]


def test_weak_string_set():
    """
    Test all of the Set operations that take an "elem" per:
    https://docs.python.org/3/library/stdtypes.html#set
    """
    unique_strings = WeakStringSet()
    unique_strings.add("string a")
    unique_strings.add("string b")

    assert "string a" in unique_strings
    assert "string b" in unique_strings
    assert "string c" not in unique_strings
    assert len(unique_strings) == 2

    unique_strings.update(
        [
            "string d",
            "string e",
        ]
    )
    assert "string d" in unique_strings
    assert "string e" in unique_strings
    assert "string f" not in unique_strings

    unique_strings.remove("string d")
    assert "string d" not in unique_strings

    unique_strings.discard("string e")
    assert "string e" not in unique_strings

    unique_strings2 = WeakStringSet(["string a", "string b"])
    assert "string a" in unique_strings2
    assert "string b" in unique_strings2
    assert "string c" not in unique_strings2
    assert len(unique_strings2) == 2


def test_weak_string_dict():
    unique_strings_scores = WeakStringDict()
    unique_strings_scores["aa"] = 0.87
    unique_strings_scores["aa"] = 0.92
    unique_strings_scores["ab"] = 4.1

    assert "aa" in unique_strings_scores
    assert "ab" in unique_strings_scores
    assert unique_strings_scores["aa"] == 0.92
    assert unique_strings_scores["aa"] != 0.87

    del unique_strings_scores["aa"]
    assert "aa" not in unique_strings_scores

    assert len(unique_strings_scores) == 1

    unique_strings_scores["cdf"] = 33.2
    assert "cdf" in unique_strings_scores
    assert unique_strings_scores["cdf"] == 33.2
    unique_strings_scores["aa"] = 0.33
    unique_strings_scores["ab"] = 0.34
    unique_strings_scores["aa"] = 0.01
    assert unique_strings_scores["aa"] == 0.01


@pytest.mark.parametrize("suffix", ["zst", "gz"])
@pytest.mark.parametrize("remove_or_keep", ["remove", "keep"])
def test_compress(suffix: str, remove_or_keep: str):
    data_dir = DataDir("test_common_datasets")
    source = Path(data_dir.join("lines.txt"))
    destination = Path(data_dir.join(f"lines.txt.{suffix}"))
    logger = get_logger(__file__)
    logger.setLevel(logging.INFO)

    write_test_content(source)
    assert source.exists()
    assert not destination.exists()

    with read_lines(source) as lines:
        assert list(lines) == line_fixtures

    remove = remove_or_keep == "remove"
    compress(source, destination, logger=logger, remove=remove)

    if remove:
        assert not source.exists(), "The source file was removed."
    else:
        assert source.exists(), "The source file was kept."

    with read_lines(destination) as lines:
        assert list(lines) == line_fixtures


@pytest.mark.parametrize("suffix", ["zst", "gz"])
@pytest.mark.parametrize("remove_or_keep", ["remove", "keep"])
def test_decompress(suffix: str, remove_or_keep: str):
    data_dir = DataDir("test_common_datasets")
    source = Path(data_dir.join(f"lines.txt.{suffix}"))
    destination = Path(data_dir.join("lines.txt"))
    logger = get_logger(__file__)
    logger.setLevel(logging.INFO)

    write_test_content(source)
    assert source.exists()
    assert not destination.exists()

    with read_lines(source) as lines:
        assert list(lines) == line_fixtures

    remove = remove_or_keep == "remove"
    decompress(source, destination, remove=remove, logger=logger)

    if remove:
        assert not source.exists(), "The source file was removed."
    else:
        assert source.exists(), "The source file was kept."

    assert destination.exists()

    with read_lines(destination) as lines:
        assert list(lines) == line_fixtures
