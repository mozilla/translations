import pytest
from sacremoses import MosesTokenizer

from pipeline.alignments.align import map_indices

tokenizer = MosesTokenizer("en")


@pytest.mark.parametrize(
    "orig, expected_idx_map",
    [
        ("Hi", {0: 0}),
        ("Hello, world!", {0: 0, 1: 0, 2: 1, 3: 1}),
        ("Hello,  world!", {0: 0, 1: 0, 2: 1, 3: 1}),
        ("Hello,  half-world and welcome!", {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}),
        ("Hello - world!", {0: 0, 1: 1, 2: 2, 3: 2}),
        ("Hello,- world!", {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}),
        (
            "“I will not,” retorted the Witch, “for it is now my shoe, and not yours.”",
            {
                0: 0,
                1: 0,
                2: 1,
                3: 2,
                4: 2,
                5: 2,
                6: 3,
                7: 4,
                8: 5,
                9: 5,
                10: 6,
                11: 6,
                12: 7,
                13: 8,
                14: 9,
                15: 10,
                16: 11,
                17: 11,
                18: 12,
                19: 13,
                20: 14,
                21: 14,
                22: 14,
            },
        ),
    ],
)
def test_remap_indices(orig, expected_idx_map):
    """
    Test mapping word indices of Moses tokenized text to whitespace tokenized ones
    """
    tokenized = tokenizer.tokenize(orig)
    tokenized_str = " ".join(tokenized)
    print(tokenized_str)

    idx_map = map_indices(tokenized_str, orig)

    assert idx_map == expected_idx_map
