import pytest
from pipeline.common.marian import marian_args_to_dict


@pytest.mark.parametrize(
    "marian_args,dict_value",
    [
        #
        (["--input", "file.txt"], {"input": "file.txt"}),
        (["--vocab", "en.spm", "fr.spm"], {"vocab": ["en.spm", "fr.spm"]}),
        (
            ["--", "--input", "file.in", "--output", "file.out"],
            {"input": "file.in", "output": "file.out"},
        ),
    ],
)
def test_marian_args_to_dict(marian_args: list[str], dict_value: dict):
    assert marian_args_to_dict(marian_args) == dict_value
