import json
import os
import pathlib

import pytest
from fixtures import DataDir

from pipeline.clean.opuscleaner.generate_filters import Mode, generate


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_filters")


@pytest.mark.parametrize(
    "params",
    [
        # verify defaults
        ("en", "ru", "mtdata_Tilde-airbaltic-1-eng-rus", Mode.custom, "default.filters.json", []),
        (
            "fr",
            "en",
            "mtdata_Tilde-airbaltic-1-eng-rus",
            Mode.custom,
            "default.filters.json",
            ["currency_mismatch"],
        ),
        (
            "fr",
            "en",
            "opus_ELRC-3075-wikipedia_health/v1",
            Mode.custom,
            "default.filters.json",
            ["currency_mismatch"],
        ),
        # verify langauge specific config is used for any direction
        (
            "ru",
            "en",
            "opus_ELRC-3075-wikipedia_health/v1",
            Mode.custom,
            "ru-en/opus_ELRC-3075-wikipedia_health-v1.filters.json",
            [],
        ),
        # backward direction should have a separate custom config because it can be different (for example, zh)
        (
            "en",
            "ru",
            "opus_ELRC-3075-wikipedia_health/v1",
            Mode.custom,
            "default.filters.json",
            [],
        ),
        # verify dataset specific config is used for different language pairs
        ("ru", "en", "opus_ada83/v1", Mode.custom, "opus_ada83-v1.filters.json", []),
        (
            "fr",
            "en",
            "opus_ada83/v1",
            Mode.custom,
            "opus_ada83-v1.filters.json",
            ["currency_mismatch"],
        ),
        # verify the "defaults" mode always uses the default config
        (
            "ru",
            "en",
            "opus_ELRC-3075-wikipedia_health/v1",
            Mode.defaults,
            "default.filters.json",
            [],
        ),
        (
            "fr",
            "en",
            "opus_ada83/v1",
            Mode.defaults,
            "default.filters.json",
            ["currency_mismatch"],
        ),
        # make sure Chinese uses language level default configs
        ("zh", "en", "opus_ada83/v1", Mode.custom, "zh-en/default.filters.json", []),
        ("en", "zh", "opus_ada83/v1", Mode.custom, "en-zh/default.filters.json", []),
    ],
    ids=[
        "default-en-ru",
        "default-fr-en",
        "default-fr-en-elrc",
        "lang-ru-en",
        "lang-backward",
        "dataset-ru-en",
        "dataset-fr-en",
        "override-with-default-ru-en-elrc",
        "override-with-default-fr-en-unpc",
        "zh-en",
        "en-zh",
    ],
)
def test_generate_filters(params, data_dir):
    """
    Make sure the generated filters correspond to the right custom configs
    and the template values were replaced properly
    """
    src, trg, dataset, mode, config_path, extra_filters = params
    output_path = data_dir.join("output-config.json")
    config_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.joinpath(
        "pipeline", "clean", "opuscleaner", "configs", config_path
    )

    generate(dataset=dataset, output=output_path, src=src, trg=trg, mode=mode)

    with open(output_path, "r") as f_out:
        with open(config_path, "r") as f_conf:
            actual = json.load(f_out)
            expected = json.load(f_conf)
    # for some langauge pairs additional filters are generated dynamically
    assert len(actual["filters"]) == len(expected["filters"]) + len(extra_filters)
    assert {f["filter"] for f in actual["filters"]} == (
        {f["filter"] for f in expected["filters"]} | set(extra_filters)
    )
    # check languages in whitespace filters where there are two of them
    whitespace_filters = {
        f["language"] for f in actual["filters"] if f["filter"] == "normalize_whitespace"
    }
    if len(whitespace_filters) == 2:
        assert whitespace_filters == {
            src,
            trg,
        }
    # max length value is slightly changed in opus_ELRC-3075-wikipedia_health/v1 to verify that this is the same config
    assert [f for f in actual["filters"] if f["filter"] == "max_length"][0]["parameters"][
        "MAXLENGTH"
    ] == [f for f in expected["filters"] if f["filter"] == "max_length"][0]["parameters"][
        "MAXLENGTH"
    ]
    # alpha ratios are different for zh
    alpha_ratio_filters = [f for f in actual["filters"] if f["filter"] == "alpha_ratio"]
    if alpha_ratio_filters:
        assert (
            alpha_ratio_filters[0]["parameters"]["SRCWORDRAT"]
            == [f for f in expected["filters"] if f["filter"] == "alpha_ratio"][0]["parameters"][
                "SRCWORDRAT"
            ]
        )
