import io
import json
import os
from unittest import mock

import pytest
import requests
import train_taskcluster
from tests.fixtures import DataDir

TRAIN_TASKCLUSTER_SH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "taskcluster",
        "scripts",
        "pipeline",
        "train-taskcluster.sh",
    )
)


@pytest.mark.parametrize(
    "args",
    (
        pytest.param(
            [
                "model_type",
                "training_type",
                "src",
                "trg",
                "train_set_prefix",
                "valid_set_prefix",
                "model_dir",
                "best_model_metric",
                "alignments",
                "seed",
                "teacher_mode",
                "student_model",
            ],
            id="required_only",
        ),
        pytest.param(
            [
                "model_type",
                "training_type",
                "src",
                "trg",
                "train_set_prefix",
                "valid_set_prefix",
                "model_dir",
                "best_model_metric",
                "alignments",
                "seed",
                "teacher_mode",
                "student_model",
                "pretrained_model_mode",
                "pretrained_model_type",
            ],
            id="with_pretrained_model",
        ),
        pytest.param(
            [
                "model_type",
                "training_type",
                "src",
                "trg",
                "train_set_prefix",
                "valid_set_prefix",
                "model_dir",
                "best_model_metric",
                "alignments",
                "seed",
                "teacher_mode",
                "student_model",
                "pretrained_model_mode",
                "pretrained_model_type",
                "--foo",
                "--bar",
            ],
            id="with_extra_params",
        ),
    ),
)
def test_all_args_forwarded(args):
    with mock.patch.multiple(
        train_taskcluster, subprocess=mock.DEFAULT, requests=mock.DEFAULT
    ) as tt_mock:
        tt_mock["subprocess"].run.return_value = mock.Mock(returncode=0)
        with mock.patch.dict(os.environ) as mocked_env:
            mocked_env["TASK_ID"] = "abcdef"
            mocked_env["RUN_ID"] = "0"
            mocked_env["TASKCLUSTER_ROOT_URL"] = "https://some.cluster"
            train_taskcluster.main(args)
            assert tt_mock["subprocess"].run.call_args_list == [
                mock.call([TRAIN_TASKCLUSTER_SH] + args, check=False),
            ]


FULL_ARTIFACTS = [
    {
        "storageType": "s3",
        "name": f"public/build/{artifact}",
        "expires": "2035-04-23T16:22:13.477Z",
        "contentType": "application/x-yaml",
    }
    # Note: we are depending on something under to test for construction of parts of
    # these tests. However, the value of duplicating this list into the tests instead
    # is debatable, as what we're _really_ testing is the logic around them, not
    # which ones are or are not required. It seems reasonable to keep a single source
    # of truth for this.
    for artifact in train_taskcluster.CONTINUATION_ARTIFACTS.union(
        {"vocab.src.spm", "vocab.trg.spm"}
    )
]
FULL_ARTIFACTS.append(
    {
        "storageType": "s3",
        "name": "public/logs/live.log",
        "expires": "2035-04-23T16:22:13.477Z",
        "contentType": "text/plain; charset=utf-8",
    },
)
PARTIAL_ARTIFACTS = [
    {
        "storageType": "s3",
        "name": "public/build/model.npz",
        "expires": "2035-04-23T16:22:13.477Z",
        "contentType": "application/x-yaml",
    },
    {
        "storageType": "s3",
        "name": "public/logs/live.log",
        "expires": "2035-04-23T16:22:13.477Z",
        "contentType": "text/plain; charset=utf-8",
    },
]


@pytest.mark.parametrize(
    "current_run_id,resumable_run_id,run_artifacts,artifact_response_code,orig_pretrained_model_mode,expected_pretrained_model_mode",
    (
        pytest.param(
            0,
            None,
            # not used unless resumable_run_id is set
            {},
            200,
            "",
            "",
            id="run_0_no_continuation",
        ),
        pytest.param(
            0,
            None,
            # not used unless resumable_run_id is set
            {},
            200,
            "init",
            "init",
            id="run_0_no_continuation_with_pretrained_model",
        ),
        # TODO: add some cases that test that pretrained model mode is preserved when not doing
        # autocontinuation
        # and also that it's overridden when expected
        pytest.param(
            1,
            0,
            {0: FULL_ARTIFACTS},
            200,
            "",
            "continue",
            id="run_1_continues_run_0",
        ),
        pytest.param(
            2,
            1,
            {1: FULL_ARTIFACTS},
            200,
            "",
            "continue",
            id="run_2_continues_run_1",
        ),
        pytest.param(
            2,
            0,
            {1: PARTIAL_ARTIFACTS, 0: FULL_ARTIFACTS},
            200,
            "",
            "continue",
            id="run_2_continues_run_0",
        ),
        pytest.param(
            3,
            1,
            {2: PARTIAL_ARTIFACTS, 1: FULL_ARTIFACTS, 0: PARTIAL_ARTIFACTS},
            200,
            "",
            "continue",
            id="run_3_continues_run_1",
        ),
        pytest.param(
            2,
            None,
            {1: PARTIAL_ARTIFACTS, 0: PARTIAL_ARTIFACTS},
            200,
            "",
            "",
            id="run_2_cant_continue_earlier_runs",
        ),
        pytest.param(
            2,
            None,
            {1: PARTIAL_ARTIFACTS, 0: PARTIAL_ARTIFACTS},
            200,
            "use",
            "use",
            id="run_2_cant_continue_earlier_runs_preserves_pretrained_model_mode",
        ),
        pytest.param(
            2,
            0,
            {1: PARTIAL_ARTIFACTS, 0: FULL_ARTIFACTS},
            404,
            "",
            "",
            id="artifacts_are_404",
        ),
    ),
)
def test_autocontinue(
    current_run_id,
    resumable_run_id,
    run_artifacts,
    artifact_response_code,
    orig_pretrained_model_mode,
    expected_pretrained_model_mode,
):
    with mock.patch.multiple(
        train_taskcluster, subprocess=mock.DEFAULT, requests=mock.DEFAULT
    ) as tt_mock:
        tt_mock["subprocess"].run.return_value = mock.Mock(returncode=0)
        with mock.patch.dict(os.environ) as mocked_env:
            # In production, these are set by the Taskcluster worker
            mocked_env["TASK_ID"] = "abcdef"
            mocked_env["RUN_ID"] = str(current_run_id)
            mocked_env["TASKCLUSTER_ROOT_URL"] = "https://some.cluster"

            def fake_get(url, *args, **kwargs):
                """Handles the expected requests to the Taskcluster API, and throws errors
                for any requests that shouldn't have been made."""

                resp = requests.Response()
                if url.endswith("artifacts"):
                    resp.status_code = 200
                    resp.headers = {"Content-Type": "application/json"}  # type:ignore
                    run_id = int(url.split("/runs/", 1)[1].split("/")[0])
                    resp._content = json.dumps(
                        {
                            "artifacts": run_artifacts[run_id],
                        }
                    ).encode("utf-8")
                elif any(
                    [
                        url.endswith(artifact)
                        for artifact in train_taskcluster.CONTINUATION_ARTIFACTS.union(
                            {"vocab.src.spm", "vocab.trg.spm"}
                        )
                    ]
                ):
                    # No action needed here; we will check that the right calls were
                    # made based on the current_run_id later.
                    resp.status_code = artifact_response_code
                    if resp.status_code == 200:
                        resp._content = b""
                        resp.raw = io.StringIO("")
                elif url.endswith("live.log") or url.endswith("live_backing.log"):
                    resp.status_code = 400
                    resp._content = (
                        f"train_taskcluster.py wrongly tried to download a task log: {url}"
                    )  # type:ignore
                else:
                    resp.status_code = 400
                    resp._content = (
                        f"train_taskcluster.py made a call to an unexpected URL: {url}"
                    )  # type:ignore

                return resp

            tt_mock["requests"].get = mock.Mock(wraps=fake_get)

            model_dir = DataDir("test_train_taskcluster").path
            train_taskcluster.main(
                [
                    "model_type",
                    "training_type",
                    "src",
                    "trg",
                    "train-set-prefix",
                    "valid-set-prefix",
                    model_dir,
                    "best-model-metric",
                    "alignments",
                    "seed",
                    "teacher_mode",
                    "student_model",
                    orig_pretrained_model_mode,
                ]
            )

            # The calls we're expecting are different when we resume vs. when we don't.
            # There is some overlap, but for clarity it's much simpler just to separate
            # these cases into different branches, at the cost of a small amount of
            # repetition.
            if resumable_run_id is not None:
                calls = []
                prev_run_id = current_run_id - 1
                while prev_run_id >= resumable_run_id:
                    # For each previous run until we reach a resumable one, we expect
                    # to fetch the artifact list.
                    calls.append(
                        mock.call(
                            f"https://some.cluster/api/queue/v1/task/abcdef/runs/{prev_run_id}/artifacts"
                        )
                    )

                    # However, we only expect to fetch the artifacts for the run we resume from...
                    if prev_run_id == resumable_run_id:
                        i = 0
                        for artifact in run_artifacts[prev_run_id]:
                            # ...but even then, we don't expect to download the Taskcluster logs
                            if not artifact["name"].startswith("public/logs"):
                                # or anything after the first artifact if the response code is not 200
                                if artifact_response_code == 200 or i == 0:
                                    i += 1
                                    calls.append(
                                        mock.call(
                                            f"https://some.cluster/api/queue/v1/task/abcdef/runs/{prev_run_id}/artifacts/{artifact['name']}",
                                            stream=True,
                                        ),
                                    )
                    prev_run_id = prev_run_id - 1

                assert tt_mock["requests"].get.call_args_list == calls
            else:
                # We are not continuing training - if there are earlier runs we should just see
                # calls to fetch a list of their artifacts.
                calls = []
                prev_run_id = current_run_id - 1
                while prev_run_id >= 0:
                    calls.append(
                        mock.call(
                            f"https://some.cluster/api/queue/v1/task/abcdef/runs/{prev_run_id}/artifacts"
                        )
                    )
                    prev_run_id -= 1

                assert tt_mock["requests"].get.call_args_list == calls

            assert tt_mock["subprocess"].run.call_count == 1
            # pretrained model mode is the 13th arg to the training script, but subprocess
            # is also given the script name - so we look for the expected pretrained model mode
            # in the 13th arg of the subprocess.run call
            assert (
                tt_mock["subprocess"].run.call_args_list[0][0][0][13]
                == expected_pretrained_model_mode
            )
