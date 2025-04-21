from textwrap import dedent
import pytest
from pipeline.common.command_runner import run_command_pipeline
from shlex import join
from unittest.mock import patch


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize(
    "test_case",
    [
        ([["echo", "hello"]], "hello\n"),
        (
            [
                ["echo", "hello\nworld\nhis is a test"],
                ["grep", "world"],
            ],
            "world\n",
        ),
        (
            [
                ["echo", "hello world 1\njust hello\nhello world 2\njust world"],
                ["grep", "hello"],
                ["grep", "world"],
            ],
            "hello world 1\nhello world 2\n",
        ),
    ],
)
def test_run_pipeline(capture: bool, test_case, capfd):
    commands, expected_result = test_case

    command_text = join(commands[0])
    for command in commands[1:]:
        command_text = f"{command_text} | {join(command)}"

    actual_result = run_command_pipeline(commands, capture=capture)
    if not capture:
        captured = capfd.readouterr()
        actual_result = captured.out
    assert actual_result == expected_result


def test_run_pipeline_curand_failure():
    """
    This is a unit test to ensure that custom error code propagation works. This is
    required for task restarting logic in Taskcluster.
    """
    with patch("sys.exit") as mock_exit:
        text = dedent(
            """
                Error: Curand error 203 - /builds/worker/fetches/marian-source/src/tensors/rand.cpp:74: curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT)
                Error: Aborted from marian::CurandRandomGenerator::CurandRandomGenerator(size_t, marian::DeviceId) in /builds/worker/fetches/marian-source/src/tensors/rand.cpp:74

                [CALL STACK]
                [0x55f3b0dd021f]    marian::CurandRandomGenerator::  CurandRandomGenerator  (unsigned long,  marian::DeviceId) + 0x83f
                [0x55f3b0dd08b9]    marian::  createRandomGenerator  (unsigned long,  marian::DeviceId) + 0x69
                [0x55f3b0dc9fb0]    marian::  BackendByDeviceId  (marian::DeviceId,  unsigned long) + 0xa0
                [0x55f3b0796310]    marian::ExpressionGraph::  setDevice  (marian::DeviceId,  std::shared_ptr<marian::Device>) + 0x80
                [0x55f3b0b84c15]    marian::GraphGroup::  initGraphsAndOpts  ()        + 0x1e5
                [0x55f3b0b85fa0]    marian::GraphGroup::  GraphGroup  (std::shared_ptr<marian::Options>,  std::shared_ptr<marian::IMPIWrapper>) + 0x570
                [0x55f3b0b5d123]    marian::SyncGraphGroup::  SyncGraphGroup  (std::shared_ptr<marian::Options>,  std::shared_ptr<marian::IMPIWrapper>) + 0x83
                [0x55f3b0588fb3]    marian::Train<marian::SyncGraphGroup>::  run  ()   + 0x1b53
                [0x55f3b04aea6c]    mainTrainer  (int,  char**)                        + 0x15c
                [0x7f62f55d4d90]                                                       + 0x29d90
                [0x7f62f55d4e40]    __libc_start_main                                  + 0x80
                [0x55f3b04a94c5]    _start                                             + 0x25
            """
        )
        run_command_pipeline([["echo", text]])
        mock_exit.assert_called_once_with(203)
