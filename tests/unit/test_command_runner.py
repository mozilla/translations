import pytest
from pipeline.common.command_runner import run_command_pipeline
from shlex import join


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
