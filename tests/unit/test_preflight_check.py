import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

from utils.preflight_check import main as preflight_check


def get_preflight_check_output(*args):
    f = io.StringIO()
    opened_url: Optional[str] = None

    # Capture URLs that were opened.
    def open_in_browser(url):
        nonlocal opened_url
        opened_url = url

    config = (Path(__file__).resolve().parents[1] / "fixtures" / "config.pytest.yml").as_posix()

    with redirect_stdout(f):
        preflight_check([*args, "--config", config], open_in_browser)

    return f.getvalue(), opened_url


def test_artifacts():
    output, opened_url = get_preflight_check_output("--only", "artifacts")
    assert "Artifacts" in output
    assert "Task Commands" not in output
    assert "Training config" not in output
    assert "Visualization" not in output

    assert "artifacts/parameters.yml" in output
    assert not opened_url


def test_task_group():
    output, opened_url = get_preflight_check_output("--only", "task_group")
    assert "Artifacts" not in output
    assert "Task Commands" in output
    assert "Training config" not in output
    assert "Visualization" not in output

    assert "all-pipeline-en-ru-1" in output
    assert not opened_url


def test_task_graph():
    output, opened_url = get_preflight_check_output("--only", "graph")
    assert "Artifacts" not in output
    assert "Task Commands" not in output
    assert "Training config" not in output
    assert "Visualization" in output

    assert "https://gregtatum.github.io/taskcluster-tools/" in output
    assert not opened_url


def test_task_graph_open():
    output, opened_url = get_preflight_check_output("--only", "graph", "--open_graph")
    assert "Artifacts" not in output
    assert "Task Commands" not in output
    assert "Training config" not in output
    assert "Visualization" in output

    assert "The taskgraph structure was opened in TaskCluster tools" in output
    assert (
        "https://gregtatum.github.io/taskcluster-tools/?taskGraph=http%3A//localhost%3A"
        in opened_url
    )
