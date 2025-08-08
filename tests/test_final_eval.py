from fixtures import DataDir
from pathlib import Path
import os
import pytest

ROOT_DIR = Path(__file__).parent.parent


def test_final_eval():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("This test requires the OPENAI_API_KEY to be set.")
        return

    data_dir = DataDir("test_final_eval")
    data_dir.run_task(
        "final-eval-llm",
        config=str(ROOT_DIR / "taskcluster/configs/eval.yml"),
        callback="evaluate",
        extra_args=["--api_batch_size", "4", "--max_count", "8"],
    )

    data_dir.print_tree()

    data_dir.assert_files(
        [
            "artifacts/scores.json",
            "artifacts/summary.json",
        ]
    )
