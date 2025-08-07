from copy import deepcopy

from taskgraph.taskgraph import TaskGraph

from translations_taskgraph.parameters import get_ci_training_config

PARAMS = deepcopy(get_ci_training_config())
PARAMS["target_tasks_method"] = "train-target-tasks"
PARAMS["training_config"]["continuation"]["models"] = {
    "backwards": {
        "mode": "use",
        "type": "default",
        "url": "https://storage.googleapis.com/releng-translations-dev/models/ru-en/better-teacher/student",
    },
}

MOCK_REQUESTS = [
    {
        "method": "POST",
        "url": "https://firefox-ci-tc.services.mozilla.com/api/index/v1/tasks/indexes",
        "responses": [{"json": {"tasks": []}}],
    },
    {
        "method": "POST",
        "url": "https://firefox-ci-tc.services.mozilla.com/api/queue/v1/tasks/status",
        "responses": [{"json": {"statuses": []}}],
    },
]


def test_artifact_mounts(full_task_graph: TaskGraph):
    task = [
        t
        for t in full_task_graph.tasks.values()
        if t.label == "backtranslations-train-backwards-model-ru-en"
    ][0]
    # No need to bother looking for _all_ files (we'd just duplicate
    # the full list if we did that...), but we verify that one file
    # is well formed.
    mounted_files = {m["file"]: m for m in task.task["payload"]["mounts"] if "file" in m}
    assert mounted_files["./artifacts/model.npz"]["content"] == {
        "url": "https://storage.googleapis.com/releng-translations-dev/models/ru-en/better-teacher/student/model.npz",
    }


def test_no_eval_tasks(optimized_task_graph: TaskGraph):
    """Ensure evaluate tasks for backtranslations-train-backwards-model aren't targeted.
    See https://github.com/mozilla/translations/issues/628"""
    eval_tasks = [
        task.label
        for task in optimized_task_graph.tasks.values()
        if task.label.startswith("evaluate-backward")
    ]
    assert len(eval_tasks) == 0
