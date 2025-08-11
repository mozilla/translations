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


def test_no_backwards_task(full_task_graph: TaskGraph):
    """
    Corpus continuation removes backwards task in "use" mode
    """
    task = [
        t
        for t in full_task_graph.tasks.values()
        if t.label == "backtranslations-train-backwards-model-ru-en"
    ]
    assert len(task) == 0


def test_no_eval_tasks(optimized_task_graph: TaskGraph):
    """Ensure evaluate tasks for backtranslations-train-backwards-model aren't targeted.
    See https://github.com/mozilla/translations/issues/628"""
    eval_tasks = [
        task.label
        for task in optimized_task_graph.tasks.values()
        if task.label.startswith("evaluate-backward")
    ]
    assert len(eval_tasks) == 0
