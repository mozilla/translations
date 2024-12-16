from copy import deepcopy

from taskgraph.taskgraph import TaskGraph

from translations_taskgraph.parameters import get_ci_training_config

PARAMS = deepcopy(get_ci_training_config())
PARAMS["target_tasks_method"] = "train-target-tasks"

MOCK_REQUESTS = [
    {
        "substitute_digest": {
            "build-docker-image-base": "digest_base",
            "build-docker-image-inference": "digest_inference",
            "build-docker-image-test": "digest_test",
            "build-docker-image-toolchain-build": "digest_toolchain",
            "build-docker-image-train": "digest_train",
        },
        "method": "POST",
        "url": "https://firefox-ci-tc.services.mozilla.com/api/index/v1/tasks/indexes",
        "responses": [
            {
                "json": {
                    "tasks": [
                        {
                            "namespace": "translations.cache.level-3.docker-images.v2.base.hash.{digest_base}",
                            "taskId": "build-docker-image-base",
                        },
                        {
                            "namespace": "translations.cache.level-3.docker-images.v2.inference.hash.{digest_inference}",
                            "taskId": "build-docker-image-inference",
                        },
                        {
                            "namespace": "translations.cache.level-3.docker-images.v2.test.hash.{digest_test}",
                            "taskId": "build-docker-image-test",
                        },
                        {
                            "namespace": "translations.cache.level-3.docker-images.v2.toolchain-build.hash.{digest_toolchain}",
                            "taskId": "build-docker-image-toolchain-build",
                        },
                        {
                            "namespace": "translations.cache.level-3.docker-images.v2.train.hash.{digest_train}",
                            "taskId": "build-docker-image-train",
                        },
                    ],
                },
                "status_code": 200,
            },
        ],
    },
    {
        "method": "POST",
        "url": "https://firefox-ci-tc.services.mozilla.com/api/queue/v1/tasks/status",
        "responses": [
            {
                "json": {
                    "statuses": [
                        {
                            "status": {
                                "state": "completed",
                                "expires": "3024-08-21T22:37:28.781Z",
                            },
                            "taskId": "build-docker-image-base",
                        },
                        {
                            "status": {
                                "state": "completed",
                                "expires": "3024-08-21T22:37:28.781Z",
                            },
                            "taskId": "build-docker-image-inference",
                        },
                        {
                            "status": {
                                "state": "completed",
                                "expires": "3024-08-21T22:37:28.781Z",
                            },
                            "taskId": "build-docker-image-test",
                        },
                        {
                            "status": {
                                "state": "completed",
                                "expires": "3024-08-21T22:37:28.781Z",
                            },
                            "taskId": "build-docker-image-toolchain-build",
                        },
                        {
                            "status": {
                                "state": "completed",
                                "expires": "3024-08-21T22:37:28.781Z",
                            },
                            "taskId": "build-docker-image-train",
                        },
                    ],
                },
                "status_code": 200,
            },
        ],
    },
]


def test_last_task_is_targeted(target_task_set: TaskGraph):
    """Ensure that the last task in the pipeline is targeted by default"""
    assert any([task == "all-ru-en-1" for task in target_task_set.tasks])


def test_cached_tasks_optimized_away(optimized_task_graph: TaskGraph):
    """Ensure that any tasks found in a cache route are _not_ present
    in the optimized graph (ie: they will not be scheduled)."""
    for task in optimized_task_graph.tasks.values():
        assert not task.label.startswith("build-docker-image")
