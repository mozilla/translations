"""
Typed responses for working with the Taskcluster API.
"""

from pydantic import BaseModel
from typing import Any, Optional
import taskcluster


class Run(BaseModel):
    runId: int  # 0,
    state: str  # "completed"
    reasonCreated: str  # "scheduled"
    reasonResolved: str  # "completed"
    scheduled: str  # "2024-11-12T13:50:55.910Z"

    # These are not present during an exception:
    workerGroup: Optional[str] = None  # "us-central1-a"
    workerId: Optional[str] = None  # "4909417939093873369"
    takenUntil: Optional[str] = None  # "2024-11-12T14:10:56.073Z"
    started: Optional[str] = None  # "2024-11-12T13:50:56.076Z"
    resolved: Optional[str] = None  # "2024-11-12T13:52:36.465Z"


class Metadata(BaseModel):
    description: str  # "Dummy task that ensures all parts of training pipeline will run"
    name: str  # "all-pipeline-en-lt-2"
    owner: str  # "gregtatum@users.noreply.github.com"
    source: str  # "https://github.com/mozilla/translations/blob/4b99af14117a3e662ee0a27bda93aa1170e964e7/taskcluster/kinds/all-pipeline"


class TaskExtra(BaseModel):
    index: Any
    parent: str  # "e1DMdEzNSGyGhdjaWFYpxQ"


class Task(BaseModel):
    created: str  # "2024-11-04T15:39:54.898Z"
    deadline: str  # "2024-11-24T15:39:54.898Z"
    dependencies: list[str]  # [ "CM1cp-ZWSnWkiQ96mKjmvA", "CnkDIu9LRCqI89thrhWAlQ", … ]
    expires: str  # "2025-02-02T15:39:54.898Z"
    extra: TaskExtra
    metadata: Metadata
    payload: Any
    priority: str  # "low"
    projectId: str  # "none"
    provisionerId: str  # "built-in"
    requires: str  # "all-completed"
    retries: int
    routes: list[str]  # [ "checks" ]
    schedulerId: str  # "translations-level-1"
    scopes: list[Any]
    tags: dict[str, Any]
    taskGroupId: str  # "TaeCdUs5Rqq7w1Tbf1PShQ"
    taskQueueId: str  # "built-in/succeed"
    workerType: str  # "succeed"

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.task(*args, **kwargs)
        return Task(**response)


class Status(BaseModel):
    deadline: str  # "2024-11-24T15:39:54.898Z"
    expires: str  # "2025-02-02T15:39:54.898Z"
    projectId: str  # "none"
    provisionerId: str  # "built-in"
    retriesLeft: int
    runs: list[Run]
    schedulerId: str  # "translations-level-1"
    state: str  # "completed"
    taskGroupId: str  # "TaeCdUs5Rqq7w1Tbf1PShQ"
    taskId: str  # "CmximseBTi-d8tcWOl-KZA"
    taskQueueId: str  # "built-in/succeed"
    workerType: str  # "succeed"

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.status(*args, **kwargs)
        return Status(**response["status"])


class TaskAndStatus(BaseModel):
    task: Task
    status: Status

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        # This requires 2 API calls, even though other APIs return both.
        return TaskAndStatus(
            task=Task.call(queue, *args, **kwargs), status=Status.call(queue, *args, **kwargs)
        )


class GetTaskGroup(BaseModel):
    taskGroupId: str  # 'I9uKJEPvQd-1zeItJK0cOQ'
    schedulerId: str  # 'translations-level-1'
    expires: str  # '2025-11-07T21:56:15.759Z'

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.getTaskGroup(*args, **kwargs)
        return GetTaskGroup(**response)


class ListTaskGroup(BaseModel):
    expires: str  # "2025-11-04T16:39:54.296Z"
    schedulerId: str  # "translations-level-1"
    taskGroupId: str  # "TaeCdUs5Rqq7w1Tbf1PShQ"
    tasks: list[TaskAndStatus]

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.listTaskGroup(*args, **kwargs)
        return ListTaskGroup(**response)


class Artifact(BaseModel):
    storageType: str  # "s3"
    name: str  # "public/build/lex.50.50.enlt.s2t.bin.gz"
    expires: str  # "2025-11-04T15:39:54.626Z"
    contentType: str  # "application/gzip"


class ListArtifacts(BaseModel):
    artifacts: list[Artifact]

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.listArtifacts(*args, **kwargs)
        return ListArtifacts(**response)


class ListDependentTasks(BaseModel):
    taskId: str
    tasks: list[TaskAndStatus]

    @staticmethod
    def call(queue: taskcluster.Queue, *args, **kwargs):
        response: Any = queue.listDependentTasks(*args, **kwargs)
        return ListDependentTasks(**response)
