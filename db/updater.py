"""
Translations DB updater - Processes training runs from TaskCluster and GCS and stores data in SQLite.

PYTHONPATH=$(pwd) python db/updater.py --db-path data/db/db.sqlite
"""

import argparse
import json
import logging
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests
import taskcluster
from taskgraph.util.taskcluster import get_artifact, get_artifact_url

from db.models import Evaluation, Corpus, WordAlignedCorpus, Model, TrainingRun, Task
from db.sql import DatabaseManager

warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "translations-data-prod"
BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data"
ROOT_DIR = Path(__file__).parent.parent
MODEL_REGISTRY_DIR = ROOT_DIR / "data/db"
SQLITE_PATH = MODEL_REGISTRY_DIR / "db.sqlite"
SQLITE_GCS_OBJECT = "db/db.sqlite"

os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"


class TaskClusterClient:
    """
    Client for interacting with Mozilla's TaskCluster API.

    Provides methods to retrieve task information from TaskCluster task groups.
    Handles pagination automatically to fetch all tasks in a group, and gracefully
    handles expired task groups by catching 404 errors.
    """

    def __init__(self):
        self.queue = taskcluster.Queue(
            options={"rootUrl": "https://firefox-ci-tc.services.mozilla.com"}
        )

    def get_tasks_for_group(self, task_group_id: str) -> list[dict]:
        tasks = []
        try:
            list_task_group = self.queue.listTaskGroup(task_group_id)
            tasks.extend(list_task_group["tasks"])

            for _ in range(10):
                if not list_task_group.get("continuationToken"):
                    break
                list_task_group = self.queue.listTaskGroup(
                    task_group_id, continuationToken=list_task_group["continuationToken"]
                )
                tasks.extend(list_task_group["tasks"])
        except taskcluster.exceptions.TaskclusterRestFailure as error:
            if error.status_code == 404:
                logger.debug(f"Task group expired: {task_group_id}")
            else:
                raise error
        return tasks


class BlobInfo:
    """
    Lightweight representation of a Google Cloud Storage blob.

    Provides basic blob information (name, size, creation time) and methods to interact
    with blobs using unauthenticated HTTP requests. This allows read-only access to public
    GCS buckets without requiring authentication credentials.
    """

    def __init__(
        self, name: str, size: Optional[int] = None, time_created: Optional[datetime] = None
    ):
        self.name = name
        self.size = size
        self.time_created = time_created

    def download_as_text(self) -> str:
        url = f"https://storage.googleapis.com/{BUCKET_NAME}/{self.name}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def exists(self) -> bool:
        url = f"https://storage.googleapis.com/{BUCKET_NAME}/{self.name}"
        response = requests.head(url)
        return response.ok


class GCSClient:
    """
    Client for accessing Google Cloud Storage buckets with minimal authentication.

    Uses unauthenticated HTTP requests for all read operations (listing blobs, downloading files)
    by querying the public GCS JSON API. Only initializes the authenticated Google Cloud Storage
    SDK when upload operations are needed. Caches the entire bucket structure in memory on first
    access to minimize API calls.
    """

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self._all_blobs = None
        self._client = None
        self._bucket = None

    def _get_authenticated_bucket(self):
        if self._bucket is None:
            from google.cloud import storage

            self._client = storage.Client(project=PROJECT_NAME)
            self._bucket = self._client.get_bucket(self.bucket_name)
        return self._bucket

    def _ensure_blobs_loaded(self):
        if self._all_blobs is not None:
            return

        logger.info("Loading GCS bucket structure...")
        self._all_blobs = []

        try:
            list_url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket_name}/o"
            params = {"maxResults": 1000}

            while True:
                response = requests.get(list_url, params=params)
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    time_created = None
                    if item.get("timeCreated"):
                        time_created = datetime.fromisoformat(
                            item["timeCreated"].replace("Z", "+00:00")
                        )

                    blob = BlobInfo(
                        name=item["name"], size=int(item.get("size", 0)), time_created=time_created
                    )
                    self._all_blobs.append(blob)

                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break
                params["pageToken"] = next_page_token

        except Exception as e:
            logger.error(f"Error loading blobs: {e}")
            self._all_blobs = []

        logger.info(f"Loaded {len(self._all_blobs)} files from GCS")

    def get_subdirectories(self, prefix: str) -> set[str]:
        self._ensure_blobs_loaded()

        subdirs = set()
        prefix_len = len(prefix)
        for blob in self._all_blobs:
            if blob.name.startswith(prefix):
                remainder = blob.name[prefix_len:]
                if "/" in remainder:
                    subdir_name = remainder.split("/")[0]
                    subdirs.add(f"{prefix}{subdir_name}/")
        return subdirs

    def list_blobs(self, prefix: str):
        self._ensure_blobs_loaded()
        return [blob for blob in self._all_blobs if blob.name.startswith(prefix)]

    def get_blob(self, blob_url: str):
        self._ensure_blobs_loaded()
        for blob in self._all_blobs:
            if blob.name == blob_url:
                return blob
        return None

    def download_sqlite(self, path: Path) -> bool:
        url = f"https://storage.googleapis.com/{self.bucket_name}/{SQLITE_GCS_OBJECT}"
        response = requests.head(url)
        if not response.ok:
            logger.info("No existing database found in GCS")
            return False

        response = requests.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
        logger.info("Downloaded database from GCS")
        return True

    def upload_sqlite(self, path: Path):
        bucket = self._get_authenticated_bucket()
        blob = bucket.blob(SQLITE_GCS_OBJECT)
        blob.cache_control = "public, max-age=60"
        blob.content_type = "application/vnd.sqlite3"
        blob.upload_from_filename(path)
        logger.info("Uploaded database to GCS")

    def upload_json(self, gcs_path: str, content: str):
        bucket = self._get_authenticated_bucket()
        blob = bucket.blob(gcs_path)
        blob.cache_control = "public, max-age=3600"
        blob.content_type = "application/json"
        blob.upload_from_string(content)
        logger.info(f"Uploaded {gcs_path} to GCS")


class GCSDataCollector:
    """
    Collects training data from Google Cloud Storage bucket structure.

    Discovers training runs by analyzing the GCS directory structure under models/ and corpus/.
    Extracts model artifacts, evaluation results, and corpus information by parsing file paths
    and downloading metadata files. Handles multiple task groups per training run and various
    model types (backwards, teacher, student, etc.).
    """

    def __init__(self, gcs_client: GCSClient):
        self.gcs = gcs_client

    def get_training_runs_by_langpair(self) -> dict[str, list[TrainingRun]]:
        langpairs = [
            model_prefix.split("/")[1] for model_prefix in self.gcs.get_subdirectories("models/")
        ]

        runs_by_langpair = {}
        training_runs_by_name = {}

        for langpair in langpairs:
            training_runs = []
            runs_by_langpair[langpair] = training_runs

            for task_group_prefix in self.gcs.get_subdirectories(f"models/{langpair}/"):
                name_task_group_tuple = task_group_prefix.split("/")[2]
                if name_task_group_tuple.endswith("None"):
                    continue

                name = name_task_group_tuple[:-23]
                task_group_id = name_task_group_tuple[-22:]
                key = f"{langpair} {name}"

                if key in training_runs_by_name:
                    training_runs_by_name[key].task_group_ids.append(task_group_id)
                    training_runs_by_name[key].task_group_ids.sort()
                else:
                    training_run = TrainingRun.create(name, [task_group_id], langpair)
                    training_runs.append(training_run)
                    training_runs_by_name[key] = training_run

        return runs_by_langpair

    def collect_models(self, training_run: TrainingRun, task_group_id: str):
        models = self._get_models(training_run, task_group_id)

        if "backward" in models:
            training_run.backwards = models["backward"]
        if "teacher_1" in models:
            training_run.teacher_1 = models["teacher_1"]
        if "teacher_2" in models:
            training_run.teacher_2 = models["teacher_2"]
        if "student" in models:
            training_run.student = models["student"]
        if "student_finetuned" in models:
            training_run.student_finetuned = models["student_finetuned"]
        if "student_quantized" in models:
            training_run.student_quantized = models["student_quantized"]
        if "student_exported" in models:
            training_run.student_exported = models["student_exported"]
            if training_run.student_quantized:
                training_run.student_exported.flores = training_run.student_quantized.flores

    def _get_models(self, training_run: TrainingRun, task_group_id: str) -> dict[str, Model]:
        prefix = f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"

        model_dirs = self.gcs.get_subdirectories(prefix)
        models = {}

        model_type_map = {
            "backward": "backward",
            "backwards": "backward",
            "teacher0": "teacher_1",
            "teacher1": "teacher_2",
            "student": "student",
            "student-finetuned": "student_finetuned",
            "quantized": "student_quantized",
            "exported": "student_exported",
        }

        for model_dir in model_dirs:
            if model_dir.endswith("evaluation/"):
                continue

            gcs_model_name = model_dir.rstrip("/").split("/")[-1]
            if gcs_model_name not in model_type_map:
                continue

            kind = model_type_map[gcs_model_name]

            model = Model(task_group_id=task_group_id)
            model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{model_dir}"

            blobs = self.gcs.list_blobs(prefix=model_dir)
            model.artifact_urls = [
                f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
            ]

            if blobs:
                earliest_blob = min(
                    blobs, key=lambda b: b.time_created if b.time_created else datetime.max
                )
                if earliest_blob.time_created:
                    model.date = earliest_blob.time_created

            eval_blob = self._get_flores_eval(training_run, task_group_id, gcs_model_name)
            if eval_blob:
                model.flores = self._parse_flores_results(eval_blob)

            models[kind] = model

        return models

    def get_export_metadata(self, artifact_folder: str) -> Optional[dict]:
        import json

        if not artifact_folder:
            return None

        metadata_path = (
            artifact_folder.replace(f"https://storage.googleapis.com/{BUCKET_NAME}/", "")
            + "metadata.json"
        )

        blob = self.gcs.get_blob(metadata_path)
        if not blob or not blob.exists():
            return None

        try:
            return json.loads(blob.download_as_text())
        except Exception as e:
            logger.debug(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def _get_flores_eval(self, training_run: TrainingRun, task_group_id: str, gcs_model_name: str):
        eval_base = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/evaluation/"
        )

        eval_dirs = self.gcs.get_subdirectories(eval_base)
        if not eval_dirs:
            return None

        # Map GCS model names to their evaluation directory names (including legacy names)
        eval_dir_names = [gcs_model_name]
        if gcs_model_name == "quantized":
            eval_dir_names.append("speed")

        # Check for standard flores-devtest in model-specific subdirectories (legacy structure)
        for eval_dir in eval_dirs:
            eval_dir_name = eval_dir.rstrip("/").split("/")[-1]

            if not any(eval_dir_name == name for name in eval_dir_names):
                continue

            blobs = self.gcs.list_blobs(prefix=eval_dir)
            for blob in blobs:
                if not blob.name.endswith(".metrics.json"):
                    continue

                if "devtest" in blob.name and "aug-" not in blob.name:
                    return blob

        # Check for flores-devtest in dedicated evaluation subdirectories (newer structure)
        for eval_dir in eval_dirs:
            eval_dir_name = eval_dir.rstrip("/").split("/")[-1]

            if (
                f"{gcs_model_name}-flores-devtest" in eval_dir_name
                and "-aug-" not in eval_dir_name
            ):
                blobs = self.gcs.list_blobs(prefix=eval_dir)
                for blob in blobs:
                    if blob.name.endswith(".metrics.json"):
                        return blob

        return None

    @staticmethod
    def _parse_flores_results(flores_blob) -> Evaluation:
        import json

        results = json.loads(flores_blob.download_as_text())
        comet = results.get("comet", {}).get("score")
        if comet:
            comet *= 100.0
        return Evaluation(
            chrf=results["chrf"]["score"], bleu=results["bleu"]["score"], comet=comet
        )

    def collect_corpora(self, training_run: TrainingRun):
        for task_group_id in training_run.task_group_ids:
            corpus_prefix = f"corpus/{training_run.langpair}/{training_run.name}_{task_group_id}/"

            corpus_types = self.gcs.get_subdirectories(corpus_prefix)
            for corpus_dir in corpus_types:
                corpus_type = corpus_dir.rstrip("/").split("/")[-1]

                if corpus_type == "parallel":
                    corpus = self._get_corpus(training_run, corpus_dir)
                    if corpus:
                        if not training_run.parallel_corpus:
                            training_run.parallel_corpus = corpus
                elif corpus_type == "distillation":
                    corpus = self._get_corpus(training_run, corpus_dir)
                    if corpus:
                        if not training_run.distillation_corpus:
                            training_run.distillation_corpus = corpus
                elif corpus_type == "backtranslations":
                    corpus = self._get_corpus(training_run, corpus_dir)
                    if corpus:
                        if not training_run.backtranslations_corpus:
                            training_run.backtranslations_corpus = corpus

    def _get_corpus(self, training_run: TrainingRun, corpus_prefix: str) -> Optional[Corpus]:
        source_blob_name = f"{corpus_prefix}corpus.{training_run.source_lang}.zst"
        target_blob_name = f"{corpus_prefix}corpus.{training_run.target_lang}.zst"

        source_blob = self.gcs.get_blob(source_blob_name)
        target_blob = self.gcs.get_blob(target_blob_name)

        if not (source_blob and source_blob.exists() and target_blob and target_blob.exists()):
            return None

        source_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{source_blob_name}"
        target_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{target_blob_name}"

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=source_blob.size or 0,
            target_bytes=target_blob.size or 0,
        )

    def get_config(
        self, langpair: str, training_run_name: str, task_group_id: str
    ) -> Optional[dict]:
        import yaml

        config_path = f"experiments/{langpair}/{training_run_name}_{task_group_id}/config.yml"
        try:
            blob = self.gcs.get_blob(config_path)
            if blob and blob.exists():
                config_text = blob.download_as_text()
                return yaml.safe_load(config_text)
        except Exception as e:
            logger.debug(f"Failed to load config: {e}")
        return None

    def get_date(self, training_run: TrainingRun) -> Optional[datetime]:
        for task_group_id in training_run.task_group_ids:
            prefix = f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
            blobs = self.gcs.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.time_created:
                    return blob.time_created
        return None


class TaskClusterDataCollector:
    """
    Collects training data from TaskCluster task metadata.

    Supplements GCS data with information from TaskCluster tasks, including task IDs,
    completion dates, and artifact URLs. Matches tasks to models by analyzing task names
    with regex patterns. Also retrieves corpus data from TaskCluster artifacts when not
    available in GCS.
    """

    def __init__(self, tc_client: TaskClusterClient):
        self.tc = tc_client

    def get_tasks_for_group(self, task_group_id: str) -> tuple[list[Task], list[dict]]:
        tc_tasks = self.tc.get_tasks_for_group(task_group_id)
        if not tc_tasks:
            return [], []

        task_objects = []
        for task_dict in tc_tasks:
            task_id = task_dict["status"]["taskId"]
            task_group_id_inner = task_dict["status"]["taskGroupId"]
            created_date = task_dict["task"]["created"]
            task_name = task_dict["task"]["metadata"].get("name", "")

            state = None
            resolved_date = None
            runs = task_dict["status"].get("runs", [])
            if runs:
                last_run = runs[-1]
                state = last_run.get("state")
                resolved_date = last_run.get("resolved")

            task_obj = Task(
                task_id=task_id,
                task_group_id=task_group_id_inner,
                created_date=created_date,
                state=state,
                task_name=task_name,
                resolved_date=resolved_date,
            )
            task_objects.append(task_obj)

        return task_objects, tc_tasks

    @staticmethod
    def update_date_started(training_run: TrainingRun, tasks: list[Task]):
        for task in tasks:
            date = str_to_datetime(task.created_date)
            if training_run.date_started is None or date < training_run.date_started:
                training_run.date_started = date

    def supplement_models_with_task_metadata(self, training_run: TrainingRun, tasks: list[Task]):
        if not tasks:
            return

        model_task_map = {
            "backwards": (
                r"^train-backwards-|^backtranslations-train-backwards-model-",
                training_run.backwards,
            ),
            "teacher_1": (
                r"^train-teacher-.*-1|^train-teacher-model-.*-1",
                training_run.teacher_1,
            ),
            "teacher_2": (r"^train-teacher-model-.*-2", training_run.teacher_2),
            "student": (
                r"^train-student-|^distillation-student-model-train-",
                training_run.student,
            ),
            "student_finetuned": (
                r"^finetune-student|^distillation-student-model-finetune-",
                training_run.student_finetuned,
            ),
            "student_quantized": (r"^quantize-", training_run.student_quantized),
            "student_exported": (r"^export-", training_run.student_exported),
        }

        for model_name, (pattern, model) in model_task_map.items():
            if model and not model.task_id:
                task = find_latest_task(tasks, match_by_label(pattern))
                if task:
                    model.task_id = task.task_id
                    completed_date = get_completed_time_from_task(task)
                    if completed_date:
                        if not model.date:
                            model.date = completed_date
                        else:
                            model_date_naive = (
                                model.date.replace(tzinfo=None)
                                if model.date.tzinfo
                                else model.date
                            )
                            if completed_date < model_date_naive:
                                model.date = completed_date
                    logger.debug(f"Supplemented {model_name} with task metadata")

    def collect_corpora(self, training_run: TrainingRun, tasks: list[Task]):
        if not training_run.parallel_corpus_aligned:
            training_run.parallel_corpus_aligned = self._get_word_aligned_corpus(
                training_run, find_latest_task(tasks, match_by_label(r"^corpus-align-parallel-"))
            )

        if not training_run.backtranslations_corpus_aligned:
            training_run.backtranslations_corpus_aligned = self._get_word_aligned_corpus(
                training_run,
                find_latest_task(
                    tasks,
                    match_by_label(r"^alignments-backtranslated-|^corpus-align-backtranslations-"),
                ),
            )

        if not training_run.distillation_corpus_aligned:
            training_run.distillation_corpus_aligned = self._get_word_aligned_corpus(
                training_run,
                find_latest_task(
                    tasks, match_by_label(r"^alignments-student-|^corpus-align-distillation-")
                ),
            )

        if not training_run.parallel_corpus:
            training_run.parallel_corpus = self._get_corpus(
                training_run,
                find_latest_task(tasks, match_by_label(r"^merge-corpus-|^corpus-merge-parallel-")),
            )

        if not training_run.backtranslations_corpus:
            training_run.backtranslations_corpus = self._get_mono_corpus(training_run, tasks)

        if not training_run.distillation_corpus:
            training_run.distillation_corpus = self._get_corpus(
                training_run,
                find_latest_task(tasks, match_by_label(r"^distillation-corpus-final-filtering-")),
            )

    @staticmethod
    def _get_corpus(training_run: TrainingRun, task: Optional[Task]) -> Optional[Corpus]:
        if not task:
            return None

        task_id = task.task_id
        source_url = get_artifact_url(
            task_id, f"public/build/corpus.{training_run.source_lang}.zst"
        )
        target_url = get_artifact_url(
            task_id, f"public/build/corpus.{training_run.target_lang}.zst"
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )

    @staticmethod
    def _get_word_aligned_corpus(
        training_run: TrainingRun, task: Optional[Task]
    ) -> Optional[WordAlignedCorpus]:
        if not task:
            return None

        task_id = task.task_id
        alignments_url = get_artifact_url(task.task_id, "public/build/corpus.aln.zst")
        source_url = get_artifact_url(
            task_id, f"public/build/corpus.tok-icu.{training_run.source_lang}.zst"
        )
        target_url = get_artifact_url(
            task_id, f"public/build/corpus.tok-icu.{training_run.target_lang}.zst"
        )

        alignments_head = requests.head(alignments_url, allow_redirects=True)
        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not all([alignments_head.ok, source_head.ok, target_head.ok]):
            return None

        return WordAlignedCorpus(
            source_url=source_url,
            target_url=target_url,
            alignments_url=alignments_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
            alignments_bytes=int(alignments_head.headers.get("content-length", 0)),
        )

    @staticmethod
    def _get_mono_corpus(training_run: TrainingRun, tasks: list[Task]) -> Optional[Corpus]:
        source_task = find_latest_task(
            tasks,
            match_by_label(r"^collect-mono-trg-|^backtranslations-mono-trg-dechunk-translations-"),
        )
        target_task = find_latest_task(
            tasks,
            match_by_label(r"^collect-mono-src-|^distillation-mono-src-dechunk-translations-"),
        )

        if not source_task or not target_task:
            return None

        source_url = get_artifact_url(
            source_task.task_id, f"public/build/mono.{training_run.source_lang}.zst"
        )
        target_url = get_artifact_url(
            target_task.task_id, f"public/build/mono.{training_run.target_lang}.zst"
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )


class ReleaseStatusCollector:
    REMOTE_SETTINGS_URL = (
        "https://firefox.settings.services.mozilla.com/v1/buckets/main/"
        "collections/translations-models-v2/records"
    )

    FILTER_EXPRESSION_TO_STATUS = {
        "": "Release",
        "env.appinfo.OS == 'Android'": "Release Android",
        "env.appinfo.OS != 'Android'": "Release Desktop",
        "env.channel == 'default' || env.channel == 'nightly'": "Nightly",
    }

    @staticmethod
    def fetch_deployed_models() -> dict[str, str]:
        response = requests.get(ReleaseStatusCollector.REMOTE_SETTINGS_URL)
        response.raise_for_status()
        data = response.json()

        hash_to_status = {}
        for record in data.get("data", []):
            decompressed_hash = record.get("decompressedHash")
            filter_expression = record.get("filter_expression", "")
            if not decompressed_hash:
                continue

            status = ReleaseStatusCollector.FILTER_EXPRESSION_TO_STATUS.get(filter_expression)
            if status:
                hash_to_status[decompressed_hash] = status

        return hash_to_status


class FinalEvalsCollector:
    """
    Collects final evaluation data from GCS bucket or local directory.

    Discovers evaluation files by analyzing file names following the pattern:
    <src>-<trg>__<dataset>__<translator>__<model>__latest__<type>.json
    Groups files by evaluation entry and loads metric scores, storing them in
    the database with LLM sub-scores for llm-ref metrics.
    """

    LOCAL_PREFIX = Path("data/final_evals")

    def __init__(self, gcs_client: GCSClient):
        self.gcs = gcs_client
        self.local_dir = None
        # For testing use
        # self.local_dir = self.LOCAL_PREFIX

    def collect(self, db: "DatabaseManager"):
        if self.local_dir and self.local_dir.exists():
            files = self._list_local_files()
        elif self.gcs:
            files = self._list_gcs_files()
        else:
            logger.info("No final_evals source configured, skipping")
            return

        evals_by_key = self._group_files(files)
        logger.info(f"Found {len(evals_by_key)} final evaluation entries")

        for key, file_group in evals_by_key.items():
            self._process_evaluation(db, key, file_group)

    def _list_local_files(self) -> list[tuple[str, Path]]:
        files = []
        for path in self.local_dir.glob("*__latest__*"):
            files.append((path.name, path))
        return files

    def _list_gcs_files(self) -> list[tuple[str, BlobInfo]]:
        files = []
        for blob in self.gcs.list_blobs("final-evals/"):
            if "__latest__" in blob.name:
                filename = blob.name.split("/")[-1]
                files.append((filename, blob))
        return files

    def _group_files(self, files: list) -> dict:
        evals = {}
        for filename, source in files:
            parsed = self._parse_filename(filename)
            if not parsed:
                continue

            key = (parsed["langpair"], parsed["dataset"], parsed["translator"], parsed["model"])
            if key not in evals:
                evals[key] = {"translations": None, "metrics": {}}

            if parsed["file_type"] == "translations":
                evals[key]["translations"] = source
            elif parsed["file_type"].endswith(".metrics"):
                metric_name = parsed["file_type"].replace(".metrics", "")
                if metric_name not in evals[key]["metrics"]:
                    evals[key]["metrics"][metric_name] = {}
                evals[key]["metrics"][metric_name]["metrics"] = source
            elif parsed["file_type"].endswith(".scores"):
                metric_name = parsed["file_type"].replace(".scores", "")
                if metric_name not in evals[key]["metrics"]:
                    evals[key]["metrics"][metric_name] = {}
                evals[key]["metrics"][metric_name]["scores"] = source

        return evals

    def _parse_filename(self, filename: str) -> dict | None:
        parts = filename.replace(".json", "").split("__")
        if len(parts) < 5:
            return None
        if parts[4] != "latest":
            return None

        langpair = parts[0]
        if "-" not in langpair:
            return None
        src, trg = langpair.split("-", 1)

        return {
            "langpair": langpair,
            "src": src,
            "trg": trg,
            "dataset": parts[1],
            "translator": parts[2],
            "model": parts[3],
            "file_type": parts[5] if len(parts) > 5 else "translations",
        }

    def _process_evaluation(self, db: "DatabaseManager", key: tuple, file_group: dict):
        import json

        langpair, dataset, translator, model_name = key
        src, trg = langpair.split("-", 1)

        translations_url = None
        if file_group["translations"]:
            source = file_group["translations"]
            if isinstance(source, Path):
                translations_url = f"/{self.LOCAL_PREFIX}/{source.name}"
            else:
                translations_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{source.name}"

        model_id = None
        if translator == "bergamot":
            model_id = db.find_model_id_by_name(src, trg, model_name)

        db.conn.execute(
            """
            INSERT INTO final_evals(source_lang, target_lang, dataset, translator, model_name, model_id, translations_url)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(source_lang, target_lang, dataset, translator, model_name) DO UPDATE SET
                model_id=excluded.model_id,
                translations_url=excluded.translations_url
            """,
            (src, trg, dataset, translator, model_name, model_id, translations_url),
        )

        row = db.conn.execute(
            "SELECT id FROM final_evals WHERE source_lang=? AND target_lang=? AND dataset=? AND translator=? AND model_name=?",
            (src, trg, dataset, translator, model_name),
        ).fetchone()
        eval_id = row[0]

        for metric_name, metric_files in file_group["metrics"].items():
            if "metrics" not in metric_files:
                continue

            source = metric_files["metrics"]
            try:
                if isinstance(source, Path):
                    metrics_data = json.loads(source.read_text())
                else:
                    metrics_data = json.loads(source.download_as_text())
            except Exception as e:
                logger.debug(f"Failed to load metrics {source}: {e}")
                continue

            corpus_score = metrics_data.get("score", 0)
            details_json = json.dumps(metrics_data.get("details", {}))

            scores_url = None
            if "scores" in metric_files:
                scores_source = metric_files["scores"]
                if isinstance(scores_source, Path):
                    scores_url = f"/{self.LOCAL_PREFIX}/{scores_source.name}"
                else:
                    scores_url = (
                        f"https://storage.googleapis.com/{BUCKET_NAME}/{scores_source.name}"
                    )

            db.conn.execute(
                """
                INSERT INTO final_eval_metrics(eval_id, metric_name, corpus_score, details_json, scores_url)
                VALUES(?,?,?,?,?)
                ON CONFLICT(eval_id, metric_name) DO UPDATE SET
                    corpus_score=excluded.corpus_score,
                    details_json=excluded.details_json,
                    scores_url=excluded.scores_url
                """,
                (eval_id, metric_name, corpus_score, details_json, scores_url),
            )

            if metric_name.startswith("llm"):
                metric_row = db.conn.execute(
                    "SELECT id FROM final_eval_metrics WHERE eval_id=? AND metric_name=?",
                    (eval_id, metric_name),
                ).fetchone()
                metric_id = metric_row[0]

                details = metrics_data.get("details", {})
                scores = details.get("scores", {})
                summaries = details.get("summary", {})

                db.conn.execute(
                    "DELETE FROM final_eval_llm_scores WHERE metric_id=?", (metric_id,)
                )

                for criterion, score in scores.items():
                    summary = summaries.get(criterion, "")
                    db.conn.execute(
                        """
                        INSERT INTO final_eval_llm_scores(metric_id, criterion, score, summary)
                        VALUES(?,?,?,?)
                        """,
                        (metric_id, criterion, score, summary),
                    )


class PublicModelsJsonGenerator:
    """Generates public JSON catalog of Firefox translation models for bulk download."""

    GCS_OUTPUT_PATH = "db/models.json"
    BASE_URL = f"https://storage.googleapis.com/{BUCKET_NAME}"

    def __init__(self, gcs: GCSClient):
        self.gcs = gcs

    def generate(self, db: DatabaseManager, upload: bool = False):
        logger.info("Generating public models JSON")
        models_by_langpair = self._query_models(db)
        if not models_by_langpair:
            logger.warning("No exported models found for public JSON")
            return

        json_content = self._build_json(models_by_langpair)

        local_path = MODEL_REGISTRY_DIR / "models.json"
        local_path.write_text(json_content)
        logger.info(f"Wrote {local_path}")

        if upload:
            self.gcs.upload_json(self.GCS_OUTPUT_PATH, json_content)

    def _query_models(self, db: DatabaseManager) -> dict:
        cursor = db.conn.execute(
            """
            SELECT
                tr.source_lang || '-' || tr.target_lang as langpair,
                tr.source_lang,
                tr.target_lang,
                e.architecture,
                e.byte_size,
                e.hash,
                e.model_statistics,
                e.release_status,
                m.id as model_id,
                m.artifact_folder
            FROM models m
            JOIN training_runs tr ON m.run_id = tr.id
            JOIN exports e ON m.id = e.model_id
            WHERE m.kind = 'student_exported'
            ORDER BY
                langpair,
                e.architecture,
                CASE WHEN e.release_status LIKE 'Release%' THEN 0 ELSE 1 END,
                m.date DESC
        """
        )

        models_by_langpair = {}
        seen_langpair_arch = set()

        for row in cursor.fetchall():
            (
                langpair,
                src,
                trg,
                arch,
                byte_size,
                hash_val,
                model_stats_json,
                release_status,
                model_id,
                artifact_folder,
            ) = row

            key = (langpair, arch)
            if key in seen_langpair_arch:
                continue
            seen_langpair_arch.add(key)

            artifact_urls = self._get_artifact_urls(db, model_id)
            files = self._extract_files(artifact_urls, hash_val, byte_size)
            if not files.get("model"):
                continue

            metrics = self._get_metrics(db, model_id)
            model_stats = json.loads(model_stats_json) if model_stats_json else None

            model_entry = {
                "architecture": arch,
                "releaseStatus": release_status,
                "sourceLanguage": src,
                "targetLanguage": trg,
                "files": files,
            }

            if model_stats:
                model_entry["modelStatistics"] = {
                    "parameters": model_stats.get("parameters"),
                    "encoderParameters": model_stats.get("encoder_parameters"),
                    "decoderParameters": model_stats.get("decoder_parameters"),
                }

            if metrics:
                model_entry["metrics"] = {"flores200-plus": metrics}

            if langpair not in models_by_langpair:
                models_by_langpair[langpair] = []
            models_by_langpair[langpair].append(model_entry)

        return models_by_langpair

    def _get_artifact_urls(self, db: DatabaseManager, model_id: int) -> list[str]:
        cursor = db.conn.execute("SELECT url FROM artifacts WHERE model_id = ?", (model_id,))
        return [row[0] for row in cursor.fetchall()]

    def _extract_files(self, urls: list[str], model_hash: str, model_size: int) -> dict:
        files = {}
        for url in urls:
            path = url.replace(f"{self.BASE_URL}/", "")

            if ".intgemm.alphas.bin.gz" in url:
                files["model"] = {
                    "path": path,
                    "uncompressedSize": model_size,
                    "uncompressedHash": model_hash,
                }
            elif "lex." in url and ".s2t.bin.gz" in url:
                files["lexicalShortlist"] = {"path": path}
            elif "srcvocab." in url and ".spm.gz" in url:
                files["srcVocab"] = {"path": path}
            elif "trgvocab." in url and ".spm.gz" in url:
                files["trgVocab"] = {"path": path}
            elif "vocab." in url and ".spm.gz" in url:
                files["vocab"] = {"path": path}

        return files

    def _get_metrics(self, db: DatabaseManager, model_id: int) -> dict:
        cursor = db.conn.execute(
            """
            SELECT fem.metric_name, fem.corpus_score
            FROM final_evals fe
            JOIN final_eval_metrics fem ON fe.id = fem.eval_id
            WHERE fe.model_id = ?
              AND fe.dataset = 'flores200-plus'
              AND fe.translator = 'bergamot'
        """,
            (model_id,),
        )

        metrics = {}
        for metric_name, corpus_score in cursor.fetchall():
            if corpus_score is not None:
                metrics[metric_name] = corpus_score

        return metrics

    def _build_json(self, models_by_langpair: dict) -> str:
        output = {
            "generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "baseUrl": self.BASE_URL,
            "models": models_by_langpair,
        }
        return json.dumps(output, indent=2)


class Updater:
    """
    Main orchestrator for building the training runs database.

    Coordinates data collection from both GCS and TaskCluster, merges the information,
    and writes it to a SQLite database. Supports incremental updates by preserving
    existing data and only fetching new information. Can download an existing database
    from GCS, update it with new runs, and upload it back.
    """

    def __init__(self):
        gcs_client = GCSClient(BUCKET_NAME)
        tc_client = TaskClusterClient()
        self.gcs = gcs_client
        self.gcs_collector = GCSDataCollector(gcs_client)
        self.tc_collector = TaskClusterDataCollector(tc_client)
        self.final_evals_collector = FinalEvalsCollector(gcs_client=self.gcs)
        self.public_models_generator = PublicModelsJsonGenerator(gcs_client)
        self.db = None

    def build_database(self, upload: bool, db_path: Path, overwrite: bool = False):
        self._init_database(overwrite, db_path)

        runs_by_langpair = self.gcs_collector.get_training_runs_by_langpair()

        for training_runs in runs_by_langpair.values():
            for training_run in training_runs:
                self._process_training_run(training_run)

        self._update_release_statuses()

        self.final_evals_collector.collect(self.db)

        self.public_models_generator.generate(self.db, upload=upload)

        self._finalize_database(upload)

    def _init_database(self, overwrite: bool, db_path: Path):
        db_path.parent.mkdir(exist_ok=True, parents=True)
        if not overwrite and not db_path.exists():
            self.gcs.download_sqlite(db_path)

        self.db = DatabaseManager(db_path, rebuild=overwrite)

        if not overwrite:
            existing_count = len(self.db.get_existing_training_run_keys())
            logger.info(f"Found {existing_count} existing training runs in database")

    def _process_training_run(self, training_run: TrainingRun):
        logger.info(f"Processing {training_run.name} {training_run.langpair}")

        self._collect_data_from_gcs(training_run)

        run_id = self.db.upsert_training_run(training_run)

        self._preserve_existing_model_metadata(training_run, run_id)

        self._save_task_groups(training_run, run_id)

        tasks = self._get_or_fetch_tasks(training_run, run_id)

        if tasks:
            self._supplement_with_task_data(training_run, tasks, run_id)

        if not training_run.date_started:
            self._set_fallback_date(training_run)

        self._save_training_run_data(training_run, run_id)

    def _collect_data_from_gcs(self, training_run: TrainingRun):
        for task_group_id in training_run.task_group_ids:
            self.gcs_collector.collect_models(training_run, task_group_id)
        self.gcs_collector.collect_corpora(training_run)

    def _preserve_existing_model_metadata(self, training_run: TrainingRun, run_id: int):
        existing_models = self.db.get_models_for_run(run_id)
        model_metadata = {model.kind: (model.task_id, model.date) for model in existing_models}

        for model_attr in [
            "backwards",
            "teacher_1",
            "teacher_2",
            "student",
            "student_finetuned",
            "student_quantized",
            "student_exported",
        ]:
            model = getattr(training_run, model_attr, None)
            if model and model_attr in model_metadata:
                existing_task_id, existing_date = model_metadata[model_attr]
                if existing_task_id and not model.task_id:
                    model.task_id = existing_task_id
                if existing_date and not model.date:
                    model.date = existing_date

    def _save_task_groups(self, training_run: TrainingRun, run_id: int):
        for task_group_id in training_run.task_group_ids:
            if self.db.has_task_group_config(task_group_id):
                continue

            config = self.gcs_collector.get_config(
                training_run.langpair, training_run.name, task_group_id
            )
            if config:
                logger.debug("Loaded config from GCS")
                self.db.upsert_task_group(run_id, task_group_id, config)
                if not training_run.experiment_config:
                    training_run.experiment_config = config
            else:
                self.db.upsert_task_group(run_id, task_group_id)

    def _get_or_fetch_tasks(self, training_run: TrainingRun, run_id: int) -> list[Task]:
        tasks = []
        for task_group_id in training_run.task_group_ids:
            if self.db.is_task_group_expired(task_group_id):
                logger.debug("Skipping expired task group")
                continue

            group_tasks = self.db.get_tasks_for_task_group(task_group_id)
            if group_tasks:
                logger.debug(f"Using {len(group_tasks)} cached tasks")
                tasks.extend(group_tasks)
                continue

            task_objects, tc_tasks = self.tc_collector.get_tasks_for_group(task_group_id)
            if task_objects:
                logger.info(f"Fetched {len(task_objects)} tasks from TaskCluster")
                tasks.extend(task_objects)
                self.db.write_tasks(tc_tasks, task_group_id)
            elif not tc_tasks:
                self.db.mark_task_group_expired(task_group_id)

        return tasks

    def _supplement_with_task_data(
        self, training_run: TrainingRun, tasks: list[Task], run_id: int
    ):
        self.tc_collector.update_date_started(training_run, tasks)
        self.tc_collector.supplement_models_with_task_metadata(training_run, tasks)

        if not self._has_corpora_in_db(run_id):
            self.tc_collector.collect_corpora(training_run, tasks)

    def _has_corpora_in_db(self, run_id: int) -> bool:
        corpora = self.db.get_corpora_for_run(run_id)
        return len(corpora) > 0

    def _set_fallback_date(self, training_run: TrainingRun):
        gcs_date = self.gcs_collector.get_date(training_run)
        if gcs_date:
            training_run.date_started = gcs_date

    def _save_training_run_data(self, training_run: TrainingRun, run_id: int):
        self.db.upsert_training_run(training_run)

        self.db.write_corpus(run_id, "parallel", 1, training_run.parallel_corpus_aligned)
        self.db.write_corpus(
            run_id, "backtranslations", 1, training_run.backtranslations_corpus_aligned
        )
        self.db.write_corpus(run_id, "distillation", 1, training_run.distillation_corpus_aligned)
        self.db.write_corpus(run_id, "parallel", 0, training_run.parallel_corpus)
        self.db.write_corpus(run_id, "backtranslations", 0, training_run.backtranslations_corpus)
        self.db.write_corpus(run_id, "distillation", 0, training_run.distillation_corpus)

        self.db.write_model(run_id, "backward", training_run.backwards)
        self.db.write_model(run_id, "teacher_1", training_run.teacher_1)
        self.db.write_model(run_id, "teacher_2", training_run.teacher_2)
        self.db.write_model(run_id, "student", training_run.student)
        self.db.write_model(run_id, "student_finetuned", training_run.student_finetuned)
        self.db.write_model(run_id, "student_quantized", training_run.student_quantized)
        model_id = self.db.write_model(run_id, "student_exported", training_run.student_exported)

        if model_id and not self.db.has_export(model_id):
            self._save_export_metadata(training_run, model_id)

    def _save_export_metadata(self, training_run: TrainingRun, model_id: int):
        if not training_run.student_exported or not training_run.student_exported.artifact_folder:
            return

        metadata = self.gcs_collector.get_export_metadata(
            training_run.student_exported.artifact_folder
        )
        if metadata:
            self.db.write_export(model_id, metadata)

    def _update_release_statuses(self):
        logger.info("Updating release statuses from Firefox Remote Settings")
        try:
            hash_to_status = ReleaseStatusCollector.fetch_deployed_models()
            logger.info(f"Found {len(hash_to_status)} deployed models in Remote Settings")
            self.db.update_release_statuses(hash_to_status)
        except Exception as e:
            logger.warning(f"Failed to update release statuses: {e}")

    def _finalize_database(self, upload: bool):
        self.db.conn.execute("ANALYZE")
        self.db.conn.commit()
        self.db.conn.close()

        if upload:
            self.gcs.upload_sqlite(SQLITE_PATH)
        else:
            logger.info(f"Wrote {SQLITE_PATH}")


def get_completed_time_from_task(task: Task) -> Optional[datetime]:
    if task.state == "completed" and task.resolved_date:
        return str_to_datetime(task.resolved_date)
    return None


def get_config(action_task_id: str) -> Optional[dict]:
    try:
        return get_artifact(action_task_id, "public/parameters.yml")["training_config"]
    except Exception:
        return None


def str_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")


def match_by_label(pattern: str) -> Callable[[Task], bool]:
    def _match(task: Task) -> bool:
        if not task.state or task.state != "completed":
            return False
        return re.match(pattern, task.task_name or "") is not None

    return _match


def find_latest_task(tasks: list[Task], match_func: Callable[[Task], bool]) -> Optional[Task]:
    matching_tasks = [task for task in tasks if match_func(task)]
    if not matching_tasks:
        return None
    return max(
        matching_tasks, key=lambda t: datetime.strptime(t.created_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Translations DB updater - Build SQLite database from GCS and Taskcluster data"
    )
    parser.add_argument(
        "--db-path", type=Path, help="The path to the local SQLite file", default=SQLITE_PATH
    )
    parser.add_argument("--upload", action="store_true", help="Upload the DB file to GCS")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild database from scratch instead of incremental update",
    )

    args = parser.parse_args()

    updater = Updater()
    updater.build_database(upload=args.upload, overwrite=args.overwrite, db_path=args.db_path)


if __name__ == "__main__":
    main()
