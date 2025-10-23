"""
Translations DB updater - Processes training runs from TaskCluster and GCS and stores data in SQLite.
"""

import argparse
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests
import taskcluster
from google.cloud import storage
from taskgraph.util.taskcluster import get_artifact, get_artifact_url

from db.models import Evaluation, Corpus, WordAlignedCorpus, Model, TrainingRun, Task
from db.sql import DatabaseManager

warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

PROJECT_NAME = "translations-data-prod"
BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data"
ROOT_DIR = Path(__file__).parent.parent
MODEL_REGISTRY_DIR = ROOT_DIR / "data/db"
SQLITE_PATH = MODEL_REGISTRY_DIR / "db.sqlite"
SQLITE_GCS_OBJECT = "db/db.sqlite"

MODEL_REGISTRY_DIR.mkdir(exist_ok=True)
os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"


class TaskClusterClient:
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
                print("Task group expired:", task_group_id)
            else:
                raise error
        return tasks


class GCSClient:
    def __init__(self, bucket_name: str):
        self.client = storage.Client(project=PROJECT_NAME)
        self.bucket = self.client.get_bucket(bucket_name)

    def get_subdirectories(self, prefix: str) -> set[str]:
        print(f"Listing {BUCKET_NAME}/{prefix}")
        blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")
        prefixes = set()
        for page in blobs.pages:
            prefixes.update(page.prefixes)
        return prefixes

    def upload_sqlite(self, path: Path):
        blob = self.bucket.blob(SQLITE_GCS_OBJECT)
        blob.cache_control = "public, max-age=60"
        blob.content_type = "application/vnd.sqlite3"
        blob.upload_from_filename(path)
        print(f"Uploaded gs://{BUCKET_NAME}/{SQLITE_GCS_OBJECT}")

    def list_blobs(self, prefix: str):
        """List blobs with a workaround for pagination issues"""
        try:
            # Use a different approach that avoids pagination issues
            return list(self.bucket.list_blobs(prefix=prefix, max_results=1000))
        except (TypeError, ValueError) as e:
            print(f"Pagination issue with prefix {prefix}, trying alternative approach: {e}")
            try:
                # Alternative: iterate manually without setting max_results
                blobs = []
                for blob in self.bucket.list_blobs(prefix=prefix):
                    blobs.append(blob)
                    if len(blobs) >= 1000:  # Safety limit
                        break
                return blobs
            except Exception as e2:
                print(f"Alternative approach also failed for {prefix}: {e2}")
                return []

    def get_blob(self, blob_url: str):
        return self.bucket.get_blob(blob_url)


class GCSDataCollector:
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
        print(f"Scanning GCS for models: {prefix}")

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
            print(f"Found model on GCS: {gcs_model_name} -> {kind}")

            model = Model(task_group_id=task_group_id)
            model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{model_dir}"

            blobs = self.gcs.list_blobs(prefix=model_dir)
            model.artifact_urls = [
                f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
            ]

            if blobs:
                earliest_blob = min(blobs, key=lambda b: b.time_created if b.time_created else datetime.max)
                if earliest_blob.time_created:
                    model.date = earliest_blob.time_created
                    print(f"Set model date from GCS: {model.date}")

            eval_blob = self._get_flores_eval(training_run, task_group_id, gcs_model_name)
            if eval_blob:
                model.flores = self._parse_flores_results(eval_blob)
                print(f"Loaded evaluation for {kind}: {model.flores}")
            else:
                print(f"No evaluation found for {kind}")

            models[kind] = model

        return models

    def _get_flores_eval(self, training_run: TrainingRun, task_group_id: str, gcs_model_name: str):
        eval_base = f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/evaluation/"

        eval_dirs = self.gcs.get_subdirectories(eval_base)
        if not eval_dirs:
            print(f"No evaluation directory found at {eval_base}")
            return None

        # Map GCS model names to their evaluation directory names (including legacy names)
        eval_dir_names = [gcs_model_name]
        if gcs_model_name == "quantized":
            eval_dir_names.append("speed")  # "speed" is legacy name for quantized

        # Check for standard flores-devtest in model-specific subdirectories (legacy structure)
        for eval_dir in eval_dirs:
            eval_dir_name = eval_dir.rstrip("/").split("/")[-1]

            # Check if this directory matches our model
            if not any(eval_dir_name == name for name in eval_dir_names):
                continue

            blobs = self.gcs.list_blobs(prefix=eval_dir)
            for blob in blobs:
                if not blob.name.endswith(".metrics.json"):
                    continue

                # Only use standard flores-devtest (no augmentation variants)
                if "-flores-devtest" in blob.name and "-aug-" not in blob.name:
                    print(f"Found standard flores-devtest evaluation: {blob.name}")
                    return blob

        # Check for flores-devtest in dedicated evaluation subdirectories (newer structure)
        for eval_dir in eval_dirs:
            eval_dir_name = eval_dir.rstrip("/").split("/")[-1]

            # Match patterns like "quantized-flores-devtest-en-pl"
            if f"{gcs_model_name}-flores-devtest" in eval_dir_name and "-aug-" not in eval_dir_name:
                blobs = self.gcs.list_blobs(prefix=eval_dir)
                for blob in blobs:
                    if blob.name.endswith(".metrics.json"):
                        print(f"Found flores-devtest evaluation: {blob.name}")
                        return blob

        print(f"No flores-devtest evaluation found for {gcs_model_name} in {eval_base}")
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

    def get_config(self, langpair: str, training_run_name: str, task_group_id: str) -> Optional[dict]:
        import yaml

        config_path = f"experiments/{langpair}/{training_run_name}_{task_group_id}/config.yml"
        try:
            blob = self.gcs.get_blob(config_path)
            if blob and blob.exists():
                config_text = blob.download_as_text()
                return yaml.safe_load(config_text)
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
        return None

    def get_date(self, training_run: TrainingRun) -> Optional[datetime]:
        for task_group_id in training_run.task_group_ids:
            prefix = f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
            blobs = self.gcs.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.time_created:
                    print(f"Using GCS blob creation date: {blob.time_created}")
                    return blob.time_created
        return None


class TaskClusterDataCollector:
    def __init__(self, tc_client: TaskClusterClient):
        self.tc = tc_client

    def get_tasks_for_group(self, task_group_id: str) -> tuple[list[Task], list[dict]]:
        tc_tasks = self.tc.get_tasks_for_group(task_group_id)
        if not tc_tasks:
            return [], []

        print(f"Fetched: {len(tc_tasks)} tasks from {task_group_id}")
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
            "backwards": (r"^train-backwards-|^backtranslations-train-backwards-model-", training_run.backwards),
            "teacher_1": (r"^train-teacher-.*-1|^train-teacher-model-.*-1", training_run.teacher_1),
            "teacher_2": (r"^train-teacher-model-.*-2", training_run.teacher_2),
            "student": (r"^train-student-|^distillation-student-model-train-", training_run.student),
            "student_finetuned": (r"^finetune-student|^distillation-student-model-finetune-", training_run.student_finetuned),
            "student_quantized": (r"^quantize-", training_run.student_quantized),
            "student_exported": (r"^export-", training_run.student_exported),
        }

        for model_name, (pattern, model) in model_task_map.items():
            if model:
                task = find_latest_task(tasks, match_by_label(pattern))
                if task:
                    model.task_id = task.task_id
                    completed_date = get_completed_time_from_task(task)
                    if completed_date:
                        if not model.date:
                            model.date = completed_date
                        else:
                            model_date_naive = model.date.replace(tzinfo=None) if model.date.tzinfo else model.date
                            if completed_date < model_date_naive:
                                model.date = completed_date
                    print(f"Supplemented {model_name} with task metadata")

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


class Updater:
    def __init__(self):
        gcs_client = GCSClient(BUCKET_NAME)
        tc_client = TaskClusterClient()
        self.gcs = gcs_client
        self.gcs_collector = GCSDataCollector(gcs_client)
        self.tc_collector = TaskClusterDataCollector(tc_client)

    def build_database(self, upload: bool):
        db = DatabaseManager(SQLITE_PATH, rebuild=True)

        comet_results = fetch_json(
            "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
        )
        bleu_results = fetch_json(
            "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/bleu-results.json"
        )

        runs_by_langpair = self.gcs_collector.get_training_runs_by_langpair()

        i = 0
        for training_runs in runs_by_langpair.values():
            # if i == 10:
            #     break
            #
            # i += 1
            for training_run in training_runs:
                # # TODO: remove after testing
                #
                # # TODO: remove after testing
                # if "spring-2024" not in training_run.name: # training_run.name != "retrain_hr_fix_names" and
                #     continue
                print("Processing", training_run.name, training_run.langpair)

                for task_group_id in training_run.task_group_ids:
                    self.gcs_collector.collect_models(training_run, task_group_id)

                self._collect_flores_comparisons(training_run, comet_results, bleu_results)
                self.gcs_collector.collect_corpora(training_run)

                run_id = db.upsert_training_run(training_run)

                for task_group_id in training_run.task_group_ids:
                    config = self.gcs_collector.get_config(
                        training_run.langpair, training_run.name, task_group_id
                    )
                    if config:
                        print(f"Loaded config from GCS for {training_run.name} ({task_group_id})")
                        db.upsert_task_group(run_id, task_group_id, config)
                        if not training_run.experiment_config:
                            training_run.experiment_config = config
                    else:
                        db.upsert_task_group(run_id, task_group_id)

                db_tasks = db.get_tasks_for_training_run(run_id)
                if db_tasks:
                    print(f"Using SQLite database: {len(db_tasks)} tasks for {training_run.name}")
                    tasks = db_tasks
                else:
                    tasks = []
                    for task_group_id in training_run.task_group_ids:
                        group_tasks = db.get_tasks_for_task_group(task_group_id)
                        if group_tasks:
                            print(f"Using SQLite database: {len(group_tasks)} tasks from {task_group_id}")
                            tasks.extend(group_tasks)
                            continue

                        task_objects, tc_tasks = self.tc_collector.get_tasks_for_group(task_group_id)
                        if task_objects:
                            tasks.extend(task_objects)
                            db.write_tasks(tc_tasks, task_group_id)

                self.tc_collector.update_date_started(training_run, tasks)
                self.tc_collector.supplement_models_with_task_metadata(training_run, tasks)
                self.tc_collector.collect_corpora(training_run, tasks)

                if not training_run.date_started:
                    gcs_date = self.gcs_collector.get_date(training_run)
                    if gcs_date:
                        training_run.date_started = gcs_date
                        print(f"Set date_started from GCS: {gcs_date}")

                db.upsert_training_run(training_run)
                db.write_run_comparisons(run_id, training_run)

                db.write_corpus(run_id, "parallel", 1, training_run.parallel_corpus_aligned)
                db.write_corpus(
                    run_id, "backtranslations", 1, training_run.backtranslations_corpus_aligned
                )
                db.write_corpus(
                    run_id, "distillation", 1, training_run.distillation_corpus_aligned
                )
                db.write_corpus(run_id, "parallel", 0, training_run.parallel_corpus)
                db.write_corpus(
                    run_id, "backtranslations", 0, training_run.backtranslations_corpus
                )
                db.write_corpus(run_id, "distillation", 0, training_run.distillation_corpus)

                db.write_model(run_id, "backward", training_run.backwards)
                db.write_model(run_id, "teacher_1", training_run.teacher_1)
                db.write_model(run_id, "teacher_2", training_run.teacher_2)
                db.write_model(run_id, "student", training_run.student)
                db.write_model(run_id, "student_finetuned", training_run.student_finetuned)
                db.write_model(run_id, "student_quantized", training_run.student_quantized)
                db.write_model(run_id, "student_exported", training_run.student_exported)

        db.conn.execute("ANALYZE")
        db.conn.commit()
        db.conn.close()

        if upload:
            self.gcs.upload_sqlite(SQLITE_PATH)
        else:
            print(f"Wrote {SQLITE_PATH}")

    @staticmethod
    def _collect_flores_comparisons(
            training_run: TrainingRun, comet_results: dict, bleu_results: dict
    ):
        comet = comet_results.get(training_run.langpair)
        if comet:
            training_run.comet_flores_comparison = comet["flores-test"]

        bleu = bleu_results.get(training_run.langpair)
        if bleu:
            training_run.bleu_flores_comparison = bleu["flores-test"]



# Utility functions
def fetch_json(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


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
    parser.add_argument("--upload", action="store_true", help="Upload the DB file to GCS")

    args = parser.parse_args()

    registry = Updater()
    registry.build_database(args.upload)


if __name__ == "__main__":
    main()
