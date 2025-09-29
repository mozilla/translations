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
MODEL_REGISTRY_DIR = ROOT_DIR / "data/model-registry"
SQLITE_PATH = MODEL_REGISTRY_DIR / "model-registry.db"
SQLITE_GCS_OBJECT = "models/model-registry.db"

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

    def get_config_from_gcs(
        self, langpair: str, training_run_name: str, task_group_id: str
    ) -> Optional[dict]:
        import yaml

        config_path = f"experiments/{langpair}/{training_run_name}_{task_group_id}/config.yml"
        try:
            blob = self.bucket.get_blob(config_path)
            if blob and blob.exists():
                config_text = blob.download_as_text()
                return yaml.safe_load(config_text)
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
        return None


class Updater:
    def __init__(self):
        self.gcs = GCSClient(BUCKET_NAME)
        self.tc = TaskClusterClient()

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

    def get_tasks_for_training_run(
        self, training_run: TrainingRun, db: DatabaseManager
    ) -> list[Task]:
        # Try database first
        db_tasks = db.get_tasks_for_training_run(training_run.langpair, training_run.name)
        if db_tasks:
            print(f"Using SQLite database: {len(db_tasks)} tasks for {training_run.name}")
            self._update_date_started(training_run, db_tasks)
            return db_tasks

        # Fetch from TaskCluster and store in database
        tasks = []
        for task_group_id in training_run.task_group_ids:
            # Check database for specific task group
            group_tasks = db.get_tasks_for_task_group(task_group_id)
            if group_tasks:
                print(f"Using SQLite database: {len(group_tasks)} tasks from {task_group_id}")
                tasks.extend(group_tasks)
                continue

            # Fetch from TaskCluster and store in database
            tc_tasks = self.tc.get_tasks_for_group(task_group_id)
            if tc_tasks:
                print(f"Fetched: {len(tc_tasks)} tasks from {task_group_id}")
                # Convert dict tasks to Task objects
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

                tasks.extend(task_objects)
                db.write_tasks(tc_tasks, training_run.langpair, training_run.name, task_group_id)

        self._update_date_started(training_run, tasks)
        return tasks

    def _update_date_started(self, training_run: TrainingRun, tasks: list[Task]):
        for task in tasks:
            date = str_to_datetime(task.created_date)
            if training_run.date_started is None or date < training_run.date_started:
                training_run.date_started = date

    def collect_experiment_config(self, training_run: TrainingRun):
        for task_group_id in training_run.task_group_ids:
            config = self.gcs.get_config_from_gcs(
                training_run.langpair, training_run.name, task_group_id
            )
            if config:
                training_run.experiment_config = config
                print(f"Loaded config from GCS for {training_run.name} ({task_group_id})")
                break

    def collect_flores_comparisons(
        self, training_run: TrainingRun, comet_results: dict, bleu_results: dict
    ):
        comet = comet_results.get(training_run.langpair)
        if comet:
            training_run.comet_flores_comparison = comet["flores-test"]

        bleu = bleu_results.get(training_run.langpair)
        if bleu:
            training_run.bleu_flores_comparison = bleu["flores-test"]

    def collect_models(
        self,
        tasks: list[Task],
        training_run: TrainingRun,
    ):
        # Backwards model
        backwards = find_latest_task(
            tasks, match_by_label(r"^train-backwards-|^backtranslations-train-backwards-model-")
        )
        if backwards:
            training_run.backwards = self._get_model(
                backwards, training_run, tasks, "backwards", "backward", "backwards"
            )

        # Teacher models
        teacher_1 = find_latest_task(
            tasks, match_by_label(r"^train-teacher-.*-1|^train-teacher-model-.*-1")
        )
        if teacher_1:
            training_run.teacher_1 = self._get_model(
                teacher_1, training_run, tasks, "teacher", "teacher0", "teacher0"
            )

        teacher_2 = find_latest_task(tasks, match_by_label(r"^train-teacher-model-.*-2"))
        if teacher_2:
            training_run.teacher_2 = self._get_model(
                teacher_2, training_run, tasks, "teacher", "teacher1", "teacher1"
            )

        # Student models
        student = find_latest_task(
            tasks, match_by_label(r"^train-student-|^distillation-student-model-train-")
        )
        if student:
            training_run.student = self._get_model(
                student, training_run, tasks, "student", "student", "student"
            )

        student_finetuned = find_latest_task(
            tasks, match_by_label(r"^finetune-student|^distillation-student-model-finetune-")
        )
        if student_finetuned:
            training_run.student_finetuned = self._get_model(
                student_finetuned,
                training_run,
                tasks,
                "finetuned-student",
                "student-finetuned",
                "student-finetuned",
            )

        student_quantized = find_latest_task(tasks, match_by_label(r"^quantize-"))
        if student_quantized:
            training_run.student_quantized = self._get_model(
                student_quantized, training_run, tasks, "quantized", "quantized", "speed"
            )

        student_exported = find_latest_task(tasks, match_by_label(r"^export-"))
        if student_exported:
            training_run.student_exported = self._get_model(
                student_exported, training_run, tasks, "export", "exported", "exported"
            )
            if training_run.student_quantized:
                training_run.student_exported.flores = training_run.student_quantized.flores

    def collect_corpora(self, training_run: TrainingRun, tasks: list[Task]):
        # Word aligned corpora
        training_run.parallel_corpus_aligned = self._get_word_aligned_corpus(
            training_run, find_latest_task(tasks, match_by_label(r"^corpus-align-parallel-"))
        )

        training_run.backtranslations_corpus_aligned = self._get_word_aligned_corpus(
            training_run,
            find_latest_task(
                tasks,
                match_by_label(r"^alignments-backtranslated-|^corpus-align-backtranslations-"),
            ),
        )

        training_run.distillation_corpus_aligned = self._get_word_aligned_corpus(
            training_run,
            find_latest_task(
                tasks, match_by_label(r"^alignments-student-|^corpus-align-distillation-")
            ),
        )

        # Raw corpora
        training_run.parallel_corpus = self._get_corpus(
            training_run,
            find_latest_task(tasks, match_by_label(r"^merge-corpus-|^corpus-merge-parallel-")),
        )

        training_run.backtranslations_corpus = self._get_mono_corpus(training_run, tasks)

        training_run.distillation_corpus = self._get_corpus(
            training_run,
            find_latest_task(tasks, match_by_label(r"^distillation-corpus-final-filtering-")),
        )

    def _get_model(
        self,
        task: Task,
        training_run: TrainingRun,
        all_tasks: list[Task],
        tc_model_name: str,
        gcs_model_name: str,
        gcs_eval_name: str,
    ) -> Model:
        task_group_id = task.task_group_id
        model = Model(
            task_group_id=task_group_id,
            task_id=task.task_id,
            task_name=task.task_name,
            date=get_completed_time_from_task(task),
        )

        # Get evaluation
        flores_blob = self._get_flores_eval_blob(
            training_run, task_group_id, gcs_eval_name, tc_model_name
        )
        if not flores_blob:
            eval_task = self._find_eval_task(all_tasks, tc_model_name, gcs_model_name)
            if eval_task:
                flores_blob = self._get_flores_eval_blob(
                    training_run, eval_task.task_group_id, gcs_eval_name, tc_model_name
                )

        if flores_blob:
            model.flores = self._parse_flores_results(flores_blob, tc_model_name)

        # Get artifacts
        prefix = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/{gcs_model_name}/"
        )
        model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{prefix}"

        blobs = self.gcs.list_blobs(prefix=prefix)
        model.artifact_urls = [
            f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
        ]

        return model

    def _get_model_without_evals(
        self, task: Task, training_run: TrainingRun, model_name: str
    ) -> Model:
        task_group_id = task.task_group_id
        model = Model(
            task_group_id=task_group_id,
            task_id=task.task_id,
            task_name=task.task_name,
            date=get_completed_time_from_task(task),
        )

        prefix = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/{model_name}/"
        )
        model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{prefix}"

        blobs = self.gcs.list_blobs(prefix=prefix)
        model.artifact_urls = [
            f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
        ]

        return model

    def _get_corpus(self, training_run: TrainingRun, task: Optional[Task]) -> Optional[Corpus]:
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

    def _get_word_aligned_corpus(
        self, training_run: TrainingRun, task: Optional[Task]
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

    def _get_mono_corpus(self, training_run: TrainingRun, tasks: list[Task]) -> Optional[Corpus]:
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

    def _get_flores_eval_blob(
        self, training_run: TrainingRun, task_group_id: str, gcs_eval_name: str, tc_model_name: str
    ):
        # Map tc_model_name to the actual evaluation file prefix
        eval_prefixes = {
            "backwards": "backwards",
            "teacher": "teacher",
            "student": "student",
            "finetuned-student": "finetuned-student",
            "quantized": "quantized",
            "export": "exported",
        }

        eval_prefix = eval_prefixes.get(tc_model_name, tc_model_name)

        for lang in [training_run.source_lang, training_run.langpair]:
            blob_url = (
                f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
                f"evaluation/{gcs_eval_name}/"
                f"{eval_prefix}-flores-devtest-{lang}_devtest.metrics.json"
            )
            print(f"Looking for evaluation blob: {blob_url}")
            blob = self.gcs.get_blob(blob_url)
            if blob and blob.exists():
                print(f"Found evaluation blob: {blob_url}")
                return blob
        print(f"No evaluation blob found for {tc_model_name} in {training_run.name}")
        return None

    def _find_eval_task(self, tasks: list[Task], tc_model_name: str, gcs_model_name: str):
        label_regex = f"^evaluate-{tc_model_name}-flores-"
        if gcs_model_name == "teacher0":
            label_regex = r"^evaluate-teacher-flores-.*1"
        elif gcs_model_name == "teacher1":
            label_regex = r"^evaluate-teacher-flores-.*2"
        return find_latest_task(tasks, match_by_label(label_regex))

    def _parse_flores_results(self, flores_blob, tc_model_name: str) -> Evaluation:
        import json

        results = json.loads(flores_blob.download_as_text())
        comet = results.get("comet", {}).get("score")
        if comet:
            comet *= 100.0
        return Evaluation(
            chrf=results["chrf"]["score"], bleu=results["bleu"]["score"], comet=comet
        )

    def build_database(self, upload: bool):
        db = DatabaseManager(SQLITE_PATH)

        # Fetch external evaluations
        comet_results = fetch_json(
            "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
        )
        bleu_results = fetch_json(
            "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/bleu-results.json"
        )

        runs_by_langpair = self.get_training_runs_by_langpair()
        i = 0
        for training_runs in runs_by_langpair.values():
            # TODO: remove after debugging
            i += 1
            if i == 10:
                break
            for training_run in training_runs:
                # TODO: remove after debugging
                if training_run.name.startswith("retrain_hr"):
                    continue
                print("Processing", training_run.name, training_run.langpair)

                tasks = self.get_tasks_for_training_run(training_run, db)

                self.collect_experiment_config(training_run)
                self.collect_models(tasks, training_run)
                self.collect_flores_comparisons(training_run, comet_results, bleu_results)
                self.collect_corpora(training_run, tasks)

                # Write to database
                run_id = db.upsert_training_run(training_run)
                db.write_run_comparisons(run_id, training_run)

                # Write corpora
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

                # Write models
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
