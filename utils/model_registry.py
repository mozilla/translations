"""
This script processes and organizes translations training runs, including gathering
corpora, models, and evaluation metrics from Taskcluster and Google Cloud Storage.
It structures the data into JSON files for use in the model registry. These files
are persisted to the GCS in the following structure:

gs://moz-fx-translations-data--303e-prod-translations-data/models/en-el
│
├── spring-2024.json                        <- The JSONified class TrainingRun.
├── spring-2024_Fv23lalyTfSfbmGx0YxypA
│   ├── evaluation
│   ├── tasks.json                          <- list[{ task: Task, status: Status }]
│   ├── teacher0
│   └── teacher1
├── spring-2024_O7cfmFR_SuaZNg8d8b0EWQ
│   ├── backward
│   ├── evaluation
│   ├── tasks.json                          <- list[{ task: Task, status: Status }]
│   └── vocab
└── spring-2024_Y3ThG3XkTxG4ROUQK2LpVg
    ├── evaluation
    ├── exported
    ├── quantized
    ├── student
    ├── student-finetuned
    └── tasks.json                          <- list[{ task: Task, status: Status }]

Notes:
 - Google Cloud Storage authentication is required

Features:
- Fetches training runs and associated Taskcluster task groups.
- Extracts corpora (aligned and non-aligned) and stores metadata.
- Collects trained models and their evaluation metrics.
- Generates structured JSON output for static site usage.
- Supports caching of fetched data to optimize repeated runs.

Usage:
 - This can't be run with poetry due to requirement conflicts. First create a venv:

   python -m venv venv && source venv/bin/activate
   pip install google-cloud-storage taskcluster taskcluster-taskgraph

 - Run with aggressive caching:
   python utils/model_registry.py

 - Re-build a single run:
   rm site/model-registry/training-runs/spring-2024-sk-en.json
   python utils/model_registry.py

 - Re-run with a cleared HTTP cache:
   python utils/model_registry.py -- --clear_cache

 - Completely rebuild everything
   python script.py -- --clear_cache --overwrite_runs
"""

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Optional, Union

# Poetry has some install issues with dependencies here:
from google.cloud import storage  # type: ignore
import requests
import taskcluster
import warnings
import shelve
from taskgraph.util.taskcluster import get_artifact, get_artifact_url

# This script is not integrated into a production environment, so suppress the auth warning.
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

PROJECT_NAME = "translations-data-prod"
BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data"
ROOT_DIR = Path(__file__).parent.parent
MODEL_REGISTRY_DIR = ROOT_DIR / "data/model-registry"
TRAINING_RUNS_DIR = MODEL_REGISTRY_DIR / "training-runs"
CACHE_FILE = MODEL_REGISTRY_DIR / "cache.pickle"

MODEL_REGISTRY_DIR.mkdir(exist_ok=True)

os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"

client = storage.Client(project=PROJECT_NAME)
bucket = client.get_bucket(BUCKET_NAME)


def get_gcs_subdirectories(prefix: str, cache: Optional[shelve.Shelf]) -> set[str]:
    """
    Get the subdirectories of the a given prefix for a Google Cloud Storage bucket.
    """
    cache_key = f"get_subdirectories-{BUCKET_NAME}-{prefix}"
    if cache is not None:
        data = cache.get(cache_key, None)
        if data:
            return data

    print(f"Listing {BUCKET_NAME}/{prefix}")
    blobs = bucket.list_blobs(
        prefix=prefix,
        # Specify a delimiter to only return the objects in the directory
        delimiter="/",
    )

    prefixes: set[str] = set()

    for page in blobs.pages:
        prefixes.update(page.prefixes)

    if cache is not None:
        cache[cache_key] = prefixes

    return prefixes


@dataclass
class Evaluation:
    """
    A structured class for storing the evaluations for a model. This gets stored in
    the training run's JSON.
    """

    chrf: Optional[float]
    bleu: Optional[float]
    comet: Optional[float]

    @staticmethod
    def create():
        return Evaluation(
            chrf=None,
            bleu=None,
            comet=None,
        )


@dataclass
class Corpus:
    """
    Each file contains a newline separated "sentence" in the language. Each line
    in the source matches the translation in the target sentence. There is no tokenization
    that is applied to this corpus.
    """

    source_url: str
    source_bytes: int
    target_url: str
    target_bytes: int

    @staticmethod
    def from_task(
        training_run: "TrainingRun",
        task: Optional[dict],
    ) -> Optional["Corpus"]:
        """
        Builds a Corpus from a task if a task is passed in.
        """
        if task is None:
            print("  [corpus] task missing")
            return None

        taskId = task["status"]["taskId"]
        print("  [corpus]", task_name(task))
        print("  [corpus]", task_url(taskId))
        source_url = get_artifact_url(
            taskId,
            f"public/build/corpus.{training_run.source_lang}.zst",
        )
        target_url = get_artifact_url(
            taskId,
            f"public/build/corpus.{training_run.target_lang}.zst",
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            print("  [corpus] corpus missing")
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )

    @staticmethod
    def from_mono_tasks(
        training_run: "TrainingRun",
        tasks: list[dict],
    ) -> Optional["Corpus"]:
        """
        The monolingual files are in separate tasks, so the lookups are a bit different.
        """
        source_task = (
            # This task was renamed
            find_latest_task(tasks, match_by_label(r"^collect-mono-trg-"))
            or find_latest_task(
                tasks, match_by_label(r"^backtranslations-mono-trg-dechunk-translations-")
            )
        )
        target_task = (
            # This task was renamed.
            find_latest_task(tasks, match_by_label(r"^collect-mono-src-"))
            or find_latest_task(
                tasks, match_by_label(r"^distillation-mono-src-dechunk-translations-")
            )
        )

        if source_task is None or target_task is None:
            print("  [corpus] mono tasks missing")
            return None

        print("  [corpus]", task_name(source_task))
        print("  [corpus]", task_url(source_task))
        source_url = get_artifact_url(
            source_task["status"]["taskId"],
            f"public/build/mono.{training_run.source_lang}.zst",
        )

        print("  [corpus]", task_name(target_task))
        print("  [corpus]", task_url(target_task))
        target_url = get_artifact_url(
            target_task["status"]["taskId"],
            f"public/build/mono.{training_run.target_lang}.zst",
        )

        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not source_head.ok or not target_head.ok:
            print("  [corpus] corpus missing")
            return None

        return Corpus(
            source_url=source_url,
            target_url=target_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
        )


@dataclass
class WordAlignedCorpus:
    """
    Each file contains a newline separated "sentence" in the language. Each line
    in the source matches the translation in the target sentence. The text is tokenized
    based on the words, where " ▁ " represents a logical word break that has whitespace
    in the original, while " " represents a logical word break that did not have
    whitespace in the original text.

    Example tokenizations:
    "machine translation" -> "machine ▁ translation"
    "机器翻译" -> "机器 翻译"

    The alignments represent how the source sentence's words are aligned to the target
    sentence words. They are tuples of word indexes. A word on the source sentence
    can map to multiple words in the target and vice versa.

    0-3 1-2 1-4 2-0 2-1 2-5
    0-0 1-1 1-2
    0-0 1-0 1-1 2-1 3-2
    """

    source_url: str
    target_url: str
    alignments_url: str

    source_bytes: int
    target_bytes: int
    alignments_bytes: int

    @staticmethod
    def from_task(
        training_run: "TrainingRun", alignments_task: Optional[dict]
    ) -> Optional["WordAlignedCorpus"]:
        """
        Builds a WordAlignedCorpus from a task if a task is passed in.
        """

        if alignments_task is None:
            print("  [word-aligned-corpus] No alignments task")
            return None

        task_id = alignments_task["status"]["taskId"]
        alignments_url = get_artifact_url(alignments_task, "public/build/corpus.aln.zst")
        print("  [word-aligned-corpus]", task_name(alignments_task), task_url(task_id))
        source_url = get_artifact_url(
            task_id,
            f"public/build/corpus.tok-icu.{training_run.source_lang}.zst",
        )
        target_url = get_artifact_url(
            task_id,
            f"public/build/corpus.tok-icu.{training_run.target_lang}.zst",
        )

        alignments_head = requests.head(alignments_url, allow_redirects=True)
        source_head = requests.head(source_url, allow_redirects=True)
        target_head = requests.head(target_url, allow_redirects=True)

        if not alignments_head.ok or not source_head.ok or not target_head.ok:
            print("  [word-aligned-corpus] could not find the files from task")
            return None

        return WordAlignedCorpus(
            source_url=source_url,
            target_url=target_url,
            alignments_url=alignments_url,
            source_bytes=int(source_head.headers.get("content-length", 0)),
            target_bytes=int(target_head.headers.get("content-length", 0)),
            alignments_bytes=int(alignments_head.headers.get("content-length", 0)),
        )


@dataclass
class Model:
    """
    All of the known information about a given model. This model could be a
    back translation, student, or teacher, etc. This information is JSON serialized.
    """

    date: Optional[datetime]
    config: Optional[dict]
    task_group_id: Optional[dict]
    task_id: Optional[dict]
    task_name: Optional[dict]
    flores: Optional[Evaluation]
    artifact_folder: Optional[str]
    artifact_urls: list[str]

    @staticmethod
    def create():
        return Model(
            date=None,
            config=None,
            task_group_id=None,
            task_id=None,
            task_name=None,
            flores=None,
            artifact_folder=None,
            artifact_urls=[],
        )

    def sync_live_log(self, training_run: "TrainingRun", gcs_model_name: str):
        """
        If no live log was synced, do it now.
        """
        tasks_gcs_path = f"models/{training_run.langpair}/{training_run.name}_{self.task_group_id}/{gcs_model_name}/live.log"
        live_log_blob = bucket.blob(tasks_gcs_path)

        if live_log_blob.exists():
            return

        url = get_artifact_url(
            self.task_id,
            "public/logs/live.log",
        )
        print(f"  [log] Downloading {url}")
        response = requests.get(url)
        if response.ok:
            print(f"  [log] Uploading live log to GCS {tasks_gcs_path}")
            live_log_blob.upload_from_string(response.text)
        else:
            print(
                f"  [log] The live log failed to download with status {response.status_code}: {response.text}"
            )


@dataclass
class TrainingRun:
    """
    A training run has a unique name, and language pair. It can take multiple task groups
    to complete a training run. This struct represents the collection of all tasks sorted
    by date, with the most recent task being picked for the final artifacts.
    """

    name: str  # e.g. "spring-2024"
    langpair: str
    source_lang: str
    target_lang: str
    task_group_ids: list[str]
    date_started: Optional[datetime]

    # e.g. { "google": 0.8708, ... }
    comet_flores_comparison: dict[str, float]
    bleu_flores_comparison: dict[str, float]

    # Aligned Corpora
    parallel_corpus_aligned: Optional[WordAlignedCorpus]
    backtranslations_corpus_aligned: Optional[WordAlignedCorpus]
    distillation_corpus_aligned: Optional[WordAlignedCorpus]

    # Non-aligned Corpora
    parallel_corpus: Optional[Corpus]
    backtranslations_corpus: Optional[Corpus]
    distillation_corpus: Optional[Corpus]

    # Models
    backwards: Optional[Model]
    teacher_1: Optional[Model]
    teacher_2: Optional[Model]
    teacher_ensemble_flores: Optional[Evaluation]

    student: Optional[Model]
    student_finetuned: Optional[Model]
    student_quantized: Optional[Model]
    student_exported: Optional[Model]

    @staticmethod
    def create(name: str, task_group_ids: list[str], langpair):
        source_lang, target_lang = langpair.split("-")
        return TrainingRun(
            name=name,
            langpair=langpair,
            source_lang=source_lang,
            target_lang=target_lang,
            task_group_ids=task_group_ids,
            date_started=None,
            comet_flores_comparison={},
            bleu_flores_comparison={},
            # Aligned Corpora
            parallel_corpus_aligned=None,
            backtranslations_corpus_aligned=None,
            distillation_corpus_aligned=None,
            # Non-aligned Corpora
            parallel_corpus=None,
            backtranslations_corpus=None,
            distillation_corpus=None,
            # Models
            backwards=None,
            teacher_1=None,
            teacher_2=None,
            teacher_ensemble_flores=None,
            student=None,
            student_finetuned=None,
            student_quantized=None,
            student_exported=None,
        )

    def get_json_cache_path(self) -> Path:
        """
        The JSON gets a local cache, which is useful for viewing and debugging the
        generate artifacts.
        """
        return TRAINING_RUNS_DIR / f"{self.name}-{self.langpair}.json"

    def get_json_gcs_path(self):
        """
        The path the JSON in the GCS bucket, used with the bucket.blob interface.
        """
        return f"models/{self.langpair}/{self.name}.json"

    def get_json_gcs_url(self):
        """
        The full gs:// url path.
        """
        return f"gs://{BUCKET_NAME}/{self.get_json_gcs_path()}"


class JsonEncoder(json.JSONEncoder):
    """Converts a dataclass into a JSON serializable struct"""

    def default(self, o: Any):
        if is_dataclass(o):
            return asdict(o)  # type: ignore
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def get_training_runs_by_langpair(
    cache: Optional[shelve.Shelf],
) -> dict[str, list[TrainingRun]]:
    """
    Training runs are stored in the following structure. Extract out the information into
    a structured format.

    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_DtSyAeaVRoGNZDnUKscGWw/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_NPlcq4JZRRCj0ksitTDSVw/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_Ov3G4D_DRJa-4qTlILkPhg/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_TSvbd6EuTmGayUQtIP3Lbg/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_bQQme71PS4eZRDl3NM-kgA/
    gs://moz-fx-translations-data--303e-prod-translations-data/models/en-cs/spring-2024_bbjDBFoDTNGSUo2if3ET_A/
    """

    # e.g. ['en-fi', 'en-da', 'en-hu', 'en-sr', ...]
    langpairs = [
        model_prefix.split("/")[1] for model_prefix in get_gcs_subdirectories("models/", cache)
    ]

    runs_by_langpair: dict[str, list[TrainingRun]] = {}
    training_runs_by_name: dict[str, TrainingRun] = {}

    for langpair in langpairs:
        training_runs: list[TrainingRun] = []
        runs_by_langpair[langpair] = training_runs

        # e.g { "models/en-lv/spring-2024_J3av8ewURni5QQqP2u3QRg/", ... }
        for task_group_prefix in get_gcs_subdirectories(f"models/{langpair}/", cache):
            # e.g. "spring-2024_J3av8ewURni5QQqP2u3QRg"
            name_task_group_tuple = task_group_prefix.split("/")[2]
            # Some old test runs without group ID
            if name_task_group_tuple.endswith("None"):
                continue

            # Task Group IDs are 22 letters long, and contain "_", so don't split on "_"
            # which is used as a delimiter. Only rely on the hard coded length, which
            # is simpler than using this regex:
            # https://github.com/taskcluster/taskcluster/blob/3249015448f795d30ebbc3c3304c3b6d86c39284/services/auth/schemas/constants.yml#L11-L12
            name = name_task_group_tuple[:-23]
            task_group_id = name_task_group_tuple[-22:]
            key = f"{langpair} {name}"

            training_task_group = training_runs_by_name.get(key, None)
            if training_task_group:
                training_task_group.task_group_ids.append(task_group_id)
                # Sort the task group ids so the generated artifact is stable.
                training_task_group.task_group_ids.sort()
            else:
                training_task_group = TrainingRun.create(
                    name=name, task_group_ids=[task_group_id], langpair=langpair
                )
                training_runs.append(training_task_group)
                training_runs_by_name[key] = training_task_group

    return runs_by_langpair


def print_training_runs_tree(runs_by_training_pair: dict[str, list[TrainingRun]]):
    """
    This is a debugging function that prints the training runs as a tree. This function
    was AI generated, but human reviewed.
    """
    last_langpair_index = len(runs_by_training_pair) - 1

    print("\nTraining Runs")
    for langpair_index, (langpair, training_runs) in enumerate(runs_by_training_pair.items()):
        prefix_langpair = "└──" if langpair_index == last_langpair_index else "├──"
        print(f"{prefix_langpair} {langpair}")

        last_run_index = len(training_runs) - 1
        for run_index, training_run in enumerate(training_runs):
            prefix_run = "└──" if run_index == last_run_index else "├──"
            connector = "    " if langpair_index == last_langpair_index else "│   "
            print(f"{connector}{prefix_run} {training_run.name}")

            task_groups = training_run.task_group_ids
            last_task_index = len(task_groups) - 1
            for task_index, task_group_id in enumerate(task_groups):
                prefix_task = "└──" if task_index == last_task_index else "├──"
                sub_connector = "    " if run_index == last_run_index else "│   "
                print(
                    f"{connector}{sub_connector}{prefix_task} https://firefox-ci-tc.services.mozilla.com/tasks/groups/{task_group_id}"
                )


# The JSON from the evaluations repo.
# {
#   "en-af": {
#     "flores-dev": { "nllb": 0.8566, "google": 0.8708, ... }
#     "flores-test": {...}
#     ...
#   },
#   ...
# }
EvaluationJson = dict[str, dict[str, dict[str, float]]]

Task = dict[str, dict]


def iterate_training_runs(
    runs_by_training_pair: dict[str, list[TrainingRun]],
    upload: bool,
    cache: Optional[shelve.Shelf],
):
    """
    Reduce the complexity required for iterating over the training runs and their tasks.
    """
    for training_runs in runs_by_training_pair.values():
        for training_run in training_runs:
            yield training_run, get_tasks_in_all_runs(training_run, upload, cache)

import sqlite3

SQLITE_PATH = MODEL_REGISTRY_DIR / "model-registry.db"
SQLITE_GCS_OBJECT = "models/model-registry.db"

def init_db(sqlite_path: Path) -> sqlite3.Connection:
    if sqlite_path.exists():
        sqlite_path.unlink()  # rebuild from scratch each run
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE training_runs (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      langpair TEXT NOT NULL,
      source_lang TEXT NOT NULL,
      target_lang TEXT NOT NULL,
      date_started TEXT,
      UNIQUE (langpair, name)
    );

    CREATE TABLE run_comparisons (
      run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
      metric TEXT NOT NULL,          -- 'bleu' | 'comet'
      provider TEXT NOT NULL,        -- 'google', 'nllb', etc.
      score REAL NOT NULL
    );

    CREATE TABLE corpora (
      id INTEGER PRIMARY KEY,
      run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
      type TEXT NOT NULL,            -- 'parallel' | 'backtranslations' | 'distillation'
      aligned INTEGER NOT NULL,      -- 0/1
      source_url TEXT,
      target_url TEXT,
      alignments_url TEXT,
      source_bytes INTEGER,
      target_bytes INTEGER,
      alignments_bytes INTEGER
    );
    CREATE INDEX idx_corpora_run_type ON corpora(run_id, type, aligned);

    CREATE TABLE models (
      id INTEGER PRIMARY KEY,
      run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
      kind TEXT NOT NULL,            -- 'backward' | 'teacher_1' | 'teacher_2' | 'student' | ...
      date TEXT,
      task_group_id TEXT,
      task_id TEXT,
      task_name TEXT,
      artifact_folder TEXT,
      UNIQUE (run_id, kind)
    );
    CREATE INDEX idx_models_run_kind ON models(run_id, kind);
    CREATE INDEX idx_models_date ON models(date);

    CREATE TABLE evaluations (
      model_id INTEGER PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
      chrf REAL, bleu REAL, comet REAL
    );
    CREATE INDEX idx_eval_bleu ON evaluations(bleu);
    CREATE INDEX idx_eval_comet ON evaluations(comet);

    CREATE TABLE artifacts (
      id INTEGER PRIMARY KEY,
      model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      url TEXT NOT NULL
    );
    """)
    return conn


def upsert_training_run(conn: sqlite3.Connection, tr: "TrainingRun") -> int:
    cur = conn.execute("""
      INSERT INTO training_runs(name, langpair, source_lang, target_lang, date_started)
      VALUES(?,?,?,?,?)
      ON CONFLICT(langpair, name) DO UPDATE SET
        source_lang=excluded.source_lang,
        target_lang=excluded.target_lang,
        date_started=excluded.date_started
    """, (tr.name, tr.langpair, tr.source_lang, tr.target_lang,
          tr.date_started.isoformat() if tr.date_started else None))
    # fetch id
    row = conn.execute("SELECT id FROM training_runs WHERE langpair=? AND name=?",
                       (tr.langpair, tr.name)).fetchone()
    return int(row[0])

def write_run_comparisons(conn: sqlite3.Connection, run_id: int, tr: "TrainingRun"):
    conn.execute("DELETE FROM run_comparisons WHERE run_id=?", (run_id,))
    for provider, score in (tr.bleu_flores_comparison or {}).items():
        conn.execute("INSERT INTO run_comparisons(run_id, metric, provider, score) VALUES(?,?,?,?)",
                     (run_id, "bleu", provider, float(score)))
    for provider, score in (tr.comet_flores_comparison or {}).items():
        conn.execute("INSERT INTO run_comparisons(run_id, metric, provider, score) VALUES(?,?,?,?)",
                     (run_id, "comet", provider, float(score)))

def write_corpus(conn: sqlite3.Connection, run_id: int, typ: str, aligned: int, c):
    if not c: return
    conn.execute("""
      INSERT INTO corpora(run_id, type, aligned, source_url, target_url, alignments_url,
                          source_bytes, target_bytes, alignments_bytes)
      VALUES(?,?,?,?,?,?,?,?,?)
    """, (run_id, typ, aligned,
          getattr(c, "source_url", None),
          getattr(c, "target_url", None),
          getattr(c, "alignments_url", None),
          getattr(c, "source_bytes", None),
          getattr(c, "target_bytes", None),
          getattr(c, "alignments_bytes", None)))

def write_model(conn: sqlite3.Connection, run_id: int, kind: str, m: Optional["Model"]) -> Optional[int]:
    if not m: return None
    conn.execute("""
      INSERT INTO models(run_id, kind, date, task_group_id, task_id, task_name, artifact_folder)
      VALUES(?,?,?,?,?,?,?)
      ON CONFLICT(run_id, kind) DO UPDATE SET
        date=excluded.date,
        task_group_id=excluded.task_group_id,
        task_id=excluded.task_id,
        task_name=excluded.task_name,
        artifact_folder=excluded.artifact_folder
    """, (run_id, kind,
          m.date.isoformat() if m.date else None,
          m.task_group_id, m.task_id, m.task_name, m.artifact_folder))
    row = conn.execute("SELECT id FROM models WHERE run_id=? AND kind=?", (run_id, kind)).fetchone()
    model_id = int(row[0])
    # evaluation
    conn.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
    if m.flores:
        conn.execute("INSERT INTO evaluations(model_id, chrf, bleu, comet) VALUES(?,?,?,?)",
                     (model_id, m.flores.chrf, m.flores.bleu, m.flores.comet))
    # artifacts
    conn.execute("DELETE FROM artifacts WHERE model_id=?", (model_id,))
    for url in (m.artifact_urls or []):
        conn.execute("INSERT INTO artifacts(model_id, url) VALUES(?,?)", (model_id, url))
    return model_id

def upload_sqlite_to_gcs(path: Path):
    blob = bucket.blob(SQLITE_GCS_OBJECT)
    blob.cache_control = "public, max-age=60"  # or use manifest pattern; tune to your needs
    blob.content_type = "application/vnd.sqlite3"
    blob.upload_from_filename(path)
    print(f"Uploaded gs://{BUCKET_NAME}/{SQLITE_GCS_OBJECT}")



def build_sqlite_for_training_runs(
    runs_by_training_pair: dict[str, list[TrainingRun]],
    overwrite_runs: bool,   # ignored now; DB is rebuilt fresh
    upload: bool,
    cache: Optional[shelve.Shelf],
):
    conn = init_db(SQLITE_PATH)

    # Pre-fetch external evaluation JSON once (as before)
    comet_results_by_langpair: EvaluationJson = fetch_json(
        "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
    )
    bleu_results_by_langpair: EvaluationJson = fetch_json(
        "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/bleu-results.json"
    )

    i = 0

    for tr, tasks in iterate_training_runs(runs_by_training_pair, upload, cache):
        i += 1
        if i == 10:
            break
        print("Processing", tr.name, tr.langpair)

        collect_models(tasks, tr, upload)
        collect_flores_comparisons(tr, comet_results_by_langpair, bleu_results_by_langpair)
        collect_corpora(tr, tasks)

        run_id = upsert_training_run(conn, tr)
        write_run_comparisons(conn, run_id, tr)

        # corpora (aligned and non-aligned)
        write_corpus(conn, run_id, "parallel", 1, tr.parallel_corpus_aligned)
        write_corpus(conn, run_id, "backtranslations", 1, tr.backtranslations_corpus_aligned)
        write_corpus(conn, run_id, "distillation", 1, tr.distillation_corpus_aligned)
        write_corpus(conn, run_id, "parallel", 0, tr.parallel_corpus)
        write_corpus(conn, run_id, "backtranslations", 0, tr.backtranslations_corpus)
        write_corpus(conn, run_id, "distillation", 0, tr.distillation_corpus)

        # models (+ evals + artifacts)
        write_model(conn, run_id, "backward", tr.backwards)
        write_model(conn, run_id, "teacher_1", tr.teacher_1)
        write_model(conn, run_id, "teacher_2", tr.teacher_2)
        write_model(conn, run_id, "student", tr.student)
        write_model(conn, run_id, "student_finetuned", tr.student_finetuned)
        write_model(conn, run_id, "student_quantized", tr.student_quantized)
        write_model(conn, run_id, "student_exported", tr.student_exported)

    # finalize
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    if upload:
        upload_sqlite_to_gcs(SQLITE_PATH)
    else:
        print(f"Wrote {SQLITE_PATH}")




def get_tasks_in_all_runs(
    training_run: TrainingRun, upload: bool, cache: Optional[shelve.Shelf]
) -> list[Task]:
    """
    Get a flat list of the tasks in every TaskGroup of the training run. These are
    tasks are arbitrarily sorted. If picking a task from it use find_latest_task and
    find_earliest_task.

    Note that the tasks will be pulled from GCS first, and TaskCluster second. If the
    --upload parameter is set, the tasks will be saved to GCS storage if they are not
    present.
    """
    queue = taskcluster.Queue(options={"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})

    tasks_in_all_runs: list[Task] = []
    for task_group_id in training_run.task_group_ids:
        cache_key = f"list_task_group-{task_group_id}"
        tasks = None
        prefix = "Fetched"
        # e.g.
        # "models/en-sk/spring-2024_MRw1u6KIRgO056Isf0GKpA/tasks.json"
        tasks_gcs_path = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/tasks.json"
        )
        tasks_blob = bucket.blob(tasks_gcs_path)

        if cache is not None:
            tasks = cache.get(cache_key, None)
            if tasks is not None:
                prefix = "Using cached"
                if upload and not tasks_blob.exists():
                    print("Uploading tasks (from cache) to GCS:", tasks_gcs_path)
                    tasks_blob.upload_from_string(json.dumps(tasks, indent=2))

        if tasks is None:
            tasks = []
            if tasks_blob.exists():
                print(f"Downloading tasks: {tasks_gcs_path}")
                tasks = json.loads(tasks_blob.download_as_string())
                assert isinstance(tasks, list), "Expected the tasks to be a list"
            else:
                try:
                    list_task_group: Any = queue.listTaskGroup(task_group_id)
                    tasks.extend(list_task_group["tasks"])

                    # Do a bounded lookup of more tasks. 10 should be a reasonable limit.
                    for _ in range(10):
                        if not list_task_group.get("continuationToken", None):
                            break
                        list_task_group: Any = queue.listTaskGroup(
                            task_group_id,
                            continuationToken=list_task_group["continuationToken"],
                        )
                        tasks.extend(list_task_group["tasks"])
                except taskcluster.exceptions.TaskclusterRestFailure as error:
                    # 404 errors indicate expired task groups.
                    if error.status_code == 404:
                        print("Task group expired:", task_group_id)
                    elif error.status_code == 400:
                        raise error
                    else:
                        raise error

                if upload:
                    print("Uploading tasks to GCS:", tasks_gcs_path)
                    tasks_blob.upload_from_string(json.dumps(tasks, indent=2))

        if cache is not None:
            cache[cache_key] = tasks

        tasks_in_all_runs.extend(tasks)

        print(f"{prefix} {len(tasks)} tasks from {task_group_id}")
        for task in tasks:
            date = str_to_datetime(task["task"]["created"])

            if training_run.date_started is None or date < training_run.date_started:
                training_run.date_started = date

    return tasks_in_all_runs


def collect_flores_comparisons(
    training_run: TrainingRun,
    comet_results_by_langpair: EvaluationJson,
    bleu_results_by_langpair: EvaluationJson,
):
    """
    Mutate the training run with the flores evaluations.
    """
    comet_results = comet_results_by_langpair.get(training_run.langpair, None)
    if comet_results:
        training_run.comet_flores_comparison = comet_results["flores-test"]
    bleu_results = bleu_results_by_langpair.get(training_run.langpair, None)
    if bleu_results:
        training_run.bleu_flores_comparison = bleu_results["flores-test"]


def collect_models(tasks: list[Task], training_run: TrainingRun, upload: bool):
    """
    Lookup models from Google Cloud Storage.
    """
    backwards = find_latest_task(
        tasks,
        # This was renamed
        match_by_label(r"^train-backwards-")
        or match_by_label(r"^backtranslations-train-backwards-model-"),
    )
    if backwards:
        training_run.backwards = get_model_without_evals(
            backwards,
            training_run,
            upload,
            model_name="backward",
        )

    train_teacher_1 = find_latest_task(
        tasks,
        match_by_label(r"^train-teacher-.*-1") or match_by_label(r"^train-teacher-model-.*-1"),
    )
    if train_teacher_1:
        training_run.teacher_1 = get_model(
            train_teacher_1,
            training_run,
            tasks,
            upload,
            tc_model_name="teacher",
            gcs_model_name="teacher0",
            gcs_eval_name="teacher0",
        )

    train_teacher_2 = find_latest_task(tasks, match_by_label(r"^train-teacher-model-.*-2"))
    if train_teacher_2:
        training_run.teacher_2 = get_model(
            train_teacher_2,
            training_run,
            tasks,
            upload,
            tc_model_name="teacher",
            gcs_model_name="teacher1",
            gcs_eval_name="teacher1",
        )

    student_finetuned = find_latest_task(
        tasks,
        match_by_label(r"^finetune-student")
        or match_by_label(r"^distillation-student-model-finetune-"),
    )
    if student_finetuned:
        training_run.student_finetuned = get_model(
            student_finetuned,
            training_run,
            tasks,
            upload,
            tc_model_name="finetuned-student",
            gcs_model_name="student-finetuned",
            gcs_eval_name="student-finetuned",
        )

    train_student_task = find_latest_task(
        tasks,
        match_by_label(r"^train-student-")
        or match_by_label(r"^distillation-student-model-train-"),
    )
    if train_student_task:
        training_run.student = get_model(
            train_student_task,
            training_run,
            tasks,
            upload,
            tc_model_name="student",
            gcs_model_name="student",
            gcs_eval_name="student",
        )
    student_quantize_task = find_latest_task(tasks, match_by_label(r"^quantize-"))
    if student_quantize_task:
        training_run.student_quantized = get_model(
            student_quantize_task,
            training_run,
            tasks,
            upload,
            tc_model_name="quantized",
            gcs_model_name="quantized",
            gcs_eval_name="speed",
        )
    student_export_task = find_latest_task(tasks, match_by_label(r"^export-"))
    if student_export_task:
        training_run.student_exported = get_model(
            student_export_task,
            training_run,
            tasks,
            # These logs aren't useful to retain, as there is no training happening here.
            upload=False,
            tc_model_name="export",
            gcs_model_name="exported",
            gcs_eval_name="exported",
        )
        if training_run.student_quantized:
            # The export step doesn't have an explicit eval, so take
            # the one from the quantized step.
            training_run.student_exported.flores = training_run.student_quantized.flores


def collect_corpora(training_run: TrainingRun, tasks: list[Task]):
    """
    Mutate the training run with all of the corpora. Look up both the word aligned
    corpora and the older non-word aligned corpora.
    """
    # Find the word aligned corpora.
    training_run.parallel_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^corpus-align-parallel-")),
    )
    training_run.backtranslations_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        (
            # This task was renamed.
            find_latest_task(tasks, match_by_label(r"^alignments-backtranslated-"))
            or find_latest_task(tasks, match_by_label(r"^corpus-align-backtranslations-"))
        ),
    )
    training_run.distillation_corpus_aligned = WordAlignedCorpus.from_task(
        training_run,
        (
            # The task was renamed.
            find_latest_task(tasks, match_by_label(r"^alignments-student-"))
            or find_latest_task(tasks, match_by_label(r"^corpus-align-distillation-"))
        ),
    )

    # Find the raw corpora
    training_run.parallel_corpus = Corpus.from_task(
        training_run,
        (
            # This task was renamed.
            find_latest_task(tasks, match_by_label(r"^merge-corpus-"))
            or find_latest_task(tasks, match_by_label(r"^corpus-merge-parallel-"))
        ),
    )
    training_run.backtranslations_corpus = Corpus.from_mono_tasks(
        training_run,
        tasks,
    )
    training_run.distillation_corpus = Corpus.from_task(
        training_run,
        find_latest_task(tasks, match_by_label(r"^distillation-corpus-final-filtering-")),
    )


def get_model(
    task: dict,
    training_run: TrainingRun,
    tasks_in_all_runs: list[dict],
    upload: bool,
    # The model name in Taskcluster tasks.
    tc_model_name: str,
    # The model directory name in GCS.
    gcs_model_name: str,
    # The model directory name in GCS.
    gcs_eval_name: str,
) -> Model:
    """
    Lookup all of the information for a model and collect in a structure.
    """
    task_group_id = task["status"]["taskGroupId"]

    model = Model.create()
    model.config = get_config(task_group_id)
    model.task_group_id = task_group_id
    model.task_id = task["status"]["taskId"]
    model.task_name = task["task"]["metadata"]["name"]
    model.date = get_completed_time(task)

    flores_blob = get_flores_eval_blob(
        training_run,
        task_group_id,
        gcs_eval_name,
        tc_model_name,
    )
    if not flores_blob:
        # The eval wasn't in the same task group as the training.
        label_regex = f"^evaluate-{tc_model_name}-flores-"
        # These don't follow the same format, so adjust the regex. These need to match
        # the current naming convention, and the older one:
        #  - evaluate-teacher-flores-devtest-sk-en-1
        #  - evaluate-teacher-flores-devtest-sk-en-1/2
        if gcs_model_name == "teacher0":
            label_regex = r"^evaluate-teacher-flores-.*1"
        if gcs_model_name == "teacher1":
            label_regex = r"^evaluate-teacher-flores-.*2"

        eval_task = find_latest_task(tasks_in_all_runs, match_by_label(label_regex))
        if eval_task:
            flores_blob = get_flores_eval_blob(
                training_run,
                eval_task["status"]["taskGroupId"],
                gcs_eval_name,
                tc_model_name,
            )

    if flores_blob:
        print(f"  [model] loading {tc_model_name} evals")
        flores_results = json.loads(flores_blob.download_as_text())

        # Older evaluations may not have COMET integrated.
        comet = None
        if "comet" in flores_results:
            comet = flores_results["comet"]["score"] * 100.0

        model.flores = Evaluation(
            chrf=flores_results["chrf"]["score"],
            bleu=flores_results["bleu"]["score"],
            comet=comet,
        )
    else:
        print(f"  [model] {tc_model_name} evals not found")

    prefix = (
        f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/{gcs_model_name}/"
    )
    model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{prefix}"

    # List all of the artifacts.
    print(f"  [model] listing {tc_model_name} files - {model.artifact_folder}")
    blobs: Optional[Iterable[storage.Blob]] = bucket.list_blobs(prefix=prefix)
    if blobs:
        model.artifact_urls = [
            f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
        ]
    else:
        print(f"  [model] no {tc_model_name} files found")

    if upload:
        model.sync_live_log(training_run, gcs_model_name)

    return model


def get_model_without_evals(
    task: dict,
    training_run: TrainingRun,
    upload: bool,
    model_name: str,
):
    """
    The backwards model does not have evals available.
    """
    task_group_id = task["status"]["taskGroupId"]

    model = Model.create()
    model.config = get_config(task_group_id)
    model.task_group_id = task_group_id
    model.task_id = task["status"]["taskId"]
    model.task_name = task["task"]["metadata"]["name"]
    model.date = get_completed_time(task)

    prefix = f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/{model_name}/"
    model.artifact_folder = f"https://storage.googleapis.com/{BUCKET_NAME}/{prefix}"

    # List all of the artifacts.
    print(f"  [model] listing {model_name} files - {model.artifact_folder}")
    blobs: Optional[Iterable[storage.Blob]] = bucket.list_blobs(prefix=prefix)
    if blobs:
        model.artifact_urls = [
            f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}" for blob in blobs
        ]
    else:
        print(f"  [model] no {model_name} files found")

    if upload:
        model.sync_live_log(training_run, model_name)

    return model


def fetch_json(url: str):
    """
    A utility to fetch json in a single line.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_flores_eval_blob(
    training_run: TrainingRun,
    task_group_id: str,
    gcs_eval_name: str,
    tc_model_name: str,
) -> Optional[storage.Blob]:
    """
    Attempt to look up the flores eval blob from GCS.
    """
    # First try with just the source language.
    blob_url = (
        f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
        f"evaluation/{gcs_eval_name}/"
        f"{tc_model_name}-flores-devtest-{training_run.source_lang}_devtest.metrics.json"
    )
    blob = bucket.get_blob(blob_url)
    if not blob:
        # Also check with the langpair.
        blob_url = (
            f"models/{training_run.langpair}/{training_run.name}_{task_group_id}/"
            f"evaluation/{gcs_eval_name}/"
            f"{tc_model_name}-flores-devtest-{training_run.langpair}_devtest.metrics.json"
        )
        blob = bucket.get_blob(blob_url)

    return blob


def get_completed_time(task: dict) -> Optional[datetime]:
    """
    Get the time a task was completed, if it was.
    """
    for run in reversed(task["status"]["runs"]):
        if run["state"] == "completed":
            return str_to_datetime(run["resolved"])
    return None


def get_config(action_task_id: dict) -> Optional[dict]:
    """
    Get a config from the action's task id.
    """
    try:
        return get_artifact(action_task_id, "public/parameters.yml")["training_config"]
    except Exception:
        return None


def _match_by_label(task: dict, pattern: str) -> bool:
    """
    Implementor for the match-by_label function.
    """
    runs = task["status"]["runs"]
    if not runs:
        return False

    last_run = runs[-1]
    if last_run["state"] != "completed":
        return False

    return re.match(pattern, task["task"]["metadata"]["name"]) is not None


def match_by_label(pattern: str):
    """
    Match a task by a regex of its label.
    """
    return lambda task: _match_by_label(task, pattern)


def _find_task(
    tasks: list[dict], match: Callable[[dict], bool], min_or_max: Any
) -> Optional[dict]:
    """
    Implementation of the find_latest_task and find_earliest_task.
    """
    tasks = [task for task in tasks if match(task)]
    if not tasks:
        return None

    return min_or_max(
        tasks, key=lambda task: datetime.strptime(task["task"]["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
    )


def find_latest_task(tasks: list[dict], match: Callable[[dict], bool]) -> Optional[dict]:
    """
    The flattened list of tasks is unsorted, find the latest of the tasks.
    """
    return _find_task(tasks, match, max)


def find_earliest_task(tasks: list[dict], match: Callable[[dict], bool]):
    """
    The flattened list of tasks is unsorted, find the earliest of the tasks.
    """
    return _find_task(tasks, match, min)


def str_to_datetime(date_str: str) -> datetime:
    """
    Parse the taskcluster date string.
    """
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")


def task_name(task: dict) -> str:
    """Helper to get the task name"""
    return task["task"]["metadata"]["name"]


def task_url(task_id_or_task: Union[str, dict]) -> str:
    """Helper to get the task url"""
    task_id = (
        task_id_or_task
        if isinstance(task_id_or_task, str)
        else task_id_or_task["status"]["taskId"]
    )
    return f"https://firefox-ci-tc.services.mozilla.com/tasks/{task_id}"




def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Do not cache the TaskCluster calls"
    )
    parser.add_argument("--clear_cache", action="store_true", help="Clears the TaskCluster cache")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="When set to true, the artifacts are uploaded to GCS. Otherwise they are stored at to data/model-registry/",
    )
    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="By default only missing training runs are created. This recreates everything.",
    )
    args = parser.parse_args()

    cache = None
    if not args.no_cache:
        print(f"Using the cache {CACHE_FILE}")
        cache = shelve.open(str(CACHE_FILE))

    if args.clear_cache:
        print(f"Clearing the cache {CACHE_FILE}")
        if cache is None:
            cache = shelve.open(str(CACHE_FILE))
            cache.clear()
            cache.close()
            cache = None
        else:
            cache.clear()

    runs_by_training_pair = get_training_runs_by_langpair(cache)
    print_training_runs_tree(runs_by_training_pair)

    # Saves out the training runs depending on the --upload argument:
    #   - data/model-registry/training-runs/{name}-{langpair}.json
    #   - gs://{BUCKET}/models/{langpair}/{name}.json
    build_sqlite_for_training_runs(runs_by_training_pair, args.overwrite_runs, args.upload, cache)



    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
