"""
Translations DB schema and SQL operations
"""

import sqlite3
from pathlib import Path
from typing import Optional, List

from datetime import datetime
from db.models import (
    Task,
    TrainingRun,
    RunComparison,
    Evaluation,
    Model,
    Corpus,
    WordAlignedCorpus,
)


class DatabaseSchema:
    @staticmethod
    def get_schema_sql() -> str:
        return """
        CREATE TABLE training_runs (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          langpair TEXT NOT NULL,
          source_lang TEXT NOT NULL,
          target_lang TEXT NOT NULL,
          date_started TEXT,
          experiment_config TEXT,
          UNIQUE (langpair, name)
        );

        CREATE TABLE run_comparisons (
          run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
          metric TEXT NOT NULL,
          provider TEXT NOT NULL,
          score REAL NOT NULL
        );

        CREATE TABLE corpora (
          id INTEGER PRIMARY KEY,
          run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
          type TEXT NOT NULL,
          aligned INTEGER NOT NULL,
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
          kind TEXT NOT NULL,
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

        CREATE TABLE tasks (
          id INTEGER PRIMARY KEY,
          task_id TEXT NOT NULL UNIQUE,
          task_group_id TEXT NOT NULL,
          langpair TEXT NOT NULL,
          training_run_name TEXT NOT NULL,
          created_date TEXT NOT NULL,
          state TEXT,
          task_name TEXT,
          resolved_date TEXT
        );
        CREATE INDEX idx_tasks_group_id ON tasks(task_group_id);
        CREATE INDEX idx_tasks_langpair_run ON tasks(langpair, training_run_name);
        CREATE INDEX idx_tasks_created ON tasks(created_date);
        CREATE INDEX idx_tasks_state ON tasks(state);
        """


class DatabaseManager:
    def __init__(self, sqlite_path: Path, rebuild: bool = False):
        self.sqlite_path = sqlite_path
        self.conn = self._init_connection(rebuild)

    def _init_connection(self, rebuild: bool) -> sqlite3.Connection:
        if rebuild and self.sqlite_path.exists():
            self.sqlite_path.unlink()

        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Only create schema if database is new or being rebuilt
        if rebuild or not self._has_schema(conn):
            conn.executescript(DatabaseSchema.get_schema_sql())

        return conn

    def _has_schema(self, conn: sqlite3.Connection) -> bool:
        """Check if the database already has our schema"""
        try:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'"
            ).fetchone()
            return result is not None
        except sqlite3.Error:
            return False

    def upsert_training_run(self, tr) -> int:
        import json

        experiment_config_json = None
        if tr.experiment_config:
            experiment_config_json = json.dumps(tr.experiment_config)

        self.conn.execute(
            """
          INSERT INTO training_runs(name, langpair, source_lang, target_lang, date_started, experiment_config)
          VALUES(?,?,?,?,?,?)
          ON CONFLICT(langpair, name) DO UPDATE SET
            source_lang=excluded.source_lang,
            target_lang=excluded.target_lang,
            date_started=excluded.date_started,
            experiment_config=excluded.experiment_config
        """,
            (
                tr.name,
                tr.langpair,
                tr.source_lang,
                tr.target_lang,
                tr.date_started.isoformat() if tr.date_started else None,
                experiment_config_json,
            ),
        )

        row = self.conn.execute(
            "SELECT id FROM training_runs WHERE langpair=? AND name=?", (tr.langpair, tr.name)
        ).fetchone()
        return int(row[0])

    def write_run_comparisons(self, run_id: int, tr):
        self.conn.execute("DELETE FROM run_comparisons WHERE run_id=?", (run_id,))
        for provider, score in (tr.bleu_flores_comparison or {}).items():
            self.conn.execute(
                "INSERT INTO run_comparisons(run_id, metric, provider, score) VALUES(?,?,?,?)",
                (run_id, "bleu", provider, float(score)),
            )
        for provider, score in (tr.comet_flores_comparison or {}).items():
            self.conn.execute(
                "INSERT INTO run_comparisons(run_id, metric, provider, score) VALUES(?,?,?,?)",
                (run_id, "comet", provider, float(score)),
            )

    def write_corpus(self, run_id: int, typ: str, aligned: int, c):
        if not c:
            return

        self.conn.execute(
            """
          INSERT INTO corpora(run_id, type, aligned, source_url, target_url, alignments_url,
                              source_bytes, target_bytes, alignments_bytes)
          VALUES(?,?,?,?,?,?,?,?,?)
        """,
            (
                run_id,
                typ,
                aligned,
                getattr(c, "source_url", None),
                getattr(c, "target_url", None),
                getattr(c, "alignments_url", None),
                getattr(c, "source_bytes", None),
                getattr(c, "target_bytes", None),
                getattr(c, "alignments_bytes", None),
            ),
        )

    def write_model(self, run_id: int, kind: str, m) -> Optional[int]:
        if not m:
            return None

        self.conn.execute(
            """
          INSERT INTO models(run_id, kind, date, task_group_id, task_id, task_name, artifact_folder)
          VALUES(?,?,?,?,?,?,?)
          ON CONFLICT(run_id, kind) DO UPDATE SET
            date=excluded.date,
            task_group_id=excluded.task_group_id,
            task_id=excluded.task_id,
            task_name=excluded.task_name,
            artifact_folder=excluded.artifact_folder
        """,
            (
                run_id,
                kind,
                m.date.isoformat() if m.date else None,
                m.task_group_id,
                m.task_id,
                m.task_name,
                m.artifact_folder,
            ),
        )

        row = self.conn.execute(
            "SELECT id FROM models WHERE run_id=? AND kind=?", (run_id, kind)
        ).fetchone()
        model_id = int(row[0])

        self.conn.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        if m.flores:
            self.conn.execute(
                "INSERT INTO evaluations(model_id, chrf, bleu, comet) VALUES(?,?,?,?)",
                (model_id, m.flores.chrf, m.flores.bleu, m.flores.comet),
            )

        self.conn.execute("DELETE FROM artifacts WHERE model_id=?", (model_id,))
        for url in m.artifact_urls or []:
            self.conn.execute("INSERT INTO artifacts(model_id, url) VALUES(?,?)", (model_id, url))

        return model_id

    def write_tasks(self, tasks: list, langpair: str, training_run_name: str, task_group_id: str):
        self.conn.execute("DELETE FROM tasks WHERE task_group_id=?", (task_group_id,))

        for task in tasks:
            task_id = task["status"]["taskId"]
            created_date = task["task"]["created"]
            task_name = task["task"]["metadata"].get("name", "")

            # Extract state and resolved date from runs
            state = None
            resolved_date = None
            runs = task["status"].get("runs", [])
            if runs:
                last_run = runs[-1]
                state = last_run.get("state")
                resolved_date = last_run.get("resolved")

            self.conn.execute(
                """
                INSERT INTO tasks(task_id, task_group_id, langpair, training_run_name,
                                created_date, state, task_name, resolved_date)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    state=excluded.state,
                    task_name=excluded.task_name,
                    resolved_date=excluded.resolved_date
            """,
                (
                    task_id,
                    task_group_id,
                    langpair,
                    training_run_name,
                    created_date,
                    state,
                    task_name,
                    resolved_date,
                ),
            )

    def get_tasks_for_training_run(self, langpair: str, training_run_name: str) -> List[Task]:
        cursor = self.conn.execute(
            """
            SELECT task_id, task_group_id, created_date, state, task_name, resolved_date
            FROM tasks
            WHERE langpair=? AND training_run_name=?
            ORDER BY created_date
        """,
            (langpair, training_run_name),
        )

        return [Task(*row) for row in cursor.fetchall()]

    def get_tasks_for_task_group(self, task_group_id: str) -> List[Task]:
        cursor = self.conn.execute(
            """
            SELECT task_id, task_group_id, created_date, state, task_name, resolved_date
            FROM tasks
            WHERE task_group_id=?
            ORDER BY created_date
        """,
            (task_group_id,),
        )

        return [Task(*row) for row in cursor.fetchall()]

    def get_training_runs(self) -> List[TrainingRun]:
        """Get all training runs from database"""
        import json

        cursor = self.conn.execute(
            """
            SELECT name, langpair, source_lang, target_lang, date_started, experiment_config
            FROM training_runs
            ORDER BY langpair, name
        """
        )

        runs = []
        for (
            name,
            langpair,
            source_lang,
            target_lang,
            date_started,
            experiment_config,
        ) in cursor.fetchall():
            experiment_config_dict = None
            if experiment_config:
                experiment_config_dict = json.loads(experiment_config)

            run = TrainingRun(
                name=name,
                langpair=langpair,
                source_lang=source_lang,
                target_lang=target_lang,
                task_group_ids=[],  # Would need separate query to populate
                date_started=datetime.fromisoformat(date_started) if date_started else None,
                experiment_config=experiment_config_dict,
            )
            runs.append(run)
        return runs

    def get_run_comparisons(self, run_id: int) -> List[RunComparison]:
        """Get comparison scores for a training run"""
        cursor = self.conn.execute(
            """
            SELECT metric, provider, score
            FROM run_comparisons
            WHERE run_id=?
        """,
            (run_id,),
        )

        return [RunComparison(*row) for row in cursor.fetchall()]

    def get_corpora_for_run(self, run_id: int) -> List[Corpus]:
        """Get corpora for a training run"""
        cursor = self.conn.execute(
            """
            SELECT id, run_id, type, aligned, source_url, target_url, alignments_url,
                   source_bytes, target_bytes, alignments_bytes
            FROM corpora
            WHERE run_id=?
        """,
            (run_id,),
        )

        corpora = []
        for row in cursor.fetchall():
            (
                corpus_id,
                run_id,
                typ,
                aligned,
                source_url,
                target_url,
                alignments_url,
                source_bytes,
                target_bytes,
                alignments_bytes,
            ) = row
            if alignments_url:
                corpus = WordAlignedCorpus(
                    source_url=source_url or "",
                    target_url=target_url or "",
                    alignments_url=alignments_url,
                    source_bytes=source_bytes or 0,
                    target_bytes=target_bytes or 0,
                    alignments_bytes=alignments_bytes or 0,
                    id=corpus_id,
                    run_id=run_id,
                    type=typ,
                    aligned=bool(aligned),
                )
            else:
                corpus = Corpus(
                    source_url=source_url or "",
                    source_bytes=source_bytes or 0,
                    target_url=target_url or "",
                    target_bytes=target_bytes or 0,
                    id=corpus_id,
                    run_id=run_id,
                    type=typ,
                    aligned=bool(aligned),
                )
            corpora.append(corpus)
        return corpora

    def get_models_for_run(self, run_id: int) -> List[Model]:
        """Get models for a training run with evaluations and artifacts"""
        cursor = self.conn.execute(
            """
            SELECT m.id, m.run_id, m.kind, m.date, m.task_group_id, m.task_id,
                   m.task_name, m.artifact_folder, e.chrf, e.bleu, e.comet
            FROM models m
            LEFT JOIN evaluations e ON m.id = e.model_id
            WHERE m.run_id=?
        """,
            (run_id,),
        )

        models = []
        for row in cursor.fetchall():
            (
                mid,
                run_id,
                kind,
                date,
                task_group_id,
                task_id,
                task_name,
                artifact_folder,
                chrf,
                bleu,
                comet,
            ) = row

            evaluation = None
            if chrf is not None or bleu is not None or comet is not None:
                evaluation = Evaluation(chrf=chrf, bleu=bleu, comet=comet)

            # Get artifact URLs
            artifact_cursor = self.conn.execute(
                "SELECT url FROM artifacts WHERE model_id=?", (mid,)
            )
            artifact_urls = [row[0] for row in artifact_cursor.fetchall()]

            # Parse date if it exists
            parsed_date = None
            if date:
                try:
                    parsed_date = datetime.fromisoformat(date)
                except ValueError:
                    pass

            model = Model(
                id=mid,
                run_id=run_id,
                kind=kind,
                date=parsed_date,
                task_group_id=task_group_id,
                task_id=task_id,
                task_name=task_name,
                flores=evaluation,
                artifact_folder=artifact_folder,
                artifact_urls=artifact_urls,
            )
            models.append(model)

        return models


# Convenience functions for backward compatibility
def init_db(sqlite_path: Path) -> sqlite3.Connection:
    db = DatabaseManager(sqlite_path)
    return db.conn


def get_schema_sql() -> str:
    return DatabaseSchema.get_schema_sql()
