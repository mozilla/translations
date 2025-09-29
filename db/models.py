"""
Models for Translations DB entities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Evaluation:
    chrf: Optional[float] = None
    bleu: Optional[float] = None
    comet: Optional[float] = None


@dataclass
class Corpus:
    source_url: str
    source_bytes: int
    target_url: str
    target_bytes: int
    id: Optional[int] = None
    run_id: Optional[int] = None
    type: Optional[str] = None
    aligned: Optional[bool] = None


@dataclass
class WordAlignedCorpus:
    source_url: str
    target_url: str
    alignments_url: str
    source_bytes: int
    target_bytes: int
    alignments_bytes: int
    id: Optional[int] = None
    run_id: Optional[int] = None
    type: Optional[str] = None
    aligned: Optional[bool] = None


@dataclass
class Model:
    date: Optional[datetime] = None
    config: Optional[dict] = None
    task_group_id: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    flores: Optional[Evaluation] = None
    artifact_folder: Optional[str] = None
    artifact_urls: list[str] = None
    id: Optional[int] = None
    run_id: Optional[int] = None
    kind: Optional[str] = None

    def __post_init__(self):
        if self.artifact_urls is None:
            self.artifact_urls = []


@dataclass
class TrainingRun:
    name: str
    langpair: str
    source_lang: str
    target_lang: str
    task_group_ids: list[str]
    date_started: Optional[datetime] = None
    experiment_config: Optional[dict] = None
    comet_flores_comparison: dict[str, float] = None
    bleu_flores_comparison: dict[str, float] = None

    parallel_corpus_aligned: Optional[WordAlignedCorpus] = None
    backtranslations_corpus_aligned: Optional[WordAlignedCorpus] = None
    distillation_corpus_aligned: Optional[WordAlignedCorpus] = None

    parallel_corpus: Optional[Corpus] = None
    backtranslations_corpus: Optional[Corpus] = None
    distillation_corpus: Optional[Corpus] = None

    backwards: Optional[Model] = None
    teacher_1: Optional[Model] = None
    teacher_2: Optional[Model] = None
    teacher_ensemble_flores: Optional[Evaluation] = None
    student: Optional[Model] = None
    student_finetuned: Optional[Model] = None
    student_quantized: Optional[Model] = None
    student_exported: Optional[Model] = None

    def __post_init__(self):
        if self.comet_flores_comparison is None:
            self.comet_flores_comparison = {}
        if self.bleu_flores_comparison is None:
            self.bleu_flores_comparison = {}

    @classmethod
    def create(cls, name: str, task_group_ids: list[str], langpair: str):
        source_lang, target_lang = langpair.split("-")
        return cls(
            name=name,
            langpair=langpair,
            source_lang=source_lang,
            target_lang=target_lang,
            task_group_ids=task_group_ids,
        )


@dataclass
class Task:
    task_id: str
    task_group_id: str
    created_date: str
    state: Optional[str] = None
    task_name: Optional[str] = None
    resolved_date: Optional[str] = None


@dataclass
class RunComparison:
    metric: str
    provider: str
    score: float
