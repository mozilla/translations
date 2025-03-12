"""
This file contains the canonical format for the training config. It's defined as a
type-safe dataclass, and then shared in other parts of the codebase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import yaml
from translations_taskgraph.util.dataclass_helpers import KebabDataclass, StricterDataclass


@dataclass(kw_only=True)
class MarianArgs(KebabDataclass):
    training_backward: Optional[dict[str, str]] = None
    training_teacher: Optional[dict[str, str]] = None
    training_student: Optional[dict[str, str]] = None
    training_student_finetuned: Optional[dict[str, str]] = None
    decoding_backward: Optional[dict[str, str]] = None
    decoding_teacher: Optional[dict[str, str]] = None


@dataclass(kw_only=True)
class MonoMaxSentences(KebabDataclass):
    """
    Limits for monolingual datasets.
    """

    # The limit sentence size for the total dataset.
    total: int
    # Limits per downloaded dataset.
    per_dataset: int


@dataclass(kw_only=True)
class CleanerThresholds(KebabDataclass):
    default_threshold: float
    dataset_thresholds: Optional[dict[str, float]] = None


@dataclass(kw_only=True)
class MonocleanerConfig(KebabDataclass):
    mono_src: CleanerThresholds
    mono_trg: CleanerThresholds


@dataclass(kw_only=True)
class HpltMinDocScore(KebabDataclass):
    mono_src: float
    mono_trg: float


@dataclass(kw_only=True)
class PretrainedModel(KebabDataclass):
    """
    Pre-trained models use URLs as they are flexible to continue training from either
    long-term bucket storage, or from tasks in Taskcluster.
    """

    urls: list[str]
    mode: Literal["continue"] | Literal["init"] | Literal["use"]
    type: Literal["default"] | Literal["opusmt"]


@dataclass(kw_only=True)
class PretrainedModels(KebabDataclass):
    train_backwards: Optional[PretrainedModel] = None
    train_teacher: Optional[PretrainedModel] = None


@dataclass(kw_only=True)
class Experiment(KebabDataclass):
    # A name for the experiment.
    name: str
    # The source locale to train.
    src: str
    # The target locale to train.
    trg: str
    # Number of teachers to train in an ensemble.
    teacher_ensemble: int
    # Teacher training mode.
    teacher_mode: Literal["one-stage"] | Literal["two-stage"]
    # Translate with either Marian or CTranslate2.
    teacher_decoder: Literal["marian"] | Literal["ctranslate2"]
    # The student model architecture as defined in:
    #   pipeline/train/configs/model/student.{model}.yml
    student_model: Literal["tiny"] | Literal["base"]

    # Limits for the "target" monolingual data, e.g. data used for back translations.
    mono_max_sentences_trg: MonoMaxSentences
    # Limits for the "source" monolingual data, e.g. data used for student distillation.
    mono_max_sentences_src: MonoMaxSentences

    # How large of a sample to use for the Sentence Piece sample size.
    spm_sample_size: int
    #  The metric to use for the best model.
    best_model: (
        Literal["cross-entropy"]
        | Literal["ce-mean-words"]
        | Literal["perplexity"]
        | Literal["valid-script"]
        | Literal["translation"]
        | Literal["bleu"]
        | Literal["bleu-detok"]
        | Literal["bleu-segmented"]  # (deprecated, same as bleu)
        | Literal["chrf"]
    )
    # Use OpusCleaner to clean the corpus.
    # https://github.com/hplt-project/OpusCleaner
    use_opuscleaner: Literal["true"] | Literal["false"]
    # Indicates whether to use dataset specific configs.
    opuscleaner_mode: Literal["custom"] | Literal["defaults"]
    # Thresholds for running bicleaner-ai.
    # https://github.com/bitextor/bicleaner-ai
    bicleaner: CleanerThresholds
    monocleaner: MonocleanerConfig
    hplt_min_doc_score: HpltMinDocScore
    # Instead of training models from scratch, use pre-trained models.
    pretrained_models: PretrainedModels
    corpus_max_sentences: Optional[int] = None
    spm_vocab_size: Optional[int] = None


@dataclass(kw_only=True)
class Taskcluster(KebabDataclass):
    split_chunks: int
    worker_classes: dict[str, Literal["gcp-standard"] | Literal["gcp-spot"]]


@dataclass(kw_only=True)
class Datasets(KebabDataclass):
    """
    Represents the datasets used for training.
    """

    # Datasets to merge for validation while training.
    devtest: list[str]
    # Datasets for evaluation. Each will generate an evaluate-* task for each model type.
    test: list[str]
    # Parallel training corpora.
    train: list[str]
    # Monolingual datasets that are translated by the teacher model to generate the
    # data to be used for student distillation.
    mono_src: list[str]  # mono src docs
    # Monolingual datasets that are translated by the back translations model to
    # synthesize data to increase the amount of data available for teacher training.
    mono_trg: list[str]


@dataclass(kw_only=True)
class TrainingConfig(KebabDataclass):
    datasets: dict[str, list[str]]

    marian_args: MarianArgs
    experiment: Experiment

    # Taskcluster-specific pipeline configuration, eg: chunking
    taskcluster: Taskcluster

    # Enable publication to Weights and Biases
    wandb_publication: bool

    # An array of taskIds of decision or action tasks from the previous group(s) to use
    # to populate our `previous_group_kinds`. Tasks specified here will be used as long
    # as their label matches a needed task, and that task is upstream of `start-stage`.
    # (That is to say: even if a task from one of these groups has a cache digest that
    # doesn't match what the downstream task wants, it will still be used. This can be
    # used for quick iteration of functionality where the quality of the outputs is not
    # important.
    previous_group_ids: Optional[list[str]] = None

    # The stage of the pipeline to begin at, provided replacements can be found for tasks
    # upstream of this stage. Usually used in conjunction with `previous-group-ids`
    # which allows for specifying task group ids to fetch existing tasks from.
    start_stage: Optional[str] = None

    # The stage of the pipeline to run until (any stages this choice depends on will
    # be automatically included).
    target_stage: str

    # A mapping of task labels to task IDs that will be re-used in this training run.
    # For example:
    #
    # existing-tasks: {
    #         "build-docker-image-base": "BAvLUilqQ3SYqy6Ck55CUQ",
    #         "build-docker-image-test": "f0gbptvMTDaKODjqL9hlOw",
    #         "build-docker-image-toolchain-build": "LlZa8-L9TRemgyzQcAxuHw",
    #         "build-docker-image-train": "fBMJa9R5SKaXd2wgWeD5yQ",
    #         "fetch-browsermt-marian": "BRviRlEMTie8AUFf5prHvg",
    #     }
    existing_tasks: dict[str, str]

    # ------------------------------------------------------------------------------------
    # them to be used by the type system.

    @staticmethod
    def from_dict_validated(config_dict: dict):
        """
        Creates a TrainingConfig and validates it from the graph config.
        """
        training_config = TrainingConfig.from_dict(config_dict)

        with open(Path(__file__).parent / "../config.yml", "r") as file:
            valid_stages = yaml.safe_load(file)["valid-stages"]

        if training_config.start_stage and training_config.start_stage not in valid_stages:
            raise ValueError(
                f'start stage "{training_config.start_stage}" is not a valid stage in taskcluster/config.yml'
            )

        if training_config.target_stage and training_config.target_stage not in valid_stages:
            raise ValueError(
                f'target stage "{training_config.target_stage}" is not a valid stage in taskcluster/config.yml'
            )

        return training_config


@dataclass(kw_only=True)
class Parameters(StricterDataclass):
    base_repository: str
    base_ref: str
    base_rev: str
    build_date: int
    build_number: int
    do_not_optimize: list[str]
    enable_always_target: bool
    existing_tasks: dict[str, str]
    files_changed: Optional[list[str]] = None
    filters: list[str]
    head_ref: str
    head_repository: str
    head_rev: str
    head_tag: str
    level: str
    moz_build_date: str
    next_version: Optional[str]
    optimize_strategies: Optional[str]
    optimize_target_tasks: bool
    owner: str
    project: str
    pushdate: int
    pushlog_id: str
    repository_type: str
    target_tasks_method: str
    tasks_for: str
    version: Optional[str] = None
    training_config: TrainingConfig
