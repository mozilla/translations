export interface Corpus {
  source_url: string;
  source_bytes: number;
  target_url: string;
  target_bytes: number;
}

export interface ScoreComparison {
  nllb: number,
  bergamot: number,
  google: number,
  microsoft: number,
  opusmt: number,
  argos: number,
}

export interface DatasetConfig {
  devtest: string[];
  "mono-src": string[];
  "mono-trg": string[];
  test: string[];
  train: string[];
}

export type MarianArgs = Record<string, unknown>;

export interface ModelConfig {
  datasets: DatasetConfig;
  experiment: Record<string, unknown>;
  "marian-args": Record<string, unknown>;
  "target-stage": string;
  taskcluster: Record<string, unknown>;
  "wandb-publication": boolean;
}

export interface ModelRun {
  date: string;
  config: ModelConfig;
  task_group_id: string;
  task_id: string;
  task_name: string;
  flores?: {
    chrf: number;
    bleu: number;
    comet?: number;
  };
  artifact_folder: string;
  artifact_urls: string[];
}

export interface TrainingRun {
  name: string;
  langpair: string;
  source_lang: string;
  target_lang: string;
  task_group_ids: string[];
  date_started: string;
  comet_flores_comparison: Partial<ScoreComparison>;
  bleu_flores_comparison: Partial<ScoreComparison>;

  parallel_corpus_aligned?: Corpus;
  backtranslations_corpus_aligned?: Corpus;
  distillation_corpus_aligned?: Corpus;
  parallel_corpus?: Corpus;
  backtranslations_corpus?: Corpus;
  distillation_corpus?: Corpus;

  backwards?: ModelRun;
  teacher_1?: ModelRun;
  teacher_2?: ModelRun;
  student?: ModelRun;
  student_finetuned?: ModelRun;
  student_quantized?: ModelRun;
  student_exported?: ModelRun;

  teacher_ensemble_flores: null;
}

export type ModelName =
  | "backwards"
  | "teacher_1"
  | "teacher_2"
  | "student"
  | "student_finetuned"
  | "student_quantized"
  | "student_exported";

export interface ModelReference {
  name: string,
  langpair: string,
  modelName: ModelName
}
