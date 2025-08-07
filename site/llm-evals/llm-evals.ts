export type ScoreNumbers = 1 | 2 | 3 | 4 | 5
export type Score = [ScoreNumbers, string]
export type ScoreType =
  | "adequacy"
  | "fluency"
  | "terminology"
  | "hallucination"
  | "punctuation"

export interface Evaluation {
  translation: Translation,
  scores: Scores,
}

export interface Translation {
  src: string,
  trg: string,
  ref: string,
}

export type Scores = Record<ScoreType, Score>;

export interface Analysis {
  mean: number,
  median: number,
  histogram: Record<ScoreNumbers, number>
}

/**
 * Contains a human readable description of the term.
 */
interface Term {
  description: string,
  scales: Record<ScoreNumbers, string>
}

export type Terminology = Record<ScoreType, Term>;

export type Summary = Record<ScoreType, string>
