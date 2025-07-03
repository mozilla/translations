type LangPairStr = string
type DatasetStr = string
type TranslatorStr = string;
type ScoreNum = number;

type EvalResults = Record<LangPairStr, Record<DatasetStr, Record<TranslatorStr, ScoreNum>>>;

/**
 * For Remote Settings, the JSON details about the attachment.
 */
export interface Attachment {
  // e.g. "2f7c0f7bbc...ca79f0850c4de",
 hash: string;
 // e.g. 5047568,
 size: string;
 // e.g. "lex.50.50.deen.s2t.bin",
 filename: string;
 // e.g. "main-workspace/translations-models/316ebb3a-0682-42cc-8e73-a3ba4bbb280f.bin",
 location: string;
 // e.g. "application/octet-stream"
 mimetype: string;
}

/**
 * The JSON that is synced from Remote Settings for the translation models.
 */
export interface ModelRecord {
 // e.g. "0d4db293-a17c-4085-9bd8-e2e146c85000"
 id: string;
 // The full model name, e.g. "lex.50.50.deen.s2t.bin"
 name: string;
 // The BCP 47 language tag, e.g. "de"
 fromLang: string;
 // The BCP 47 language tag, e.g. "en"
 toLang: string;
 // The semver number, used for handling future format changes. e.g. 1.0
 version: string;
 // e.g. "lex"
 fileType: string;
 // The file attachment for this record
 attachment: Attachment;
 // e.g. 1673023100578
 schema: number;
 // e.g. 1673455932527
 last_modified: string;
 // A JEXL expression to determine whether this record should be pulled from Remote Settings
 // See: https://remote-settings.readthedocs.io/en/latest/target-filters.html#filter-expressions
 filter_expression: string;
}

export interface ReleaseInfo {
  release: boolean,
  beta: boolean,
  nightly: boolean,
  android: boolean,
  label: string,
}

export interface ModelStatistics {
    decoder_bytes: number,
    decoder_parameters: number,
    embeddings_bytes: number,
    encoder_bytes: number,
    encoder_parameters: number,
    parameters: number,
}


export interface ModelMetadata {
    // For instance "base-memory" or "tiny"
    architecture: string,
    // The size of the uncompressed model in bytes.
    byteSize: number,
    // The flores scores, e.g. { "bleu": 39.6, "comet": 0.8649 }
    flores: Record<string, number>,
    // The sha256 hash of the uncompressed model.
    hash: string,
    // The Marian config for the model.
    modelConfig: Record<string, any>,
    // The number of parameters and bytes for the model's size.
    modelStatistics: ModelStatistics,
    sourceLanguage: string,
    targetLanguage: string,
    // The version in Remote Settings, used to select the latest model.
    version: string,
}
