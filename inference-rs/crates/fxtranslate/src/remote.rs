//! Remote Settings model discovery.
//!
//! Mirrors `inference-rs/scripts/download_model.py`: the production
//! `translations-models-v2` collection lists one record per file (model / vocab /
//! lex / srcvocab / trgvocab) tagged with `sourceLanguage` / `targetLanguage` /
//! `version`, each carrying a zstd attachment on the settings CDN and the
//! `decompressedHash` the cache verifies against.

use tinyjson::JsonValue;

use crate::fetch::Fetch;

pub const PROD_ENDPOINT: &str = "https://firefox.settings.services.mozilla.com";
/// Production collection. NOTE: the `-v2` collection, not the legacy
/// `translations-models` — matches `download_model.py`.
pub const COLLECTION: &str = "translations-models-v2";
pub const CDN_ROOT: &str = "https://firefox-settings-attachments.cdn.mozilla.net";

/// The records endpoint for the model collection.
pub fn records_url() -> String {
    format!("{PROD_ENDPOINT}/v1/buckets/main/collections/{COLLECTION}/records")
}

/// One Remote Settings model-file record (the fields the CLI needs).
#[derive(Clone, Debug)]
pub struct Record {
    pub name: String,
    /// `model`, `vocab`, `lex`, `srcvocab`, `trgvocab`.
    pub file_type: String,
    pub src: String,
    pub trg: String,
    pub version: String,
    pub architecture: Option<String>,
    /// SHA-256 of the *decompressed* file; the cache verifies against it.
    pub decompressed_hash: Option<String>,
    /// Attachment path under the CDN root.
    pub location: String,
}

impl Record {
    /// Full CDN URL of the (zstd-compressed) attachment.
    pub fn cdn_url(&self) -> String {
        format!("{CDN_ROOT}/{}", self.location)
    }
}

fn as_str(v: Option<&JsonValue>) -> Option<String> {
    match v {
        Some(JsonValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Parse a Remote Settings `records` response body into [`Record`]s. Records
/// missing required fields (name / fileType / languages / attachment location)
/// are skipped rather than failing the whole parse — the collection carries
/// unrelated shapes over time.
pub fn parse_records(body: &str) -> Result<Vec<Record>, String> {
    let root: JsonValue = body.parse().map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = match &root {
        JsonValue::Object(o) => o,
        _ => return Err("records response is not an object".into()),
    };
    let data = match obj.get("data") {
        Some(JsonValue::Array(a)) => a,
        _ => return Err("records response has no `data` array".into()),
    };

    let mut out = Vec::new();
    for item in data {
        let rec = match item {
            JsonValue::Object(o) => o,
            _ => continue,
        };
        let attachment = match rec.get("attachment") {
            Some(JsonValue::Object(a)) => a,
            _ => continue,
        };
        let (name, file_type, src, trg, location) = match (
            as_str(rec.get("name")),
            as_str(rec.get("fileType")),
            as_str(rec.get("sourceLanguage")),
            as_str(rec.get("targetLanguage")),
            as_str(attachment.get("location")),
        ) {
            (Some(n), Some(f), Some(s), Some(t), Some(l)) => (n, f, s, t, l),
            _ => continue,
        };
        out.push(Record {
            name,
            file_type,
            src,
            trg,
            version: as_str(rec.get("version")).unwrap_or_else(|| "0".into()),
            architecture: as_str(rec.get("architecture")),
            decompressed_hash: as_str(rec.get("decompressedHash")),
            location,
        });
    }
    Ok(out)
}

/// Fetch and parse the model records.
pub fn fetch_records(fetch: &dyn Fetch) -> Result<Vec<Record>, String> {
    let body = fetch.get(&records_url())?;
    let text = String::from_utf8(body).map_err(|e| format!("records not UTF-8: {e}"))?;
    parse_records(&text)
}

/// Dotted-version sort key (`"3.1"` → `[3, 1]`); non-numeric parts sort as 0.
fn version_key(v: &str) -> Vec<u64> {
    v.split('.').map(|p| p.parse().unwrap_or(0)).collect()
}

/// The major model version this build of Marian is validated against. The
/// Remote Settings `version` encodes backend compatibility: the live
/// `translations-models-v2` collection currently ships v3 models (`3.0`, `3.1`,
/// and a `3.0a1` alpha). A major bump (v4+) means the format/backend changed and
/// fxtranslate must be rebuilt, so records outside this major are ignored here
/// rather than mistranslated. Within the major, the latest minor still wins.
pub const SUPPORTED_MAJOR: u64 = 3;

/// Whether this build can use a record of `version`: its major (the first
/// component of `version_key`) must equal [`SUPPORTED_MAJOR`]. A missing/blank
/// version (parsed as `0`) is treated as unsupported; a pre-release suffix like
/// `3.0a1` parses to major `3` and is accepted.
fn version_supported(version: &str) -> bool {
    version_key(version).first().copied() == Some(SUPPORTED_MAJOR)
}

/// The latest supported-version record of `file_type` for the pair, if any.
/// Records outside [`SUPPORTED_MAJOR`] are ignored; among the rest the latest
/// minor wins.
pub fn pick<'a>(
    records: &'a [Record],
    file_type: &str,
    src: &str,
    trg: &str,
) -> Option<&'a Record> {
    records
        .iter()
        .filter(|r| r.file_type == file_type && r.src == src && r.trg == trg)
        .filter(|r| version_supported(&r.version))
        .max_by(|a, b| version_key(&a.version).cmp(&version_key(&b.version)))
}

/// Whether a `list` query selects the pair `src`→`trg`. A `src-trg` query
/// prefix-matches each half against its side (`en-es` → `en*`→`es*`, `zh-en` →
/// `zh*`→`en*`, catching both Chinese scripts); a bare query prefix-matches
/// either side, so `es` surfaces both `es → en` and `en → es`.
pub fn language_matches(src: &str, trg: &str, query: &str) -> bool {
    match query.split_once('-') {
        Some((q_src, q_trg)) => src.starts_with(q_src) && trg.starts_with(q_trg),
        None => src.starts_with(query) || trg.starts_with(query),
    }
}

/// Unique `(src, trg)` pairs that have a supported `model` record, sorted — the
/// set the `list` command enumerates. Pairs whose only model is outside
/// [`SUPPORTED_MAJOR`] are omitted, matching what `translate` can actually load.
pub fn pairs(records: &[Record]) -> Vec<(String, String)> {
    let mut ps: Vec<(String, String)> = records
        .iter()
        .filter(|r| r.file_type == "model" && version_supported(&r.version))
        .map(|r| (r.src.clone(), r.trg.clone()))
        .collect();
    ps.sort();
    ps.dedup();
    ps
}
