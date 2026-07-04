//! Local model cache: resolve a cache dir, download a record's zstd attachment,
//! decompress it, verify its SHA-256 against the record's `decompressedHash`, and
//! store it atomically. A subsequent run with a matching hash is a cache hit and
//! skips the network.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::http::Http;
use crate::remote::Record;

/// Hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Decompress a zstd stream fully into memory (pure-Rust decoder).
pub fn zstd_decode(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut dec = ruzstd::StreamingDecoder::new(input).map_err(|e| format!("zstd init: {e}"))?;
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|e| format!("zstd decode: {e}"))?;
    Ok(out)
}

/// The verified model-file paths the engine loads. For shared-vocab pairs
/// `src_vocab == trg_vocab`; split-vocab (CJK) pairs differ. `lex` is present only
/// when the pair ships a shortlist.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub model: PathBuf,
    pub src_vocab: PathBuf,
    pub trg_vocab: PathBuf,
    pub lex: Option<PathBuf>,
}

/// A directory of cached, decompressed model files.
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// The default cache root: the platform-native cache directory (`dirs`) with
    /// an `fxtranslate/models` subtree — e.g. `~/Library/Caches/fxtranslate/models`
    /// on macOS, `$XDG_CACHE_HOME` (or `~/.cache`) `/fxtranslate/models` on Linux,
    /// `%LOCALAPPDATA%\fxtranslate\models` on Windows. Falls back to a local dir if
    /// the cache directory can't be determined. Per-pair files then live under
    /// `<root>/<src>-<trg>/` (see [`Cache::pair_dir`]).
    pub fn locate() -> Cache {
        let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".fxtranslate-cache"));
        Cache {
            root: base.join("fxtranslate").join("models"),
        }
    }

    /// A cache rooted at an explicit path (tests, or a `--cache-dir` override).
    pub fn with_root(root: impl Into<PathBuf>) -> Cache {
        Cache { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Per-pair directory: `<root>/<src>-<trg>`.
    pub fn pair_dir(&self, src: &str, trg: &str) -> PathBuf {
        self.root.join(format!("{src}-{trg}"))
    }

    /// Ensure `record`'s decompressed file is present and hash-verified, fetching
    /// (and decompressing) it via `http` only on a miss or a hash mismatch.
    /// Returns the on-disk path.
    pub fn ensure(&self, http: &dyn Http, record: &Record) -> Result<PathBuf, String> {
        let dir = self.pair_dir(&record.src, &record.trg);
        let dest = dir.join(&record.name);

        // Cache hit: file present and (if we know the expected hash) matching.
        if dest.is_file() {
            match &record.decompressed_hash {
                Some(expected) => {
                    let got = sha256_hex(&fs::read(&dest).map_err(|e| e.to_string())?);
                    if &got == expected {
                        return Ok(dest);
                    }
                    // Corrupt/partial/stale — fall through and re-fetch.
                    eprintln!(
                        "[cache] {} hash mismatch (have {}…, want {}…); re-fetching",
                        record.name,
                        &got[..8.min(got.len())],
                        &expected[..8.min(expected.len())]
                    );
                }
                None => return Ok(dest), // no hash to check against; trust presence
            }
        }

        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let compressed = http.get(&record.cdn_url())?;
        let bytes = zstd_decode(&compressed)?;

        if let Some(expected) = &record.decompressed_hash {
            let got = sha256_hex(&bytes);
            if &got != expected {
                return Err(format!(
                    "hash mismatch for {} after download: got {got}, expected {expected}",
                    record.name
                ));
            }
        }

        // Atomic write: temp then rename, so an interrupted run never leaves a
        // partial file that a later run would trust.
        let tmp = dir.join(format!(".{}.partial", record.name));
        fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
        Ok(dest)
    }
}

/// Resolve, download+cache, and verify all files needed to translate `src`→`trg`.
/// Picks the latest-version records; handles shared vs. split (CJK) vocab.
pub fn ensure_model(
    http: &dyn Http,
    cache: &Cache,
    records: &[Record],
    src: &str,
    trg: &str,
) -> Result<ModelFiles, String> {
    use crate::remote::pick;

    let model = pick(records, "model", src, trg)
        .ok_or_else(|| format!("no model for {src}-{trg} in Remote Settings"))?;
    let model_path = cache.ensure(http, model)?;

    // Shared vocab ships one `vocab`; split vocab (CJK) ships `srcvocab`/`trgvocab`.
    let (src_vocab, trg_vocab) = if let Some(v) = pick(records, "vocab", src, trg) {
        let p = cache.ensure(http, v)?;
        (p.clone(), p)
    } else {
        let sv = pick(records, "srcvocab", src, trg)
            .ok_or_else(|| format!("no vocab/srcvocab for {src}-{trg}"))?;
        let tv = pick(records, "trgvocab", src, trg)
            .ok_or_else(|| format!("no trgvocab for {src}-{trg}"))?;
        (cache.ensure(http, sv)?, cache.ensure(http, tv)?)
    };

    let lex = match pick(records, "lex", src, trg) {
        Some(l) => Some(cache.ensure(http, l)?),
        None => None,
    };

    Ok(ModelFiles {
        model: model_path,
        src_vocab,
        trg_vocab,
        lex,
    })
}
