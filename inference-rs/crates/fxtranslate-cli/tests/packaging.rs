//! Offline packaging tests: Remote Settings discovery + verified cache, driven by
//! checked-in fixtures through the mockable `Fetch` trait. No network, no engine —
//! translation correctness is the parent crate's job (issues/14-rust-only-package.md).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use fxtranslate_cli::cache::{ensure_model, sha256_hex, zstd_decode, Cache};
use fxtranslate_cli::remote::{
    fetch_records, language_matches, pairs, parse_records, records_url, Record,
};

mod common;
use common::MockFetch;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// A fresh, unique temp cache dir (no tempfile dep).
fn tmp_cache() -> Cache {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "fxtranslate-test-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    Cache::with_root(dir)
}

/// Tiny zstd fixture: decompressed bytes + their known SHA-256.
const TINY_HASH: &str = "5a9aaf6b319b6cdb5f3ef4ff520599018f2df654ddc4d9bb73dcb687092c77b8";
const TINY_PLAIN: &[u8] = b"fxtranslate tiny model fixture\n";

/// A synthetic record whose attachment (in tests) is the tiny zstd fixture, so its
/// `decompressed_hash` is one we control end to end.
fn tiny_record(name: &str, file_type: &str, src: &str, trg: &str) -> Record {
    Record {
        name: name.into(),
        file_type: file_type.into(),
        src: src.into(),
        trg: trg.into(),
        version: "3.0".into(),
        architecture: Some("base".into()),
        decompressed_hash: Some(TINY_HASH.into()),
        location: format!("cdn/{name}.zst"),
    }
}

/// Remote Settings record parsing, selection, and filtering.
mod discovery {
    use super::*;

    #[test]
    fn parses_fixture_records() {
        let recs =
            parse_records(std::str::from_utf8(&fixture("rs-models-v2.json")).unwrap()).unwrap();
        assert_eq!(recs.len(), 7, "fixture has 7 records");

        let ps = pairs(&recs);
        assert!(ps.contains(&("en".into(), "es".into())), "en-es present");
        assert!(ps.contains(&("en".into(), "ja".into())), "en-ja present");

        // Shared-vocab pair (en-es): model + vocab + lex, no split vocabs.
        assert!(fxtranslate_cli::remote::pick(&recs, "model", "en", "es").is_some());
        assert!(fxtranslate_cli::remote::pick(&recs, "vocab", "en", "es").is_some());
        assert!(fxtranslate_cli::remote::pick(&recs, "srcvocab", "en", "es").is_none());

        // Split-vocab pair (en-ja): srcvocab + trgvocab, no shared vocab.
        assert!(fxtranslate_cli::remote::pick(&recs, "vocab", "en", "ja").is_none());
        assert!(fxtranslate_cli::remote::pick(&recs, "srcvocab", "en", "ja").is_some());
        assert!(fxtranslate_cli::remote::pick(&recs, "trgvocab", "en", "ja").is_some());
    }

    #[test]
    fn fetch_records_goes_through_http() {
        let mock = MockFetch::new().route(&records_url(), fixture("rs-models-v2.json"));
        let recs = fetch_records(&mock).unwrap();
        assert_eq!(recs.len(), 7);
        assert_eq!(mock.hit_count(), 1);
    }

    #[test]
    fn pick_selects_latest_supported_minor_ignoring_higher_major() {
        let mk = |v: &str| Record {
            version: v.into(),
            ..tiny_record("model.enes", "model", "en", "es")
        };
        // Within the supported major, the latest minor wins numerically (3.10 >
        // 3.2, not lexically); higher majors (4.x / 100.x) are gated out — this
        // build only speaks the format it was validated against.
        let recs = vec![mk("3.2"), mk("3.10"), mk("4.0"), mk("100.0")];
        assert_eq!(
            fxtranslate_cli::remote::pick(&recs, "model", "en", "es")
                .unwrap()
                .version,
            "3.10"
        );
    }

    #[test]
    fn pick_tolerates_prerelease_version_suffix() {
        let mk = |v: &str| Record {
            version: v.into(),
            ..tiny_record("model.enes", "model", "en", "es")
        };
        // The live collection ships versions with a pre-release-style suffix
        // (`3.0a1`) alongside `3.0`/`3.1`. `version_key` tolerates the trailing
        // `a1` — its major is still 3 — so the gate accepts it on its own.
        assert_eq!(
            fxtranslate_cli::remote::pick(&[mk("3.0a1")], "model", "en", "es")
                .unwrap()
                .version,
            "3.0a1"
        );
        // Alongside real releases, the latest released minor still wins over the
        // suffixed one (we don't attempt to order the pre-release itself).
        let recs = vec![mk("3.0a1"), mk("3.0"), mk("3.1")];
        assert_eq!(
            fxtranslate_cli::remote::pick(&recs, "model", "en", "es")
                .unwrap()
                .version,
            "3.1"
        );
        // A suffixed version of an *unsupported* major is still gated out.
        assert!(fxtranslate_cli::remote::pick(&[mk("4.0a1")], "model", "en", "es").is_none());
    }

    #[test]
    fn pick_rejects_unsupported_major_only_pair() {
        let mk = |v: &str| Record {
            version: v.into(),
            ..tiny_record("model.enes", "model", "en", "es")
        };
        // A pair whose only models are a future major has nothing this build can
        // load — `pick` returns None (translate then reports "no model").
        let recs = vec![mk("100.0"), mk("100.5")];
        assert!(fxtranslate_cli::remote::pick(&recs, "model", "en", "es").is_none());
    }

    #[test]
    fn list_query_matches_both_directions() {
        // A language filter must surface both directions (every model pivots English).
        assert!(language_matches("es", "en", "es"), "es → en matches `es`");
        assert!(
            language_matches("en", "es", "es"),
            "en → es also matches `es`"
        );
        // Prefix catches script variants on either side.
        assert!(language_matches("zh-Hans", "en", "zh"));
        assert!(language_matches("en", "zh-Hant", "zh"));
        // A src-trg query prefix-matches each half against its side (one direction).
        assert!(language_matches("en", "es", "en-es"));
        assert!(!language_matches("es", "en", "en-es"));
        // ...and the src half is a prefix, so `zh-en` catches both Chinese scripts
        // (the whole reason for splitting the query rather than the pair string).
        assert!(language_matches("zh-Hans", "en", "zh-en"));
        assert!(language_matches("zh-Hant", "en", "zh-en"));
        assert!(!language_matches("en", "zh-Hans", "zh-en"));
        // Unrelated language doesn't match.
        assert!(!language_matches("en", "es", "fr"));
    }
}

/// zstd decode + SHA-256 verification of attachment bytes.
mod decompression {
    use super::*;

    #[test]
    fn decompress_and_hash_roundtrip() {
        let plain = zstd_decode(&fixture("tiny.bin.zst")).unwrap();
        assert_eq!(plain, TINY_PLAIN);
        assert_eq!(sha256_hex(&plain), TINY_HASH);
    }
}

/// Verified cache: download-then-hit, corrupt re-fetch, hash-mismatch rejection.
mod cache_behavior {
    use super::*;

    #[test]
    fn cache_downloads_then_hits() {
        let cache = tmp_cache();
        let rec = tiny_record("model.enes.bin", "model", "en", "es");
        let mock = MockFetch::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

        // First call downloads (and decompresses + verifies).
        let path = cache.ensure(&mock, &rec).unwrap();
        assert!(path.is_file());
        assert_eq!(std::fs::read(&path).unwrap(), TINY_PLAIN);
        assert_eq!(mock.hit_count(), 1, "first ensure downloads");

        // Second call is a cache hit — no further network.
        let path2 = cache.ensure(&mock, &rec).unwrap();
        assert_eq!(path, path2);
        assert_eq!(mock.hit_count(), 1, "second ensure is a cache hit");
    }

    #[test]
    fn cache_refetches_corrupt_file() {
        let cache = tmp_cache();
        let rec = tiny_record("model.enes.bin", "model", "en", "es");
        let mock = MockFetch::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

        let path = cache.ensure(&mock, &rec).unwrap();
        assert_eq!(mock.hit_count(), 1);

        // Corrupt the cached file (simulate a partial/damaged download).
        std::fs::write(&path, b"corrupted!!!").unwrap();

        // ensure() must detect the hash mismatch and re-fetch.
        let path2 = cache.ensure(&mock, &rec).unwrap();
        assert_eq!(path, path2);
        assert_eq!(
            std::fs::read(&path2).unwrap(),
            TINY_PLAIN,
            "restored good bytes"
        );
        assert_eq!(mock.hit_count(), 2, "corrupt file triggered a re-fetch");
    }

    #[test]
    fn cache_rejects_download_hash_mismatch() {
        let cache = tmp_cache();
        // Record claims a hash that the tiny fixture will not match.
        let mut rec = tiny_record("model.enes.bin", "model", "en", "es");
        rec.decompressed_hash = Some("0".repeat(64));
        let mock = MockFetch::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

        let err = cache.ensure(&mock, &rec).unwrap_err();
        assert!(err.contains("hash mismatch"), "got: {err}");
        assert!(
            !cache.pair_dir("en", "es").join(&rec.name).is_file(),
            "no bad file left"
        );
    }
}

/// Full model resolution: shared- vs split-vocab (CJK) wiring.
mod ensure_model_wiring {
    use super::*;

    #[test]
    fn ensure_model_shared_vocab() {
        let cache = tmp_cache();
        let recs = vec![
            tiny_record("model.enes.bin", "model", "en", "es"),
            tiny_record("vocab.enes.spm", "vocab", "en", "es"),
            tiny_record("lex.enes.bin", "lex", "en", "es"),
        ];
        let mut mock = MockFetch::new();
        for r in &recs {
            mock = mock.route(&r.cdn_url(), fixture("tiny.bin.zst"));
        }
        let files = ensure_model(&mock, &cache, &recs, "en", "es").unwrap();
        assert_eq!(files.src_vocab, files.trg_vocab, "shared vocab reused");
        assert!(files.lex.is_some(), "lex present");
        assert!(files.model.is_file());
    }

    #[test]
    fn ensure_model_split_vocab() {
        let cache = tmp_cache();
        let recs = vec![
            tiny_record("model.enja.bin", "model", "en", "ja"),
            tiny_record("srcvocab.enja.spm", "srcvocab", "en", "ja"),
            tiny_record("trgvocab.enja.spm", "trgvocab", "en", "ja"),
        ];
        let mut mock = MockFetch::new();
        for r in &recs {
            mock = mock.route(&r.cdn_url(), fixture("tiny.bin.zst"));
        }
        let files = ensure_model(&mock, &cache, &recs, "en", "ja").unwrap();
        assert_ne!(files.src_vocab, files.trg_vocab, "split vocab differs");
        assert!(files.lex.is_none(), "no shortlist for CJK split pair");
    }
}
