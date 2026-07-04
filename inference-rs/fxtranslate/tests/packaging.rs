//! Offline packaging tests: Remote Settings discovery + verified cache, driven by
//! checked-in fixtures through the mockable `Http` trait. No network, no engine —
//! translation correctness is the parent crate's job (issues/14-rust-only-package.md).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use fxtranslate::cache::{ensure_model, sha256_hex, zstd_decode, Cache};
use fxtranslate::http::MockHttp;
use fxtranslate::remote::{
    fetch_records, language_matches, pairs, parse_records, records_url, Record,
};

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
        version: "1.0".into(),
        architecture: Some("base".into()),
        decompressed_hash: Some(TINY_HASH.into()),
        location: format!("cdn/{name}.zst"),
    }
}

// ---- Remote Settings discovery ---------------------------------------------

#[test]
fn parses_fixture_records() {
    let recs = parse_records(std::str::from_utf8(&fixture("rs-models-v2.json")).unwrap()).unwrap();
    assert_eq!(recs.len(), 7, "fixture has 7 records");

    let ps = pairs(&recs);
    assert!(ps.contains(&("en".into(), "es".into())), "en-es present");
    assert!(ps.contains(&("en".into(), "ja".into())), "en-ja present");

    // Shared-vocab pair (en-es): model + vocab + lex, no split vocabs.
    assert!(fxtranslate::remote::pick(&recs, "model", "en", "es").is_some());
    assert!(fxtranslate::remote::pick(&recs, "vocab", "en", "es").is_some());
    assert!(fxtranslate::remote::pick(&recs, "srcvocab", "en", "es").is_none());

    // Split-vocab pair (en-ja): srcvocab + trgvocab, no shared vocab.
    assert!(fxtranslate::remote::pick(&recs, "vocab", "en", "ja").is_none());
    assert!(fxtranslate::remote::pick(&recs, "srcvocab", "en", "ja").is_some());
    assert!(fxtranslate::remote::pick(&recs, "trgvocab", "en", "ja").is_some());
}

#[test]
fn fetch_records_goes_through_http() {
    let mock = MockHttp::new().route(&records_url(), fixture("rs-models-v2.json"));
    let recs = fetch_records(&mock).unwrap();
    assert_eq!(recs.len(), 7);
    assert_eq!(mock.hit_count(), 1);
}

#[test]
fn pick_uses_numeric_version_not_lexical() {
    let mk = |v: &str| Record {
        version: v.into(),
        ..tiny_record("model.enes", "model", "en", "es")
    };
    let recs = vec![mk("1.0"), mk("10.0"), mk("2.5")];
    // Lexical max would be "2.5"; numeric max is "10.0".
    assert_eq!(
        fxtranslate::remote::pick(&recs, "model", "en", "es")
            .unwrap()
            .version,
        "10.0"
    );
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

// ---- Decompression + hashing ------------------------------------------------

#[test]
fn decompress_and_hash_roundtrip() {
    let plain = zstd_decode(&fixture("tiny.bin.zst")).unwrap();
    assert_eq!(plain, TINY_PLAIN);
    assert_eq!(sha256_hex(&plain), TINY_HASH);
}

// ---- Cache behavior ---------------------------------------------------------

#[test]
fn cache_downloads_then_hits() {
    let cache = tmp_cache();
    let rec = tiny_record("model.enes.bin", "model", "en", "es");
    let mock = MockHttp::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

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
    let mock = MockHttp::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

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
    let mock = MockHttp::new().route(&rec.cdn_url(), fixture("tiny.bin.zst"));

    let err = cache.ensure(&mock, &rec).unwrap_err();
    assert!(err.contains("hash mismatch"), "got: {err}");
    assert!(
        !cache.pair_dir("en", "es").join(&rec.name).is_file(),
        "no bad file left"
    );
}

// ---- ensure_model wiring (shared vs split vocab) ----------------------------

#[test]
fn ensure_model_shared_vocab() {
    let cache = tmp_cache();
    let recs = vec![
        tiny_record("model.enes.bin", "model", "en", "es"),
        tiny_record("vocab.enes.spm", "vocab", "en", "es"),
        tiny_record("lex.enes.bin", "lex", "en", "es"),
    ];
    let mut mock = MockHttp::new();
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
    let mut mock = MockHttp::new();
    for r in &recs {
        mock = mock.route(&r.cdn_url(), fixture("tiny.bin.zst"));
    }
    let files = ensure_model(&mock, &cache, &recs, "en", "ja").unwrap();
    assert_ne!(files.src_vocab, files.trg_vocab, "split vocab differs");
    assert!(files.lex.is_none(), "no shortlist for CJK split pair");
}
