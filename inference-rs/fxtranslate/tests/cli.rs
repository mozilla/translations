//! End-to-end CLI tests: argv → `Command`, and the `list` renderer's argv →
//! output, driven against a checked-in Remote Settings snapshot through the
//! mockable `Http` trait. No network, no engine — so a change to arg parsing,
//! filtering, display names, or column/color formatting is caught here against a
//! concrete expected output (not a tautology).
//!
//! The `rs-list.json` fixture is small but has the edge cases: normal pairs
//! (`es`, `fr`), Chinese script tags (`zh-Hans`/`zh-Hant`), and Norwegian
//! (`nb`, and `nn` — which has no Google display name and falls back to its code),
//! each in both directions to/from English.

use std::path::PathBuf;

use fxtranslate::cli::{parse, write_list, Command};
use fxtranslate::http::MockHttp;
use fxtranslate::remote::records_url;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// Render `list [query]` against the fixture, returning captured stdout.
fn list(query: Option<&str>, color: bool) -> String {
    let mock = MockHttp::new().route(&records_url(), fixture("rs-list.json"));
    let mut buf = Vec::new();
    write_list(&mock, query, color, &mut buf).expect("write_list ok");
    String::from_utf8(buf).unwrap()
}

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// ---- argv → Command ---------------------------------------------------------

#[test]
fn parse_list_and_query() {
    assert_eq!(
        parse(&args(&["list"])).unwrap(),
        Command::List { query: None }
    );
    assert_eq!(
        parse(&args(&["list", "es"])).unwrap(),
        Command::List {
            query: Some("es".into())
        }
    );
}

#[test]
fn parse_help_routing() {
    // `list --help` is list-specific; bare `--help` and no args are top-level.
    assert_eq!(
        parse(&args(&["list", "--help"])).unwrap(),
        Command::ListHelp
    );
    assert_eq!(parse(&args(&["--help"])).unwrap(), Command::Help);
    assert_eq!(parse(&args(&[])).unwrap(), Command::Help);
}

#[test]
fn parse_translate_with_text_and_cache_dir() {
    assert_eq!(
        parse(&args(&["en", "es", "Hello", "world"])).unwrap(),
        Command::Translate {
            src: "en".into(),
            trg: "es".into(),
            text: "Hello world".into(),
            cache_dir: None,
        }
    );
    assert_eq!(
        parse(&args(&["--cache-dir", "/tmp/c", "en", "es"])).unwrap(),
        Command::Translate {
            src: "en".into(),
            trg: "es".into(),
            text: String::new(),
            cache_dir: Some("/tmp/c".into()),
        }
    );
}

#[test]
fn parse_rejects_bad_input() {
    assert!(
        parse(&args(&["en"])).is_err(),
        "one positional is not a pair"
    );
    assert!(
        parse(&args(&["--cache-dir"])).is_err(),
        "--cache-dir with no value errors"
    );
}

// ---- list rendering: filtering + display names ------------------------------

#[test]
fn list_language_is_bidirectional() {
    // Exact output: `es` on either side, both directions; 7-char names, no padding.
    assert_eq!(
        list(Some("es"), false),
        "English (en) → Spanish (es)\nSpanish (es) → English (en)\n"
    );
}

#[test]
fn list_all_pairs_count() {
    // The fixture has 6 languages × 2 directions.
    assert_eq!(list(None, false).lines().count(), 12);
}

#[test]
fn list_chinese_script_tags_get_names_both_directions() {
    let out = list(Some("zh"), false);
    assert_eq!(out.lines().count(), 4, "zh-Hans/zh-Hant × 2 directions");
    assert!(out.contains("Chinese (Simplified)"));
    assert!(out.contains("Chinese (Traditional)"));
    assert!(out.contains("(zh-Hans)") && out.contains("(zh-Hant)"));
}

#[test]
fn list_src_trg_query_prefix_matches_each_half() {
    // `zh-en`: src prefix `zh` (both scripts) → trg `en` only.
    let out = list(Some("zh-en"), false);
    assert_eq!(out.lines().count(), 2);
    assert!(out.lines().all(|l| l.trim_end().ends_with("(en)")));
    assert!(out.contains("(zh-Hans)") && out.contains("(zh-Hant)"));
}

#[test]
fn list_unknown_tag_falls_back_to_code() {
    // `nn` (Norwegian Nynorsk) has no Google name → display name is the code.
    let out = list(Some("nn"), false);
    assert_eq!(out.lines().count(), 2);
    // The `nn → en` line renders the source display name as the bare code `nn`.
    assert!(
        out.lines().any(|l| l.starts_with("nn ")),
        "expected an `nn …` line (code fallback), got:\n{out}"
    );
    // Norwegian Bokmål, by contrast, does resolve to a name.
    assert!(list(Some("nb"), false).contains("Norwegian Bokmål"));
}

#[test]
fn list_no_match_errors() {
    let mock = MockHttp::new().route(&records_url(), fixture("rs-list.json"));
    let mut buf = Vec::new();
    let err = write_list(&mock, Some("xx"), false, &mut buf).unwrap_err();
    assert!(err.contains("no model pairs match `xx`"), "got: {err}");
    assert!(buf.is_empty(), "nothing written on no-match");
}

// ---- list rendering: column alignment + color ------------------------------

#[test]
fn list_columns_are_aligned() {
    // `zh` mixes short (English) and long (Chinese (Traditional)) names AND a long
    // source tag (zh-Hans), so alignment is non-trivial. The arrow and the target
    // tag must sit at the same byte column on every line.
    let out = list(Some("zh"), false);
    let arrow_cols: Vec<_> = out.lines().map(|l| l.find(" → ").unwrap()).collect();
    let ttag_cols: Vec<_> = out.lines().map(|l| l.rfind(" (").unwrap()).collect();
    assert!(
        arrow_cols.iter().all(|&c| c == arrow_cols[0]),
        "arrow column misaligned in:\n{out}"
    );
    assert!(
        ttag_cols.iter().all(|&c| c == ttag_cols[0]),
        "target-tag column misaligned in:\n{out}"
    );
}

#[test]
fn list_color_only_when_requested() {
    let plain = list(Some("es"), false);
    let colored = list(Some("es"), true);
    assert!(!plain.contains('\x1b'), "no ANSI when color=false");
    assert!(colored.contains("\x1b[36m"), "cyan source when color=true");
    assert!(colored.contains("\x1b[0m"), "reset present");
    // Same visible text once escapes are stripped is not asserted here; the point
    // is that color is a pure add-on gated by the flag.
}
