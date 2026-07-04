//! End-to-end CLI tests: argv → `Command`, and the `list` renderer's argv →
//! output, driven against a checked-in Remote Settings snapshot through the
//! mockable `Fetch` trait. No network, no engine.
//!
//! The list-output tests are written as **visible line-by-line snapshots**: the
//! expected block is the literal CLI output, so a reviewer can audit formatting
//! (columns, display names, sort order, fallbacks) by scanning the code. When the
//! renderer changes, update the code *and* the block together; on a mismatch the
//! helper prints the actual output as a paste-ready array to drop in.
//!
//! The `rs-list.json` fixture is small but has the edge cases: normal pairs
//! (`es`, `fr`), Chinese script tags (`zh-Hans`/`zh-Hant`), and Norwegian
//! (`nb`, and `nn` — which has no Google display name and falls back to its code),
//! each in both directions to/from English.

use std::path::PathBuf;

use fxtranslate::cli::{parse, write_list, Command};
use fxtranslate::remote::records_url;

mod common;
use common::MockFetch;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// Render `list [query]` against the fixture, returning captured stdout.
fn list(query: Option<&str>, color: bool) -> String {
    let mock = MockFetch::new().route(&records_url(), fixture("rs-list.json"));
    let mut buf = Vec::new();
    write_list(&mock, query, color, &mut buf).expect("write_list ok");
    String::from_utf8(buf).unwrap()
}

/// Assert `list [query]` (no color) renders exactly `expected`, one entry per
/// output line. On mismatch, print the actual output as a paste-ready block.
fn assert_list(query: Option<&str>, expected: &[&str]) {
    let out = list(query, false);
    let got: Vec<&str> = out.lines().collect();
    if got != expected {
        let paste = got
            .iter()
            .map(|l| format!("            {l:?},"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "`list {}` output changed. If intended, replace the expected block with:\n&[\n{paste}\n        ]\n",
            query.unwrap_or("")
        );
    }
}

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// argv → [`Command`].
mod parse_args {
    use super::*;

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
}

/// `list` rendering — visible line-by-line output snapshots.
mod list_output {
    use super::*;

    /// The whole table (no filter): sort order, every display name (incl. the `å` in
    /// Norwegian Bokmål, the `nn` code fallback, and the Chinese script names), and
    /// five-column alignment — all auditable at a glance.
    #[test]
    fn list_all_pairs() {
        assert_list(
            None,
            &[
                "English               (en)      → Spanish               (es)",
                "English               (en)      → French                (fr)",
                "English               (en)      → Norwegian Bokmål      (nb)",
                "English               (en)      → nn                    (nn)",
                "English               (en)      → Chinese (Simplified)  (zh-Hans)",
                "English               (en)      → Chinese (Traditional) (zh-Hant)",
                "Spanish               (es)      → English               (en)",
                "French                (fr)      → English               (en)",
                "Norwegian Bokmål      (nb)      → English               (en)",
                "nn                    (nn)      → English               (en)",
                "Chinese (Simplified)  (zh-Hans) → English               (en)",
                "Chinese (Traditional) (zh-Hant) → English               (en)",
            ],
        );
    }

    /// A bare language surfaces both directions; equal-width names → no padding.
    #[test]
    fn list_language_both_directions() {
        assert_list(
            Some("es"),
            &["English (en) → Spanish (es)", "Spanish (es) → English (en)"],
        );
    }

    /// `zh` (prefix, either side) → both scripts, both directions. The long source
    /// tag `(zh-Hans)` is padded so the arrow still lines up.
    #[test]
    fn list_chinese_scripts() {
        assert_list(
            Some("zh"),
            &[
                "English               (en)      → Chinese (Simplified)  (zh-Hans)",
                "English               (en)      → Chinese (Traditional) (zh-Hant)",
                "Chinese (Simplified)  (zh-Hans) → English               (en)",
                "Chinese (Traditional) (zh-Hant) → English               (en)",
            ],
        );
    }

    /// `zh-en`: the split query prefix-matches each half — src `zh*` (both scripts),
    /// trg `en` only.
    #[test]
    fn list_src_trg_pair() {
        assert_list(
            Some("zh-en"),
            &[
                "Chinese (Simplified)  (zh-Hans) → English (en)",
                "Chinese (Traditional) (zh-Hant) → English (en)",
            ],
        );
    }

    /// Norwegian: `nb` resolves to "Norwegian Bokmål"; `nn` has no Google name and
    /// shows the bare code.
    #[test]
    fn list_norwegian_names_and_code_fallback() {
        assert_list(
            Some("n"),
            &[
                "English          (en) → Norwegian Bokmål (nb)",
                "English          (en) → nn               (nn)",
                "Norwegian Bokmål (nb) → English          (en)",
                "nn               (nn) → English          (en)",
            ],
        );
    }
}

/// `list` non-happy paths: no-match error and color gating.
mod list_edges {
    use super::*;

    #[test]
    fn list_no_match_errors() {
        let mock = MockFetch::new().route(&records_url(), fixture("rs-list.json"));
        let mut buf = Vec::new();
        let err = write_list(&mock, Some("xx"), false, &mut buf).unwrap_err();
        assert!(err.contains("no model pairs match `xx`"), "got: {err}");
        assert!(buf.is_empty(), "nothing written on no-match");
    }

    #[test]
    fn list_color_only_when_requested() {
        // Color is a pure add-on gated by the flag (ANSI is illegible in a snapshot,
        // so this checks presence/absence rather than the exact bytes).
        assert!(
            !list(Some("es"), false).contains('\x1b'),
            "no ANSI when color=false"
        );
        let colored = list(Some("es"), true);
        assert!(colored.contains("\x1b[36m"), "cyan source when color=true");
        assert!(colored.contains("\x1b[0m"), "reset present");
    }
}
