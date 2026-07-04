//! End-to-end `list` tests: real argv (`fxtranslate list [lang]`) driven through
//! `cli::run` against a checked-in Remote Settings snapshot via the mockable
//! `Fetch` trait — no network, no engine.
//!
//! Each test is a **visible transcript snapshot**: the expected block is the
//! literal CLI output (the aligned table on stdout plus the `[N pairs]` trailer
//! on stderr), so a reviewer can audit formatting — columns, display names, sort
//! order, fallbacks — by scanning the code. On a mismatch the helper prints the
//! actual output as a paste-ready array to drop in.
//!
//! The `rs-list.json` fixture is small but has the edge cases: normal pairs
//! (`es`, `fr`), Chinese script tags (`zh-Hans`/`zh-Hant`), and Norwegian
//! (`nb`, and `nn` — which has no Google display name and falls back to its code),
//! each in both directions to/from English.

use std::path::PathBuf;

use fxtranslate::cli::Deps;
use fxtranslate::remote::records_url;

mod common;
use common::{assert_transcript, run_transcript, MockFetch, MockTranslator, Streams};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// Run `list` argv against the fixture, returning the combined transcript.
/// `color_tty` sets stdout as a terminal (the only thing that enables color).
fn list(args: &[&str], color_tty: bool) -> String {
    let fetch = MockFetch::new().route(&records_url(), fixture("rs-list.json"));
    let translator = MockTranslator::new(); // unused by `list`
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };
    run_transcript(
        args,
        &deps,
        Streams {
            stdout_tty: color_tty,
            ..Default::default()
        },
    )
}

/// `list` rendering — visible transcript snapshots.
mod output {
    use super::*;

    /// The whole table (no filter): sort order, every display name (incl. the `å` in
    /// Norwegian Bokmål, the `nn` code fallback, and the Chinese script names),
    /// five-column alignment, and the `[N pairs]` trailer — auditable at a glance.
    #[test]
    fn all_pairs() {
        assert_transcript(
            "list",
            &list(&["list"], false),
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
                "[12 pairs]",
            ],
        );
    }

    /// A bare language surfaces both directions; equal-width names → no padding.
    #[test]
    fn language_both_directions() {
        assert_transcript(
            "list es",
            &list(&["list", "es"], false),
            &[
                "English (en) → Spanish (es)",
                "Spanish (es) → English (en)",
                "[2 pairs]",
            ],
        );
    }

    /// `zh` (prefix, either side) → both scripts, both directions. The long source
    /// tag `(zh-Hans)` is padded so the arrow still lines up.
    #[test]
    fn chinese_scripts() {
        assert_transcript(
            "list zh",
            &list(&["list", "zh"], false),
            &[
                "English               (en)      → Chinese (Simplified)  (zh-Hans)",
                "English               (en)      → Chinese (Traditional) (zh-Hant)",
                "Chinese (Simplified)  (zh-Hans) → English               (en)",
                "Chinese (Traditional) (zh-Hant) → English               (en)",
                "[4 pairs]",
            ],
        );
    }

    /// `zh-en`: the split query prefix-matches each half — src `zh*` (both scripts),
    /// trg `en` only.
    #[test]
    fn src_trg_pair() {
        assert_transcript(
            "list zh-en",
            &list(&["list", "zh-en"], false),
            &[
                "Chinese (Simplified)  (zh-Hans) → English (en)",
                "Chinese (Traditional) (zh-Hant) → English (en)",
                "[2 pairs]",
            ],
        );
    }

    /// Norwegian: `nb` resolves to "Norwegian Bokmål"; `nn` has no Google name and
    /// shows the bare code.
    #[test]
    fn norwegian_names_and_code_fallback() {
        assert_transcript(
            "list n",
            &list(&["list", "n"], false),
            &[
                "English          (en) → Norwegian Bokmål (nb)",
                "English          (en) → nn               (nn)",
                "Norwegian Bokmål (nb) → English          (en)",
                "nn               (nn) → English          (en)",
                "[4 pairs]",
            ],
        );
    }
}

/// `list` non-happy paths: no-match error and color gating.
mod edges {
    use super::*;

    #[test]
    fn no_match_errors() {
        // No table, no `[N pairs]` trailer — just the error, on stderr.
        assert_transcript(
            "list xx",
            &list(&["list", "xx"], false),
            &["fxtranslate: no model pairs match `xx` (12 pairs available; try `fxtranslate list`)"],
        );
    }

    #[test]
    fn color_only_on_a_tty() {
        // Color is a pure add-on gated by stdout being a TTY (ANSI is illegible in a
        // snapshot, so this checks presence/absence rather than exact bytes).
        assert!(
            !list(&["list", "es"], false).contains('\x1b'),
            "no ANSI when stdout is not a TTY"
        );
        let colored = list(&["list", "es"], true);
        assert!(colored.contains("\x1b[36m"), "cyan source on a TTY");
        assert!(colored.contains("\x1b[0m"), "reset present");
    }
}
