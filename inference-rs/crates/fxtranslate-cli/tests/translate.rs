//! End-to-end `translate` tests: real argv (`fxtranslate translate <src> <trg>
//! …`) driven through `cli::run` against a fake `Translator` — no model, no
//! network. The [`MockTranslator`] session upper-cases each line and prefixes
//! the pair (`[en→es] HELLO`), so a transcript proves the pair was routed and
//! every line translated independently, without asserting real translations
//! (that's the engine's job, checked against the marian oracle elsewhere).
//!
//! Each test is a **visible transcript snapshot** of stdout+stderr interleaved —
//! exactly what a user sees, including the `[fxtranslate] resolving…/ready`
//! status lines and (for the REPL) the prompts and echoed input.

use std::path::PathBuf;

use fxtranslate_cli::cli::Deps;
use fxtranslate_cli::remote::records_url;
use fxtranslate_cli::translate::EngineTranslator;

mod common;
use common::{assert_transcript, run_transcript, MockFetch, MockTranslator, Streams};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()))
}

/// Build `Deps` for a translate run. `fetch` is unused by `translate` (the
/// translator owns discovery), so a bare `MockFetch` stands in.
fn translate(args: &[&str], translator: &MockTranslator, s: Streams) -> String {
    let fetch = MockFetch::new();
    let deps = Deps {
        fetch: &fetch,
        translator,
    };
    run_transcript(args, &deps, s)
}

/// Args mode: text on the command line → one translation, framed by status lines.
#[test]
fn from_args() {
    assert_transcript(
        "translate args",
        &translate(
            &["translate", "en", "es", "Hello world."],
            &MockTranslator::new(),
            Streams::default(),
        ),
        &[
            "[fxtranslate] resolving en→es model…",
            "[fxtranslate] ready (en→es).",
            "[en→es] HELLO WORLD.",
        ],
    );
}

/// Pipe mode: no args, piped stdin → one translation per input line (marian-style).
#[test]
fn from_piped_stdin() {
    assert_transcript(
        "translate pipe",
        &translate(
            &["translate", "en", "es"],
            &MockTranslator::new(),
            Streams {
                stdin: "Hello world.\nGoodbye.\n".to_string(),
                ..Default::default()
            },
        ),
        &[
            "[fxtranslate] resolving en→es model…",
            "[fxtranslate] ready (en→es).",
            "[en→es] HELLO WORLD.",
            "[en→es] GOODBYE.",
        ],
    );
}

/// REPL mode: stdin is a TTY, so each typed line gets a prompt; `echo` mirrors
/// the input into the transcript (as a terminal would), then the translation.
/// Blank line at the end is the final prompt printed just before EOF (Ctrl-D).
#[test]
fn interactive_repl() {
    assert_transcript(
        "translate repl",
        &translate(
            &["translate", "en", "es"],
            &MockTranslator::new(),
            Streams {
                stdin: "Hello\nWorld peace\n".to_string(),
                stdin_tty: true,
                echo: true,
                ..Default::default()
            },
        ),
        &[
            "[fxtranslate] resolving en→es model…",
            "[fxtranslate] ready (en→es).",
            "Interactive en→es. Type a sentence and press Enter; Ctrl-D to quit.",
            "en→es» Hello",
            "[en→es] HELLO",
            "en→es» World peace",
            "[en→es] WORLD PEACE",
            "en→es» ",
        ],
    );
}

/// Unresolvable pair: `load` fails after the "resolving" line — no "ready", no
/// translation, error reported to stderr.
#[test]
fn unresolvable_pair_errors() {
    assert_transcript(
        "translate error",
        &translate(
            &["translate", "en", "xx", "Hi"],
            &MockTranslator::new().unsupported("en", "xx"),
            Streams::default(),
        ),
        &[
            "[fxtranslate] resolving en→xx model…",
            "fxtranslate: no model for en-xx in Remote Settings",
        ],
    );
}

/// A future-major model is gated out end to end: driven through the *real*
/// `EngineTranslator`, `load` runs Remote Settings discovery + `ensure_model`,
/// whose version gate rejects the v100 `en → fr` model *before* any engine or
/// model files are touched — so the pair reads as unresolvable. Proves the gate
/// on the production translate path with no network and no model.
#[test]
fn unsupported_major_is_gated_out() {
    let fetch = MockFetch::new().route(&records_url(), fixture("rs-version-gate.json"));
    let translator = EngineTranslator::new(&fetch, false);
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };
    assert_transcript(
        "translate version-gate",
        &run_transcript(&["translate", "en", "fr", "Hi"], &deps, Streams::default()),
        &[
            "[fxtranslate] resolving en→fr model…",
            "fxtranslate: no model for en-fr in Remote Settings",
        ],
    );
}
