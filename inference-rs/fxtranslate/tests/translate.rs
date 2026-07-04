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

use fxtranslate::cli::Deps;

mod common;
use common::{assert_transcript, run_transcript, MockFetch, MockTranslator, Streams};

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
