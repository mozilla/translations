//! Cross-cutting CLI grammar: help routing, unknown/incomplete commands, and the
//! `parse` bits whose effect isn't observable in a command transcript (the
//! `--cache-dir` plumbing). Per-subcommand behavior lives in `list.rs` /
//! `translate.rs`; this file covers everything that isn't one subcommand.

use fxtranslate_cli::cli::{parse, Command, Deps};

mod common;
use common::{run_transcript, MockFetch, MockTranslator, Streams};

/// Drive `cli::run(args)` end-to-end (deps are unused by help/grammar paths, so
/// bare fakes stand in), returning the transcript.
fn cli(args: &[&str]) -> String {
    let fetch = MockFetch::new();
    let translator = MockTranslator::new();
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };
    run_transcript(args, &deps, Streams::default())
}

fn argv(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// `run` routes the help commands to the usage text (with a trailing newline).
/// The usage strings themselves are auditable inline in `src/cli.rs`.
mod help {
    use super::*;

    #[test]
    fn top_level() {
        let expected = format!("{}\n", fxtranslate_cli::cli::USAGE);
        assert_eq!(cli(&["--help"]), expected);
        assert_eq!(cli(&[]), expected, "no args is also top-level help");
    }

    #[test]
    fn list_specific() {
        assert_eq!(
            cli(&["list", "--help"]),
            format!("{}\n", fxtranslate_cli::cli::LIST_USAGE)
        );
    }
}

/// Bad/incomplete argv → an informative error on stderr, then the usage help.
mod grammar {
    use super::*;

    #[test]
    fn unknown_command() {
        let t = cli(&["frobnicate"]);
        assert_eq!(
            t.lines().next().unwrap(),
            "fxtranslate: unknown command `frobnicate`; expected `translate` or `list`"
        );
        assert!(t.contains("USAGE:"), "usage help follows the error");
    }

    #[test]
    fn bare_pair_is_not_a_command() {
        // `translate` is explicit now: a leading language reads as an unknown command.
        let t = cli(&["en", "es", "hola"]);
        assert_eq!(
            t.lines().next().unwrap(),
            "fxtranslate: unknown command `en`; expected `translate` or `list`"
        );
    }

    #[test]
    fn translate_needs_two_langs() {
        let t = cli(&["translate", "en"]);
        assert_eq!(
            t.lines().next().unwrap(),
            "fxtranslate: `translate` needs `<src> <trg> [text…]`; got `translate en`"
        );
        assert!(t.contains("USAGE:"), "usage help follows the error");
    }

    #[test]
    fn cache_dir_needs_a_value() {
        assert_eq!(
            cli(&["--cache-dir"]),
            "fxtranslate: --cache-dir needs a path\n"
        );
    }
}

/// `parse` cases whose effect isn't visible in a transcript — chiefly the
/// `--cache-dir` override plumbed through to `Command::Translate`.
mod parse_grammar {
    use super::*;

    #[test]
    fn translate_joins_text_and_captures_cache_dir() {
        assert_eq!(
            parse(&argv(&[
                "--cache-dir",
                "/tmp/c",
                "translate",
                "en",
                "es",
                "Hi",
                "there"
            ]))
            .unwrap(),
            Command::Translate {
                src: "en".into(),
                trg: "es".into(),
                text: "Hi there".into(),
                cache_dir: Some("/tmp/c".into()),
            }
        );
    }

    #[test]
    fn list_and_help_routing() {
        assert_eq!(
            parse(&argv(&["list", "es"])).unwrap(),
            Command::List {
                query: Some("es".into())
            }
        );
        assert_eq!(
            parse(&argv(&["list", "--help"])).unwrap(),
            Command::ListHelp
        );
        assert_eq!(parse(&argv(&["--help"])).unwrap(), Command::Help);
        assert_eq!(parse(&argv(&[])).unwrap(), Command::Help);
    }
}
