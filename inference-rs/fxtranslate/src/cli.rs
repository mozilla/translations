//! CLI surface, split from `main.rs` so it is end-to-end testable: [`parse`]
//! turns argv into a [`Command`] (no I/O), and [`write_list`] renders the model
//! list to any `Write` over any [`Http`] — so `tests/cli.rs` can drive
//! args → output against a mocked Remote Settings response, with no network and
//! no engine. `main.rs` is a thin shim: parse, then execute (engine wiring for
//! `translate` lives there since it needs a real model).

use std::io::Write;

use crate::http::Http;
use crate::lang::display_name;
use crate::remote::{fetch_records, language_matches, pairs};

pub const USAGE: &str = "\
fxtranslate — translate with Firefox Translations models

USAGE:
  fxtranslate list [lang]             Enumerate available <src>-<trg> model pairs
                                      (`list --help` for details)
  fxtranslate <src> <trg> [text…]     Translate: args if given, else stdin lines,
                                      else an interactive prompt on a TTY
OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)
  -h, --help          Show this help

EXAMPLES:
  fxtranslate list es
  echo \"Hello world.\" | fxtranslate en es
  fxtranslate en es \"Hello world.\"
  fxtranslate en es                   # interactive";

pub const LIST_USAGE: &str = "\
fxtranslate list — enumerate available translation models

USAGE:
  fxtranslate list [lang]

Every Firefox Translations model translates to or from English, so each model is a
one-way pair (`en → es` and `es → en` are two separate models). With no argument,
all pairs are listed. A [lang] argument filters to every pair where that language
appears on EITHER side — so `fxtranslate list es` shows both `es → en` and
`en → es`. Matching is by prefix, so `zh` catches `zh-Hans` and `zh-Hant`; a full
`src-trg` (e.g. `en-es`) selects one pair. Display names come from Google's
language list; a tag with no known name shows the code.

EXAMPLES:
  fxtranslate list                    # every pair
  fxtranslate list es                 # both directions for Spanish
  fxtranslate list zh                 # zh-Hans / zh-Hant, both directions
  fxtranslate list en-es              # just the en → es pair";

/// A parsed command line.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// Print top-level usage.
    Help,
    /// Print `list`-specific usage.
    ListHelp,
    /// Enumerate model pairs, optionally filtered.
    List { query: Option<String> },
    /// Translate `src`→`trg`; `text` empty = stdin/REPL.
    Translate {
        src: String,
        trg: String,
        text: String,
        cache_dir: Option<String>,
    },
}

/// Parse argv (without the program name) into a [`Command`]. Pure — no I/O — so
/// the whole arg grammar (subcommands, `--help` routing, `--cache-dir`) is unit
/// testable.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let mut cache_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut help = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => help = true,
            "--cache-dir" => {
                cache_dir = Some(it.next().ok_or("--cache-dir needs a path")?.clone());
            }
            _ => positional.push(a.clone()),
        }
    }

    match positional.first().map(String::as_str) {
        Some("list") => {
            if help {
                Ok(Command::ListHelp)
            } else {
                Ok(Command::List {
                    query: positional.get(1).cloned(),
                })
            }
        }
        // `--help` with a non-list (or no) command → top-level help.
        _ if help => Ok(Command::Help),
        None => Ok(Command::Help),
        Some(_) => {
            if positional.len() < 2 {
                return Err(format!(
                    "expected `<src> <trg>` or `list`; got `{}`\n\n{USAGE}",
                    positional.join(" ")
                ));
            }
            Ok(Command::Translate {
                src: positional[0].clone(),
                trg: positional[1].clone(),
                text: positional[2..].join(" "),
                cache_dir,
            })
        }
    }
}

/// Fetch the model records via `http`, filter by `query`
/// ([`language_matches`]), and write the aligned list to `out`. `color` toggles
/// ANSI styling (the caller decides based on TTY / `NO_COLOR`). Returns the
/// number of pairs written. Names are padded (by Unicode scalar count, matching
/// `{:<width$}`) *before* color-wrapping so escape bytes never affect columns.
pub fn write_list(
    http: &dyn Http,
    query: Option<&str>,
    color: bool,
    out: &mut dyn Write,
) -> Result<usize, String> {
    let records = fetch_records(http)?;
    let all = pairs(&records);
    let shown: Vec<_> = all
        .iter()
        .filter(|(s, t)| query.map_or(true, |q| language_matches(s, t, q)))
        .collect();
    if shown.is_empty() {
        return Err(format!(
            "no model pairs match `{}` ({} pairs available; try `fxtranslate list`)",
            query.unwrap_or(""),
            all.len()
        ));
    }

    // Column widths (Unicode scalar counts). The source *tag* is padded too, so a
    // long source script tag like `(zh-Hans)` doesn't push the arrow out of line;
    // the target tag is the last column, so it needs no padding.
    let w_src = shown
        .iter()
        .map(|(s, _)| display_name(s).chars().count())
        .max()
        .unwrap_or(0);
    let w_stag = shown
        .iter()
        .map(|(s, _)| s.chars().count() + 2) // "(" + tag + ")"
        .max()
        .unwrap_or(0);
    let w_trg = shown
        .iter()
        .map(|(_, t)| display_name(t).chars().count())
        .max()
        .unwrap_or(0);

    let (cyan, green, dim, reset) = if color {
        ("\x1b[36m", "\x1b[32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };

    for (s, t) in &shown {
        // Pad each column's plain text before color-wrapping, so escape bytes
        // never count toward width.
        let sname = format!("{:<w_src$}", display_name(s));
        let stag = format!("{:<w_stag$}", format!("({s})"));
        let tname = format!("{:<w_trg$}", display_name(t));
        writeln!(
            out,
            "{cyan}{sname}{reset} {dim}{stag}{reset} {dim}→{reset} {green}{tname}{reset} {dim}({t}){reset}"
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(shown.len())
}
