//! CLI surface, split from `main.rs` so the whole thing is end-to-end testable
//! without a network or an engine. [`parse`] turns argv into a [`Command`] (no
//! I/O); [`run`] then dispatches and executes it against injected [`Deps`] (a
//! [`Fetch`] for `list`, a [`Translator`] for `translate`) and [`Io`] (captured
//! streams + explicit TTY/`NO_COLOR` facts). So `tests/` can drive
//! argv → transcript against fakes, and `main.rs` is a thin shim that wires the
//! real network, engine, and terminal into `run`.

use std::io::{BufRead, Write};
use std::process::ExitCode;

use crate::translate::{Session, Translator};
use fxtranslate::fetch::Fetch;
use fxtranslate::lang::display_name;
use fxtranslate::remote::{fetch_records, language_matches, pairs};

pub const USAGE: &str = "\
fxtranslate — translate with Firefox Translations models

USAGE:
  fxtranslate list [lang]                     Enumerate available <src>-<trg> pairs
                                              (`list --help` for details)
  fxtranslate translate <src> <trg> [text…]   Translate: args if given, else stdin
                                              lines, else an interactive TTY prompt
OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)
  -h, --help          Show this help

EXAMPLES:
  fxtranslate list es
  echo \"Hello world.\" | fxtranslate translate en es
  fxtranslate translate en es \"Hello world.\"
  fxtranslate translate en es                 # interactive";

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
        Some("translate") => {
            // `translate` is explicit: `translate <src> <trg> [text…]`.
            if positional.len() < 3 {
                return Err(format!(
                    "`translate` needs `<src> <trg> [text…]`; got `{}`\n\n{USAGE}",
                    positional.join(" ")
                ));
            }
            Ok(Command::Translate {
                src: positional[1].clone(),
                trg: positional[2].clone(),
                text: positional[3..].join(" "),
                cache_dir,
            })
        }
        Some(other) => Err(format!(
            "unknown command `{other}`; expected `translate` or `list`\n\n{USAGE}"
        )),
    }
}

/// Fetch the model records via `fetch`, filter by `query`
/// ([`language_matches`]), and write the aligned list to `out`. `color` toggles
/// ANSI styling (the caller decides based on TTY / `NO_COLOR`). Returns the
/// number of pairs written. Names are padded (by Unicode scalar count, matching
/// `{:<width$}`) *before* color-wrapping so escape bytes never affect columns.
pub fn write_list(
    fetch: &dyn Fetch,
    query: Option<&str>,
    color: bool,
    out: &mut dyn Write,
) -> Result<usize, String> {
    let records = fetch_records(fetch)?;
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

/// The external dependencies [`run`] executes against: the network (for `list`
/// discovery) and the translator (for `translate`). `main.rs` passes the real
/// implementations; tests pass fakes.
pub struct Deps<'a> {
    pub fetch: &'a dyn Fetch,
    pub translator: &'a dyn Translator,
}

/// The I/O and terminal environment [`run`] executes against. Injected — rather
/// than probed from the process — so tests capture streams into buffers and set
/// the TTY / `NO_COLOR` facts explicitly.
pub struct Io<'a> {
    pub stdin: &'a mut dyn BufRead,
    pub stdout: &'a mut dyn Write,
    pub stderr: &'a mut dyn Write,
    /// stdin is a terminal → `translate` drops into the interactive REPL rather
    /// than reading piped lines.
    pub stdin_is_tty: bool,
    /// stdout is a terminal → `list` may color (also gated by `no_color`).
    pub stdout_is_tty: bool,
    /// `NO_COLOR` is set in the environment.
    pub no_color: bool,
}

/// Parse argv, dispatch to the matching command, and execute it against
/// `deps`/`io`. Errors are reported to `io.stderr` (prefixed `fxtranslate:`) so
/// they share the transcript with normal output; returns the process exit status.
pub fn run(args: &[String], deps: &Deps, io: &mut Io) -> ExitCode {
    match dispatch(args, deps, io) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(io.stderr, "fxtranslate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String], deps: &Deps, io: &mut Io) -> Result<(), String> {
    match parse(args)? {
        Command::Help => writeln!(io.stdout, "{USAGE}").map_err(|e| e.to_string()),
        Command::ListHelp => writeln!(io.stdout, "{LIST_USAGE}").map_err(|e| e.to_string()),
        Command::List { query } => {
            // Color only on an interactive stdout, and honor NO_COLOR.
            let color = io.stdout_is_tty && !io.no_color;
            let n = write_list(deps.fetch, query.as_deref(), color, io.stdout)?;
            writeln!(io.stderr, "[{n} pairs]").map_err(|e| e.to_string())
        }
        Command::Translate {
            src,
            trg,
            text,
            cache_dir,
        } => run_translate(deps.translator, io, &src, &trg, &text, cache_dir.as_deref()),
    }
}

/// Load the `src`→`trg` session, then translate `text` if given, else stdin
/// lines (pipe mode), else an interactive prompt (TTY). Status lines go to
/// stderr so piped stdout carries only translations.
fn run_translate(
    translator: &dyn Translator,
    io: &mut Io,
    src: &str,
    trg: &str,
    text: &str,
    cache_dir: Option<&str>,
) -> Result<(), String> {
    writeln!(io.stderr, "[fxtranslate] resolving {src}→{trg} model…").map_err(|e| e.to_string())?;
    let session = translator.load(src, trg, cache_dir)?;
    writeln!(io.stderr, "[fxtranslate] ready ({src}→{trg}).").map_err(|e| e.to_string())?;

    if !text.is_empty() {
        writeln!(io.stdout, "{}", session.translate(text)).map_err(|e| e.to_string())?;
        return Ok(());
    }

    if io.stdin_is_tty {
        return repl(session.as_ref(), io, src, trg);
    }

    // Pipe mode: one translation per input line (marian-style).
    let mut line = String::new();
    loop {
        line.clear();
        let n = io.stdin.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(()); // EOF
        }
        // Strip only the line terminator, matching `BufRead::lines`.
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        writeln!(io.stdout, "{}", session.translate(&line)).map_err(|e| e.to_string())?;
    }
}

/// Minimal interactive prompt: a line in, its translation out, until EOF.
fn repl(session: &dyn Session, io: &mut Io, src: &str, trg: &str) -> Result<(), String> {
    writeln!(
        io.stderr,
        "Interactive {src}→{trg}. Type a sentence and press Enter; Ctrl-D to quit."
    )
    .map_err(|e| e.to_string())?;
    let mut line = String::new();
    loop {
        write!(io.stderr, "{src}→{trg}» ").map_err(|e| e.to_string())?;
        io.stderr.flush().map_err(|e| e.to_string())?;
        line.clear();
        let n = io.stdin.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            writeln!(io.stderr).map_err(|e| e.to_string())?;
            return Ok(()); // EOF
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        writeln!(io.stdout, "{}", session.translate(text)).map_err(|e| e.to_string())?;
    }
}
