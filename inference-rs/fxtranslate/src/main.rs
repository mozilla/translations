//! `fxtranslate` — a batteries-included CLI for Firefox Translations models.
//!
//! ```text
//! fxtranslate list [prefix]              # enumerate available <src>-<trg> pairs
//! fxtranslate <src> <trg> [text…]        # translate args, else stdin lines, else REPL
//!   --cache-dir <DIR>                    # override the model cache location
//! ```
//!
//! With no text and a piped stdin it translates line-by-line (like marian's
//! stdin/stdout mode); on a TTY it drops into an interactive prompt. Models are
//! discovered from Remote Settings and cached under the platform cache dir
//! (`~/Library/Caches` / `$XDG_CACHE_HOME` / `%LOCALAPPDATA%`)`/fxtranslate`.

use std::io::{BufRead, IsTerminal, Write};
use std::process::ExitCode;

use fxtranslate::cache::{ensure_model, Cache, ModelFiles};
use fxtranslate::http::{Http, UreqHttp};
use fxtranslate::lang::display_name;
use fxtranslate::remote::{fetch_records, language_matches, pairs};
use inference_rs::engine::Engine;

const USAGE: &str = "\
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

const LIST_USAGE: &str = "\
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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fxtranslate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut cache_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut help = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => help = true,
            "--cache-dir" => {
                cache_dir = Some(args.next().ok_or("--cache-dir needs a path")?);
            }
            _ => positional.push(a),
        }
    }

    let http = UreqHttp;
    match positional.first().map(String::as_str) {
        Some("list") => {
            if help {
                println!("{LIST_USAGE}");
                return Ok(());
            }
            cmd_list(&http, positional.get(1).map(String::as_str))
        }
        _ if help => {
            println!("{USAGE}");
            Ok(())
        }
        None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(_) => {
            if positional.len() < 2 {
                return Err(format!(
                    "expected `<src> <trg>` or `list`; got `{}`\n\n{USAGE}",
                    positional.join(" ")
                ));
            }
            let cache = match &cache_dir {
                Some(d) => Cache::with_root(d),
                None => Cache::locate(),
            };
            let src = &positional[0];
            let trg = &positional[1];
            let text = positional[2..].join(" ");
            cmd_translate(&http, &cache, src, trg, &text)
        }
    }
}

/// Enumerate available pairs from Remote Settings. An optional `query` filters to
/// pairs where the language appears on either side (both directions), or a
/// `src-trg` prefix; see [`language_matches`]. Lines show display names + tags.
fn cmd_list(http: &dyn Http, query: Option<&str>) -> Result<(), String> {
    let records = fetch_records(http)?;
    let all = pairs(&records);
    let shown: Vec<_> = all
        .iter()
        .filter(|(s, t)| match query {
            Some(q) => language_matches(s, t, q),
            None => true,
        })
        .collect();
    if shown.is_empty() {
        return Err(format!(
            "no model pairs match `{}` ({} pairs available; try `fxtranslate list`)",
            query.unwrap_or(""),
            all.len()
        ));
    }
    // Column widths from the display names (Unicode scalar count, matching how
    // Rust's `{:<width$}` pads), so the tag / arrow / target columns line up.
    let w_src = shown
        .iter()
        .map(|(s, _)| display_name(s).chars().count())
        .max()
        .unwrap_or(0);
    let w_trg = shown
        .iter()
        .map(|(_, t)| display_name(t).chars().count())
        .max()
        .unwrap_or(0);

    // Color only on an interactive stdout, and honor NO_COLOR.
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let (cyan, green, dim, reset) = if color {
        ("\x1b[36m", "\x1b[32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };

    for (s, t) in &shown {
        // Pad the plain names to width before wrapping in color, so the escape
        // bytes never count toward the column width.
        let sname = format!("{:<w_src$}", display_name(s));
        let tname = format!("{:<w_trg$}", display_name(t));
        println!(
            "{cyan}{sname}{reset} {dim}({s}){reset} {dim}→{reset} {green}{tname}{reset} {dim}({t}){reset}"
        );
    }
    eprintln!("[{} pairs]", shown.len());
    Ok(())
}

/// Ensure the model is cached, load the engine, then translate args / stdin / REPL.
fn cmd_translate(
    http: &dyn Http,
    cache: &Cache,
    src: &str,
    trg: &str,
    text: &str,
) -> Result<(), String> {
    eprintln!("[fxtranslate] resolving {src}→{trg} model…");
    let records = fetch_records(http)?;
    let files = ensure_model(http, cache, records.as_slice(), src, trg)?;
    let engine = load_engine(&files)?;
    eprintln!("[fxtranslate] ready ({src}→{trg}).");

    if !text.is_empty() {
        println!("{}", engine.translate(text));
        return Ok(());
    }

    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        repl(&engine, src, trg)
    } else {
        // Pipe mode: one translation per input line (marian-style).
        let out = std::io::stdout();
        let mut out = out.lock();
        for line in stdin.lock().lines() {
            let line = line.map_err(|e| e.to_string())?;
            writeln!(out, "{}", engine.translate(&line)).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Minimal interactive prompt: a line in, its translation out, until EOF.
fn repl(engine: &Engine, src: &str, trg: &str) -> Result<(), String> {
    eprintln!("Interactive {src}→{trg}. Type a sentence and press Enter; Ctrl-D to quit.");
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        eprint!("{src}→{trg}» ");
        std::io::stderr().flush().ok();
        line.clear();
        let n = stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            eprintln!();
            return Ok(()); // EOF
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        println!("{}", engine.translate(text));
    }
}

/// Load the engine from cached files (shared vs. split vocab).
fn load_engine(files: &ModelFiles) -> Result<Engine, String> {
    Engine::load(&files.model, &files.src_vocab, &files.trg_vocab)
}
