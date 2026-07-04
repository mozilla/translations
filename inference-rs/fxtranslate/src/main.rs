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
use fxtranslate::remote::{fetch_records, pairs};
use inference_rs::engine::Engine;

const USAGE: &str = "\
fxtranslate — translate with Firefox Translations models

USAGE:
  fxtranslate list [prefix]           Enumerate available <src>-<trg> model pairs
  fxtranslate <src> <trg> [text…]     Translate: args if given, else stdin lines,
                                      else an interactive prompt on a TTY
OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)
  -h, --help          Show this help

EXAMPLES:
  fxtranslate list en
  echo \"Hello world.\" | fxtranslate en es
  fxtranslate en es \"Hello world.\"
  fxtranslate en es                   # interactive";

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
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--cache-dir" => {
                cache_dir = Some(args.next().ok_or("--cache-dir needs a path")?);
            }
            _ => positional.push(a),
        }
    }

    let cache = match &cache_dir {
        Some(d) => Cache::with_root(d),
        None => Cache::locate(),
    };
    let http = UreqHttp;

    match positional.first().map(String::as_str) {
        None => {
            println!("{USAGE}");
            Ok(())
        }
        Some("list") => cmd_list(&http, positional.get(1).map(String::as_str)),
        Some(_) => {
            if positional.len() < 2 {
                return Err(format!(
                    "expected `<src> <trg>` or `list`; got `{}`\n\n{USAGE}",
                    positional.join(" ")
                ));
            }
            let src = &positional[0];
            let trg = &positional[1];
            let text = positional[2..].join(" ");
            cmd_translate(&http, &cache, src, trg, &text)
        }
    }
}

/// Enumerate available pairs from Remote Settings, optionally filtered by a source
/// or `src-trg` prefix.
fn cmd_list(http: &dyn Http, prefix: Option<&str>) -> Result<(), String> {
    let records = fetch_records(http)?;
    let all = pairs(&records);
    let shown: Vec<_> = all
        .iter()
        .filter(|(s, t)| match prefix {
            Some(p) => s.starts_with(p) || format!("{s}-{t}").starts_with(p),
            None => true,
        })
        .collect();
    if shown.is_empty() {
        return Err(format!(
            "no model pairs match `{}` ({} pairs available)",
            prefix.unwrap_or(""),
            all.len()
        ));
    }
    for (s, t) in &shown {
        println!("{s} → {t}");
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
