//! `fxtranslate` — a batteries-included CLI for Firefox Translations models.
//!
//! ```text
//! fxtranslate list [lang]                # enumerate available <src>-<trg> pairs
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
use fxtranslate::cli::{self, Command};
use fxtranslate::http::{Http, UreqHttp};
use fxtranslate::remote::fetch_records;
use inference_rs::engine::Engine;

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse(&args)? {
        Command::Help => {
            println!("{}", cli::USAGE);
            Ok(())
        }
        Command::ListHelp => {
            println!("{}", cli::LIST_USAGE);
            Ok(())
        }
        Command::List { query } => {
            // Color only on an interactive stdout, and honor NO_COLOR.
            let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
            let mut out = std::io::stdout().lock();
            let n = cli::write_list(&UreqHttp, query.as_deref(), color, &mut out)?;
            eprintln!("[{n} pairs]");
            Ok(())
        }
        Command::Translate {
            src,
            trg,
            text,
            cache_dir,
        } => {
            let cache = match cache_dir {
                Some(d) => Cache::with_root(d),
                None => Cache::locate(),
            };
            cmd_translate(&UreqHttp, &cache, &src, &trg, &text)
        }
    }
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
