//! `inference-rs` CLI: translate text, or inspect a reference trace.
//!
//! Usage:
//!   # translate (greedy, en→fr happy path)
//!   cargo run -- translate <model.bin> <src.spm> <trg.spm> "Hello world."
//!   cargo run -- translate <model.bin> <vocab.spm> "Hello world."   # shared vocab
//!
//!   # inspect a recorded trace
//!   cargo run -- trace <trace-path> [num-records-to-print]

use std::collections::BTreeMap;
use std::process::ExitCode;

use inference_rs::engine::Engine;
use inference_rs::trace::Trace;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("translate") => translate(&args[1..]),
        Some("trace") => inspect_trace(&args[1..]),
        _ => {
            eprintln!("usage:");
            eprintln!("  inference-rs translate <model.bin> <src.spm> [trg.spm] <text>");
            eprintln!("  inference-rs trace <trace-path> [num-records]");
            ExitCode::FAILURE
        }
    }
}

/// `translate <model> <src.spm> [trg.spm] <text>` — greedy translation.
fn translate(args: &[String]) -> ExitCode {
    // Accept either a shared vocab (3 args) or split src/trg (4 args).
    let (model, src_vocab, trg_vocab, text) = match args {
        [model, vocab, text] => (model, vocab, vocab, text),
        [model, src, trg, text] => (model, src, trg, text),
        _ => {
            eprintln!("usage: inference-rs translate <model.bin> <src.spm> [trg.spm] <text>");
            return ExitCode::FAILURE;
        }
    };

    let mut engine = match Engine::load(model, src_vocab, trg_vocab) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load engine: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Attach a lexical shortlist if one sits beside the model (as the shipped
    // model directories bundle it) — it's the reference output-projection path.
    if let Some(path) = find_shortlist(model) {
        match inference_rs::shortlist::Shortlist::load(&path) {
            Ok(sl) => {
                eprintln!("[shortlist] {path}");
                engine = engine.with_shortlist(sl);
            }
            Err(e) => eprintln!("[shortlist] ignoring {path}: {e}"),
        }
    }

    println!("{}", engine.translate(text));
    ExitCode::SUCCESS
}

/// Look for a `lex*.bin` shortlist next to the model file.
fn find_shortlist(model: &str) -> Option<String> {
    let dir = std::path::Path::new(model).parent()?;
    let mut hits: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (name.starts_with("lex") && name.ends_with(".bin"))
                .then(|| e.path().to_string_lossy().into_owned())
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// `trace <path> [n]` — the original trace inspector.
fn inspect_trace(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("usage: inference-rs trace <trace-path> [num-records]");
        return ExitCode::FAILURE;
    };
    let head = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(10usize);

    let trace = match Trace::load(path) {
        Ok(trace) => trace,
        Err(e) => {
            eprintln!("failed to load trace '{path}': {e}");
            return ExitCode::FAILURE;
        }
    };

    print_summary(&trace, head);
    ExitCode::SUCCESS
}

fn print_summary(trace: &Trace, head: usize) {
    let total_bytes: usize = trace.records.iter().map(|r| r.data.len()).sum();
    println!("trace version {}", trace.version);
    println!(
        "{} records, {} tensor bytes ({:.1} MiB)",
        trace.len(),
        total_bytes,
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    let mut op_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for record in &trace.records {
        *op_counts.entry(record.op_type.as_str()).or_default() += 1;
    }
    let mut ops: Vec<(&str, usize)> = op_counts.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    println!("\nop types ({} distinct):", ops.len());
    for (op, count) in ops.iter().take(15) {
        println!("  {count:>6}  {op}");
    }

    println!("\nfirst {} records:", head.min(trace.len()));
    for (index, record) in trace.records.iter().take(head).enumerate() {
        let name = if record.name.is_empty() || record.name == "none" {
            String::new()
        } else {
            format!("  name={}", record.name)
        };
        println!(
            "  {index:>4}  id={}  op={}  dtype={}  shape={:?}  bytes={}  children={:?}{name}",
            record.id,
            record.op_type,
            record.dtype,
            record.shape,
            record.data.len(),
            record.children,
        );
    }
}
