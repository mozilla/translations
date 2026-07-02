//! `inference-rs` CLI: translate text, or inspect a reference trace.
//!
//! Usage:
//!   # translate (greedy). Vocab paths end in .spm: one shared, or two split.
//!   cargo run -- translate <model.bin> <vocab.spm> "Hello world."
//!   cargo run -- translate <model.bin> <src.spm> <trg.spm> "Hello world."
//!   echo "Hello world." | cargo run -- translate <model.bin> <vocab.spm>  # stdin, line by line
//!
//!   # inspect a recorded trace
//!   cargo run -- trace <trace-path> [num-records-to-print]
//!
//! Usually driven via `task inference-rs:translate -- en es --text "…"`, which
//! resolves the model + vocab from the downloaded config.

use std::collections::BTreeMap;
use std::process::ExitCode;

use inference_rs::engine::Engine;
use inference_rs::trace::Trace;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("translate") => translate(&args[1..]),
        Some("trace") => inspect_trace(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("encode") => encode(&args[1..]),
        _ => {
            eprintln!("usage:");
            eprintln!(
                "  inference-rs translate <model.bin> <src.spm> [trg.spm] [--shortlist] [text]"
            );
            eprintln!("    (no text => translate stdin line by line; shortlist off by default)");
            eprintln!("  inference-rs trace <trace-path> [num-records]");
            eprintln!("  inference-rs replay <trace-path> <model.bin>");
            eprintln!("  inference-rs encode <vocab.spm>   (stdin lines -> space-separated ids)");
            ExitCode::FAILURE
        }
    }
}

/// `encode <vocab.spm>` — tokenize each stdin line to space-separated ids (no
/// EOS). Mirrors `spm_encode --output_format=id` for tokenizer diffing.
fn encode(args: &[String]) -> ExitCode {
    let [vocab_path] = args else {
        eprintln!("usage: inference-rs encode <vocab.spm>");
        return ExitCode::FAILURE;
    };
    let vocab = match inference_rs::spm::SpmVocab::load(vocab_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to load vocab '{vocab_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    use std::io::BufRead;
    for line in std::io::stdin().lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error reading stdin: {e}");
                return ExitCode::FAILURE;
            }
        };
        let ids = vocab.encode(&line);
        let joined: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        println!("{}", joined.join(" "));
    }
    ExitCode::SUCCESS
}

/// `replay <trace> <model>` — recompute a recorded trace node-by-node and report
/// the first divergence (the parity bisector). All nodes within tolerance means
/// the ops compose correctly on that input, so any greedy mismatch is a
/// sub-tolerance near-tie rather than a node bug.
fn replay(args: &[String]) -> ExitCode {
    let [trace_path, model_path] = args else {
        eprintln!("usage: inference-rs replay <trace-path> <model.bin>");
        return ExitCode::FAILURE;
    };

    let trace = match inference_rs::trace::Trace::load(trace_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load trace '{trace_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match inference_rs::model::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load model '{model_path}': {e}");
            return ExitCode::FAILURE;
        }
    };

    let report =
        inference_rs::graph::replay(&trace, &model, inference_rs::compare::Tolerance::default());
    println!(
        "{} nodes: {} recomputed, {} matched-prefix, {} passthrough",
        report.total, report.compared, report.matched_prefix, report.passthrough
    );
    match &report.first_divergence {
        None => {
            println!(
                "no divergence — all recomputed nodes within tolerance (near-tie, not a node bug)"
            );
            ExitCode::SUCCESS
        }
        Some(d) => {
            println!(
                "first divergence at node {} (id {}, {}): {}",
                d.index, d.id, d.op_type, d.detail
            );
            ExitCode::SUCCESS
        }
    }
}

/// `translate <model> <src.spm> [trg.spm] [--shortlist] [text]` — greedy
/// translation.
///
/// The vocab paths are the positional args after the model that end in `.spm`
/// (one for a shared vocab, two for split source/target). Anything left over is
/// the text; with no text, stdin is translated line by line (model loaded once).
///
/// The lexical shortlist is **off by default** for shared-vocab pairs — it
/// restricts the output vocabulary and hurts quality on short inputs. Pass
/// `--shortlist` to enable it (the reference output-projection path). For
/// split-vocab (CJK) pairs it is required and enabled automatically. Either way
/// the `lex*.bin` sitting beside the model is found automatically.
fn translate(args: &[String]) -> ExitCode {
    const USAGE: &str =
        "usage: inference-rs translate <model.bin> <src.spm> [trg.spm] [--shortlist] [text]";

    // Split off the optional `--shortlist` flag; keep the rest positional.
    let mut use_shortlist = false;
    let mut pos: Vec<&str> = Vec::new();
    for a in args {
        if a == "--shortlist" {
            use_shortlist = true;
        } else {
            pos.push(a);
        }
    }

    let Some((model, rest)) = pos.split_first() else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

    // Collect the 1–2 leading `.spm` vocab paths; the remainder is the text.
    let n_vocab = rest
        .iter()
        .take(2)
        .take_while(|a| a.ends_with(".spm"))
        .count();
    if n_vocab == 0 {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    let src_vocab = rest[0];
    let trg_vocab = if n_vocab == 2 { rest[1] } else { rest[0] };
    let text = rest[n_vocab..].join(" ");

    // Split-vocab (CJK) models are trained to decode against the lexical
    // shortlist: without it the full-vocab argmax produces garbage, and the
    // reference engine aborts outright. So the shortlist is required there, not
    // optional — auto-enable it for split vocabs. Shared-vocab pairs keep it
    // off by default (production quality), opt-in via `--shortlist`.
    let split_vocab = src_vocab != trg_vocab;
    if split_vocab {
        use_shortlist = true;
    }

    let mut engine = match Engine::load(model, src_vocab, trg_vocab) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load engine: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Attach the lexical shortlist only when requested, auto-finding it beside
    // the model.
    if use_shortlist {
        match find_shortlist(model) {
            Some(path) => match inference_rs::shortlist::Shortlist::load(&path) {
                Ok(sl) => {
                    eprintln!("[shortlist] {path}");
                    engine = engine.with_shortlist(sl);
                }
                Err(e) => {
                    eprintln!("failed to load shortlist {path}: {e}");
                    return ExitCode::FAILURE;
                }
            },
            None => eprintln!("[shortlist] none found beside {model}; translating without one"),
        }
    }

    if text.trim().is_empty() {
        // Stream stdin: one translation per input line.
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) if !line.trim().is_empty() => println!("{}", engine.translate(&line)),
                Ok(_) => println!(),
                Err(e) => {
                    eprintln!("error reading stdin: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        println!("{}", engine.translate(&text));
    }
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
