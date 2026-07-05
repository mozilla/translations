//! `fxtranslate-oracle`: the engine's raw diagnostic binary — translate/encode
//! text, or inspect and replay a reference trace against the marian oracle.
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
//! Usually driven via `task rs:translate -- en es --text "…"`, which
//! resolves the model + vocab from the downloaded config.

use std::collections::BTreeMap;
use std::process::ExitCode;

use fxtranslate::engine::Engine;
use fxtranslate::trace::Trace;

// Under `--features dhat-heap`, route every allocation through dhat's allocator
// so the profiler can attribute the heap. No effect on default builds.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// Under `--features jemalloc`, use jemalloc — a page-returning allocator — to test
// whether settled RSS is gated by the system allocator's page retention rather
// than by our live footprint (issues/19-settled-rss-allocator.md). Configure purge
// via the `_RJEM_MALLOC_CONF` env var, e.g. `dirty_decay_ms:0,muzzy_decay_ms:0`.
// Exclusive with `dhat-heap` (only one #[global_allocator] may exist).
#[cfg(all(feature = "jemalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> ExitCode {
    // The profiler must outlive the whole run: it writes its JSON on drop, at
    // the end of `main`. Output path comes from `DHAT_OUT` (translate.py points
    // it at inference-rs/artifacts/), defaulting to dhat's own `dhat-heap.json`.
    #[cfg(feature = "dhat-heap")]
    let _dhat = {
        let out = std::env::var("DHAT_OUT").unwrap_or_else(|_| "dhat-heap.json".into());
        eprintln!("[dhat] heap profiling on; report -> {out}");
        dhat::Profiler::builder().file_name(out).build()
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("translate") => translate(&args[1..]),
        Some("trace") => inspect_trace(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("encode") => encode(&args[1..]),
        _ => {
            eprintln!("usage:");
            eprintln!(
                "  fxtranslate-oracle translate <model.bin> <src.spm> [trg.spm] [--shortlist] [text]"
            );
            eprintln!("    (no text => translate stdin line by line; shortlist off by default)");
            eprintln!("  fxtranslate-oracle trace <trace-path> [num-records]");
            eprintln!("  fxtranslate-oracle replay <trace-path> <model.bin>");
            eprintln!("  fxtranslate-oracle encode <vocab.spm>   (stdin lines -> space-separated ids)");
            ExitCode::FAILURE
        }
    }
}

/// `encode <vocab.spm>` — tokenize each stdin line to space-separated ids (no
/// EOS). Mirrors `spm_encode --output_format=id` for tokenizer diffing.
fn encode(args: &[String]) -> ExitCode {
    let [vocab_path] = args else {
        eprintln!("usage: fxtranslate-oracle encode <vocab.spm>");
        return ExitCode::FAILURE;
    };
    let vocab = match fxtranslate::spm::SpmVocab::load(vocab_path) {
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
        eprintln!("usage: fxtranslate-oracle replay <trace-path> <model.bin>");
        return ExitCode::FAILURE;
    };

    let trace = match fxtranslate::trace::Trace::load(trace_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load trace '{trace_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match fxtranslate::model::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load model '{model_path}': {e}");
            return ExitCode::FAILURE;
        }
    };

    let report =
        fxtranslate_oracle::graph::replay(&trace, &model, fxtranslate_oracle::compare::Tolerance::default());
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
/// The lexical shortlist is **off by default on every path** — it restricts the
/// output vocabulary and hurts quality on short inputs, and is off in production.
/// Pass `--shortlist` to opt in (the reference output-projection path); the
/// `lex*.bin` sitting beside the model is then found automatically. Split-vocab
/// (CJK) models produce good output only with it, but enabling it stays the
/// caller's explicit choice.
fn translate(args: &[String]) -> ExitCode {
    const USAGE: &str = "usage: fxtranslate-oracle translate <model.bin> <src.spm> [trg.spm] \
         [--shortlist] [--timing] [--blocks <file>] [--mmap] [text]";

    // Split off the optional flags; keep the rest positional.
    let mut use_shortlist = false;
    let mut timing = false;
    let mut mmap = false;
    let mut blocks: Option<&str> = None;
    let mut pos: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--shortlist" => use_shortlist = true,
            "--timing" => timing = true,
            "--mmap" => mmap = true,
            "--blocks" => {
                i += 1;
                match args.get(i) {
                    Some(p) => blocks = Some(p),
                    None => {
                        eprintln!("--blocks needs a file path\n{USAGE}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            a => pos.push(a),
        }
        i += 1;
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

    let loaded = if mmap {
        Engine::load_mmapped(model, src_vocab, trg_vocab)
    } else {
        Engine::load(model, src_vocab, trg_vocab)
    };
    let mut engine = match loaded {
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
            Some(path) => match fxtranslate::shortlist::Shortlist::load(&path) {
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

    // Block mode: read a blank-line-delimited block file (one sentence per line,
    // empty line between blocks) and batch-translate each block, matching the
    // production block unit. Translations mirror the input layout; `--timing`
    // emits one JSON span per block on stderr for the perf harness.
    if let Some(path) = blocks {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to read blocks file {path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        for (bi, chunk) in content.split("\n\n").enumerate() {
            let block: Vec<&str> = chunk.lines().filter(|l| !l.trim().is_empty()).collect();
            if block.is_empty() {
                continue;
            }
            let (outs, t) = engine.translate_batch_timed(&block);
            for o in &outs {
                println!("{o}");
            }
            println!(); // blank line between blocks, mirroring the input
            if timing {
                eprintln!(
                    "[block] {{\"block\":{bi},\"sentences\":{},\"src_tokens\":{},\"tokens\":{},\
                     \"encode_ms\":{:.4},\"ttft_ms\":{:.4},\"decode_ms\":{:.4}}}",
                    t.sentences,
                    t.src_tokens,
                    t.tokens,
                    t.encode_ms,
                    t.encode_ms + t.first_token_ms,
                    t.decode_ms
                );
            }
        }
        #[cfg(feature = "gemmology")]
        if timing {
            eprintln!(
                "[gemmology] {{\"prepared_bytes\":{}}}",
                fxtranslate::gemm::prepared_bytes()
            );
        }
        return ExitCode::SUCCESS;
    }

    // Snapshot the live heap once the model is loaded and idle — this is the
    // "retained" footprint a long-lived translation server carries between
    // sentences, distinct from the process-lifetime peak (dhat's t-gmax).
    #[cfg(feature = "dhat-heap")]
    heap_snapshot("after load (retained)");

    // Translate one line: print the result on stdout, and with `--timing` emit a
    // per-sentence timing span (JSON) on stderr for the perf harness to aggregate.
    let mut idx = 0usize;
    let mut run = |line: &str| {
        if timing {
            let (out, t) = engine.translate_timed(line);
            println!("{out}");
            eprintln!(
                "[timing] {{\"idx\":{idx},\"encode_ms\":{:.4},\"ttft_ms\":{:.4},\
                 \"decode_ms\":{:.4},\"tokens\":{}}}",
                t.encode_ms,
                t.encode_ms + t.first_token_ms,
                t.decode_ms,
                t.out_tokens
            );
        } else {
            println!("{}", engine.translate(line));
        }
        idx += 1;
    };

    if text.trim().is_empty() {
        // Stream stdin: one translation per input line.
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) if !line.trim().is_empty() => run(&line),
                Ok(_) => println!(),
                Err(e) => {
                    eprintln!("error reading stdin: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        run(&text);
    }

    // Snapshot again after translating: if this matches the post-load figure, the
    // per-sentence work leaves nothing behind (steady-state retained == loaded).
    #[cfg(feature = "dhat-heap")]
    heap_snapshot("after translate (retained)");

    ExitCode::SUCCESS
}

/// Print the current live heap and the running max (dhat feature only).
#[cfg(feature = "dhat-heap")]
fn heap_snapshot(label: &str) {
    let s = dhat::HeapStats::get();
    eprintln!(
        "[dhat] {label}: live {} bytes in {} blocks (max so far {} bytes)",
        s.curr_bytes, s.curr_blocks, s.max_bytes
    );
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
        eprintln!("usage: fxtranslate-oracle trace <trace-path> [num-records]");
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
