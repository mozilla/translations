//! `inference-rs` CLI.
//!
//! For now this is a trace inspector: it loads a reference trace (see
//! build-plan.md) and prints a summary — record count, op-type histogram, and
//! the first handful of records. It doubles as a smoke check that the reader
//! handles a real, full-size trace end to end.
//!
//! Usage:
//!   cargo run -- <trace-path> [num-records-to-print]
//!   cargo run -- inference-rs/artifacts/enfr.trace

use std::collections::BTreeMap;
use std::process::ExitCode;

use inference_rs::trace::Trace;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: inference-rs <trace-path> [num-records-to-print]");
            eprintln!("  e.g. inference-rs inference-rs/artifacts/enfr.trace");
            return ExitCode::FAILURE;
        }
    };
    let head = args.next().and_then(|n| n.parse().ok()).unwrap_or(10usize);

    let trace = match Trace::load(&path) {
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
