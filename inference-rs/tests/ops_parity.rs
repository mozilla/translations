//! Op-level parity harness (build-plan.md, step 3).
//!
//! This is the validation loop the whole strategy rests on: for each node of a
//! given op type in a recorded reference trace, pull its exact input tensors
//! and its recorded output, run the Rust op on those inputs, and assert the
//! result matches the oracle within tolerance. No graph, no ordering — pure
//! functions against golden fixtures.
//!
//! The trace is large and gitignored, so these tests skip (rather than fail)
//! when it is absent. Record one with:
//!
//!   task inference-rs:translate-reference -- en fr --text "Hello world." \
//!       --cpu-threads 1 --trace
//!
//! which writes inference-rs/artifacts/enfr.trace.

use inference_rs::compare::{compare_f32, Tolerance};
use inference_rs::ops;
use inference_rs::trace::{DType, Trace, TraceRecord};

const TRACE_PATH: &str = "artifacts/enfr.trace";

/// Load the real trace, or return `None` (and print a skip note) when it is
/// absent so the suite still passes on a fresh checkout.
fn load_trace() -> Option<Trace> {
    if !std::path::Path::new(TRACE_PATH).exists() {
        eprintln!("skipping: {TRACE_PATH} not found (record one with task inference-rs:translate-reference -- ... --trace)");
        return None;
    }
    Some(Trace::load(TRACE_PATH).expect("real trace should parse"))
}

/// A float32 record decoded to `(rows, cols, values)`, where `cols` is the last
/// (normalized) dimension and `rows` is everything else flattened.
fn rows_cols(record: &TraceRecord) -> (usize, usize, Vec<f32>) {
    let cols = *record.shape.last().expect("record has a shape") as usize;
    let rows = record.num_elements() / cols;
    (rows, cols, record.to_f32().expect("float32 record"))
}

#[test]
fn layer_normalization_parity() {
    let Some(trace) = load_trace() else { return };
    // Transformer layer norm epsilon (layers/generic.h:463).
    const EPS: f32 = 1e-6;
    let tol = Tolerance::default();

    let mut checked = 0usize;
    for (index, record) in trace.records.iter().enumerate() {
        if record.op_type != "layer_normalization" {
            continue;
        }

        let inputs = trace.inputs(index).expect("layernorm inputs resolve");
        assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "layer_normalization node {index} should have [x, gamma] or [x, gamma, beta], got {}",
            inputs.len()
        );

        let (rows, cols, x) = rows_cols(inputs[0]);
        let gamma = inputs[1].to_f32().expect("gamma is float32");
        let beta = inputs.get(2).map(|b| b.to_f32().expect("beta is float32"));

        assert_eq!(inputs[0].dtype, DType::Float32, "layernorm input is float32");
        let expected = record.to_f32().expect("layernorm output is float32");

        let actual =
            ops::layer_normalization(&x, &gamma, beta.as_deref(), rows, cols, EPS);

        let cmp = compare_f32(&actual, &expected, tol).expect("same length");
        assert!(
            cmp.all_close(),
            "layer_normalization mismatch at node index {index} (id {}): {cmp}",
            record.id
        );
        checked += 1;
    }

    // Guard against a silently-passing harness: the enfr trace is known to
    // contain many layernorm nodes, so zero checks means the loop is broken.
    assert!(
        checked > 0,
        "expected layer_normalization nodes in the trace but found none"
    );
    eprintln!("layer_normalization parity: {checked} nodes matched within tolerance");
}
