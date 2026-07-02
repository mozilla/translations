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

use std::sync::OnceLock;

use inference_rs::compare::{compare_f32, Tolerance};
use inference_rs::ops;
use inference_rs::trace::{DType, Trace, TraceRecord};

const TRACE_PATH: &str = "artifacts/enfr.trace";

/// Load the real trace once, shared across all tests. Returns `None` (with a
/// skip note) when it is absent, so the suite still passes on a fresh checkout.
fn trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            if !std::path::Path::new(TRACE_PATH).exists() {
                eprintln!("skipping: {TRACE_PATH} not found (record one with task inference-rs:translate-reference -- ... --trace)");
                None
            } else {
                Some(Trace::load(TRACE_PATH).expect("real trace should parse"))
            }
        })
        .as_ref()
}

/// A float32 record decoded to `(rows, cols, values)`, where `cols` is the last
/// dimension and `rows` is everything else flattened.
fn rows_cols(record: &TraceRecord) -> (usize, usize, Vec<f32>) {
    let cols = *record.shape.last().expect("record has a shape") as usize;
    let rows = record.num_elements() / cols;
    (rows, cols, record.to_f32().expect("float32 record"))
}

/// Drive an op over every node of `op_type` in the trace: build the Rust op's
/// output from the node's traced inputs via `run`, then assert it matches the
/// recorded output within `tol`. Fails loudly if the op type is absent, so a
/// broken selector can't masquerade as a pass.
fn parity<F>(op_type: &str, tol: Tolerance, mut run: F)
where
    F: FnMut(&Trace, usize, &[&TraceRecord]) -> Vec<f32>,
{
    let Some(trace) = trace() else { return };

    let mut checked = 0usize;
    for (index, record) in trace.records.iter().enumerate() {
        if record.op_type != op_type {
            continue;
        }
        let inputs = trace.inputs(index).expect("inputs resolve");
        let expected = record.to_f32().expect("float32 output");

        let actual = run(trace, index, &inputs);

        let cmp = compare_f32(&actual, &expected, tol).expect("same length");
        assert!(
            cmp.all_close(),
            "{op_type} mismatch at node index {index} (id {}): {cmp}",
            record.id
        );
        checked += 1;
    }

    assert!(checked > 0, "expected {op_type} nodes in the trace but found none");
    eprintln!("{op_type} parity: {checked} nodes matched within tolerance");
}

/// Decode a float32 input, asserting the dtype so a wrong-dtype fixture is a
/// clear failure rather than a silent misread.
fn f32_input(record: &TraceRecord) -> Vec<f32> {
    assert_eq!(record.dtype, DType::Float32, "expected float32 input");
    record.to_f32().expect("float32 input")
}

#[test]
fn layer_normalization_parity() {
    // Transformer layer norm epsilon (layers/generic.h:463).
    const EPS: f32 = 1e-6;
    parity("layer_normalization", Tolerance::default(), |_, index, inputs| {
        assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "layer_normalization node {index} should have [x, gamma] or [x, gamma, beta], got {}",
            inputs.len()
        );
        let (rows, cols, x) = rows_cols(inputs[0]);
        let gamma = f32_input(inputs[1]);
        let beta = inputs.get(2).map(|b| f32_input(b));
        ops::layer_normalization(&x, &gamma, beta.as_deref(), rows, cols, EPS)
    });
}

#[test]
fn relu_parity() {
    parity("ReLU", Tolerance::default(), |_, _, inputs| {
        assert_eq!(inputs.len(), 1, "ReLU is unary");
        ops::relu(&f32_input(inputs[0]))
    });
}

#[test]
fn negate_parity() {
    parity("negate", Tolerance::default(), |_, _, inputs| {
        assert_eq!(inputs.len(), 1, "negate is unary");
        ops::negate(&f32_input(inputs[0]))
    });
}

#[test]
fn add_parity() {
    parity("+", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 2, "+ node {index} should be binary");
        let a = f32_input(inputs[0]);
        let b = f32_input(inputs[1]);
        let (out, _) = ops::add(&a, &inputs[0].shape, &b, &inputs[1].shape);
        out
    });
}

#[test]
fn highway_parity() {
    parity("highway", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 3, "highway node {index} should have [in1, in2, t]");
        // marian HighwayForward: out = σ(child2)·child0 + (1 − σ(child2))·child1.
        let in1 = f32_input(inputs[0]);
        let in2 = f32_input(inputs[1]);
        let t = f32_input(inputs[2]);
        ops::highway(&in1, &in2, &t)
    });
}

#[test]
fn softmax_parity() {
    parity("softmax", Tolerance::default(), |_, _, inputs| {
        assert_eq!(inputs.len(), 1, "softmax is unary here (mask already folded in)");
        let (rows, cols, x) = rows_cols(inputs[0]);
        ops::softmax(&x, rows, cols)
    });
}
