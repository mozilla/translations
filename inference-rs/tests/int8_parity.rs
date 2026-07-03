//! int8 GEMM unit parity.
//!
//! Validate the quantized path as one unit at the affine's float
//! output, not by matching the opaque packed weights. For each `intgemmAffine`
//! node we take the shifted activation `A` and prepared bias from the trace, the
//! logical int8 weight `B` from the model file (the trace's own `B` is in the
//! unusable packed layout), run the Rust affine, and assert it reproduces the
//! recorded float output within tolerance.
//!
//! `unquant_mult = scalar/(quantMultA·quantMultB)` is a single scalar over the
//! whole matrix; we recover it from the largest-magnitude output element and
//! then require every element to match — a strong check (one degree of freedom
//! fitting M·N outputs) that simultaneously validates the weight layout, the
//! integer GEMM, and the bias handling.
//!
//! Both the trace and the model are gitignored, so this skips when either is
//! absent. The model comes from `task inference-rs:download-model -- en fr`.
//!
//! Exercises the trace reader + comparator, which live behind `instrumentation`.
#![cfg(feature = "instrumentation")]

use std::sync::OnceLock;

use inference_rs::compare::{compare_f32, Tolerance};
use inference_rs::model::Model;
use inference_rs::ops;
use inference_rs::trace::Trace;

const TRACE_PATH: &str = "artifacts/enfr.trace";
const MODEL_PATH: &str = "../data/models/enfr/model.enfr.intgemm.alphas.bin";

fn trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            std::path::Path::new(TRACE_PATH)
                .exists()
                .then(|| Trace::load(TRACE_PATH).expect("trace parses"))
        })
        .as_ref()
}

fn model() -> Option<&'static Model> {
    static MODEL: OnceLock<Option<Model>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            std::path::Path::new(MODEL_PATH)
                .exists()
                .then(|| Model::load(MODEL_PATH).expect("model parses"))
        })
        .as_ref()
}

#[test]
fn intgemm_affine_parity() {
    let (Some(trace), Some(model)) = (trace(), model()) else {
        eprintln!("skipping int8 parity: trace or model absent");
        return;
    };
    let tol = Tolerance::default();

    let mut checked = 0usize;
    let mut skipped_derived = 0usize;
    for (index, record) in trace.records.iter().enumerate() {
        if record.op_type != "intgemmAffine" {
            continue;
        }
        let inputs = trace.inputs(index).expect("affine inputs resolve");
        assert_eq!(inputs.len(), 3, "affine is [A, B, bias]");
        let (a_rec, b_rec, bias_rec) = (inputs[0], inputs[1], inputs[2]);

        // We can only pull a logical weight from the model when B is a named
        // parameter. The final logit projection derives B via
        // intgemmSelectColumnsB (packed) and has no plain model weight; skip it.
        let weight_name = b_rec.name.rsplit("::").next().unwrap_or(&b_rec.name);
        let Some(weight) = model.get(weight_name).filter(|_| b_rec.op_type == "param") else {
            skipped_derived += 1;
            continue;
        };

        // A: shifted activation, unsigned bytes, shape [.., M, K].
        let k = *a_rec.shape.last().unwrap() as usize;
        let m = a_rec.num_elements() / k;
        let a: &[u8] = &a_rec.data;

        // B: logical int8 [N, K] (transposed weight); N derived from element count.
        let b = weight.int8_transposed().expect("int8 weight");
        assert!(
            b.len() % k == 0,
            "weight length {} not divisible by K {k}",
            b.len()
        );
        let n = b.len() / k;

        let bias = bias_rec.to_f32().expect("prepared bias is float32");
        assert_eq!(bias.len(), n, "bias length must be N");

        let expected = record.to_f32().expect("affine output is float32");

        // Recover unquant_mult from the largest-magnitude integer dot product.
        let zeros = vec![0.0f32; n];
        let int_dot = ops::intgemm_affine(a, m, k, b, n, 1.0, &zeros);
        let p = (0..int_dot.len())
            .max_by(|&x, &y| int_dot[x].abs().total_cmp(&int_dot[y].abs()))
            .expect("nonempty");
        assert!(int_dot[p] != 0.0, "degenerate affine at node {index}");
        let unquant = (expected[p] - bias[p % n]) / int_dot[p];

        let actual = ops::intgemm_affine(a, m, k, b, n, unquant, &bias);
        let cmp = compare_f32(&actual, &expected, tol).expect("same length");
        assert!(
            cmp.all_close(),
            "intgemmAffine mismatch at node {index} (weight {weight_name}): {cmp}"
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "expected intgemmAffine nodes with model weights"
    );
    eprintln!(
        "intgemmAffine parity: {checked} nodes matched within tolerance \
         ({skipped_derived} skipped: derived/packed B)"
    );
}
