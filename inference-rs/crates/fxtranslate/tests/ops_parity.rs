//! Op-level parity harness.
//!
//! For each node of a given op type in a recorded reference trace, pull its
//! exact input tensors and its recorded output, run the Rust op on those inputs,
//! and assert the result matches the oracle within tolerance. No graph, no
//! ordering — pure functions against golden fixtures.
//!
//! The trace is large and gitignored, so these tests skip (rather than fail)
//! when it is absent. Record one with:
//!
//!   task rs:translate-reference -- en fr --text "Hello world." \
//!       --cpu-threads 1 --trace
//!
//! which writes inference-rs/artifacts/enfr.trace.
//!
//! Exercises the trace reader + comparator, which live behind `instrumentation`.
#![cfg(feature = "instrumentation")]

use std::sync::OnceLock;

use fxtranslate::compare::{compare_f32, Tolerance};
use fxtranslate::ops;
use fxtranslate::trace::{DType, Trace, TraceRecord};

const TRACE_PATH: &str = "../../artifacts/enfr.trace";

/// Load the real trace once, shared across all tests. Returns `None` (with a
/// skip note) when it is absent, so the suite still passes on a fresh checkout.
fn trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            if !std::path::Path::new(TRACE_PATH).exists() {
                eprintln!("skipping: {TRACE_PATH} not found (record one with task rs:translate-reference -- ... --trace)");
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

        // Expose the recorded output to attribute-recovery helpers.
        CURRENT_OUTPUT.with(|c| *c.borrow_mut() = (expected.clone(), record.shape.clone()));
        let actual = run(trace, index, &inputs);

        let cmp = compare_f32(&actual, &expected, tol).expect("same length");
        assert!(
            cmp.all_close(),
            "{op_type} mismatch at node index {index} (id {}): {cmp}",
            record.id
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "expected {op_type} nodes in the trace but found none"
    );
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
    parity(
        "layer_normalization",
        Tolerance::default(),
        |_, index, inputs| {
            assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "layer_normalization node {index} should have [x, gamma] or [x, gamma, beta], got {}",
            inputs.len()
        );
            let (rows, cols, x) = rows_cols(inputs[0]);
            let gamma = f32_input(inputs[1]);
            let beta = inputs.get(2).map(|b| f32_input(b));
            ops::layer_normalization(&x, &gamma, beta.as_deref(), rows, cols, EPS)
        },
    );
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
        assert_eq!(
            inputs.len(),
            3,
            "highway node {index} should have [in1, in2, t]"
        );
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
        assert_eq!(
            inputs.len(),
            1,
            "softmax is unary here (mask already folded in)"
        );
        let (rows, cols, x) = rows_cols(inputs[0]);
        ops::softmax(&x, rows, cols)
    });
}

// --- Structural ops -------------------------------------------------------------
//
// These carry their scalar / axes as node attributes that the trace does not
// record. We recover the attribute from the traced input/output data itself,
// then run the parameterized op — so the test still proves the Rust op
// reproduces the reference, and would catch a wrong op even though the
// attribute is inferred.

#[test]
fn reshape_parity() {
    // reshape is a metadata-only view: data order is unchanged.
    parity("reshape", Tolerance::default(), |_, _, inputs| {
        assert_eq!(inputs.len(), 1, "reshape is unary");
        let record = inputs[0];
        ops::reshape(&f32_input(record), &record.shape)
    });
}

#[test]
fn scalar_mult_parity() {
    parity("scalar_mult", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 1, "scalar_mult is unary");
        let x = f32_input(inputs[0]);
        // Recover the multiplier from the first element with a nonzero input.
        let scalar = recover_ratio(&x, index)
            .expect("scalar_mult needs a nonzero input element to recover the scalar");
        ops::scalar_mult(&x, scalar)
    });
}

#[test]
fn scalar_add_parity() {
    // The addend is recovered per node inside the harness via the output; run
    // with a placeholder and let `parity` supply the recovered value.
    let Some(trace) = trace() else { return };
    let mut checked = 0usize;
    for (index, record) in trace.records.iter().enumerate() {
        if record.op_type != "scalar_add" {
            continue;
        }
        let inputs = trace.inputs(index).expect("inputs resolve");
        assert_eq!(inputs.len(), 1, "scalar_add is unary");
        let x = f32_input(inputs[0]);
        let expected = record.to_f32().expect("float32 output");
        // Recover the addend directly: out - in at the first element.
        let scalar = expected[0] - x[0];
        let actual = ops::scalar_add(&x, scalar);
        let cmp = compare_f32(&actual, &expected, Tolerance::default()).expect("same length");
        assert!(
            cmp.all_close(),
            "scalar_add mismatch at index {index}: {cmp}"
        );
        checked += 1;
    }
    assert!(checked > 0, "expected scalar_add nodes but found none");
    eprintln!("scalar_add parity: {checked} nodes matched within tolerance");
}

#[test]
fn transpose_parity() {
    parity("transpose", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 1, "transpose is unary");
        let in_record = inputs[0];
        let x = f32_input(in_record);
        let expected = current_output(index); // recorded output for shape context
                                              // Search the permutations of the axes consistent with in->out shapes for
                                              // the one that reproduces the recorded output.
        let perm = recover_transpose_perm(&x, &in_record.shape, &expected.0, &expected.1)
            .unwrap_or_else(|| panic!("no transpose perm reproduces node {index}"));
        let (out, _) = ops::transpose(&x, &in_record.shape, &perm);
        out
    });
}

#[test]
fn slice_view_parity() {
    parity("sliceView", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 1, "sliceView is unary");
        let x = f32_input(inputs[0]);
        let (expected, _) = current_output(index);
        // sliceView is memory-consecutive: find the contiguous offset whose block
        // equals the output.
        let offset = recover_slice_offset(&x, &expected)
            .unwrap_or_else(|| panic!("no contiguous slice reproduces node {index}"));
        ops::slice_contiguous(&x, offset, expected.len())
    });
}

// --- Gather & batched matmul ----------------------------------------------------

#[test]
fn rows_parity() {
    parity("rows", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 2, "rows node {index} is [data, indices]");
        let data_rec = inputs[0];
        assert_eq!(data_rec.shape.len(), 2, "rows data must be 2-D");
        let data = f32_input(data_rec);
        let (num_rows, width) = (data_rec.shape[0] as usize, data_rec.shape[1] as usize);
        let indices = inputs[1].to_u32().expect("rows indices are uint32");
        ops::rows(&data, num_rows, width, &indices)
    });
}

#[test]
fn cols_parity() {
    parity("cols", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 2, "cols node {index} is [data, indices]");
        let data_rec = inputs[0];
        assert_eq!(data_rec.shape.len(), 2, "cols data must be 2-D");
        let data = f32_input(data_rec);
        let (num_rows, width) = (data_rec.shape[0] as usize, data_rec.shape[1] as usize);
        let indices = inputs[1].to_u32().expect("cols indices are uint32");
        ops::cols(&data, num_rows, width, &indices)
    });
}

#[test]
fn bdot_parity() {
    parity("bdot", Tolerance::default(), |_, index, inputs| {
        assert_eq!(inputs.len(), 2, "bdot node {index} is [a, b]");
        let a = f32_input(inputs[0]);
        let b = f32_input(inputs[1]);
        let (expected, out_shape) = current_output(index);
        recover_bdot(
            &a,
            &inputs[0].shape,
            &b,
            &inputs[1].shape,
            &expected,
            &out_shape,
        )
        .unwrap_or_else(|| panic!("no bdot (transA,transB,scale) reproduces node {index}"))
    });
}

/// bdot carries transA/transB/scale as node attributes. Try the four transpose
/// combinations whose dims are consistent and whose output shape matches;
/// recover `scale` from the largest-magnitude output element, then accept the
/// combination that reproduces the whole tensor.
fn recover_bdot(
    a: &[f32],
    a_shape: &[i32],
    b: &[f32],
    b_shape: &[i32],
    expected: &[f32],
    out_shape: &[i32],
) -> Option<Vec<f32>> {
    let want_mn = &out_shape[out_shape.len().saturating_sub(2)..];
    for &transa in &[false, true] {
        for &transb in &[false, true] {
            let (ra, ca) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
            let (rb, cb) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);
            let (m, k) = if transa { (ca, ra) } else { (ra, ca) };
            let (kb, n) = if transb { (cb, rb) } else { (rb, cb) };
            if k != kb || want_mn != [m, n] {
                continue;
            }
            let (unscaled, _) = ops::bdot(a, a_shape, transa, b, b_shape, transb, 1.0);
            // Recover scale at the largest-magnitude element.
            let argmax = (0..unscaled.len())
                .max_by(|&x, &y| unscaled[x].abs().total_cmp(&unscaled[y].abs()))?;
            if unscaled[argmax] == 0.0 {
                continue;
            }
            let scale = expected[argmax] / unscaled[argmax];
            let scaled: Vec<f32> = unscaled.iter().map(|v| v * scale).collect();
            if scaled
                .iter()
                .zip(expected)
                .all(|(x, e)| (x - e).abs() <= 1e-5 + 1e-3 * e.abs())
            {
                return Some(scaled);
            }
        }
    }
    None
}

// --- attribute-recovery helpers ----------------------------------------------

thread_local! {
    // Lets the closures above reach the recorded output for shape context
    // without changing the `parity` signature. Set per node before `run`.
    static CURRENT_OUTPUT: std::cell::RefCell<(Vec<f32>, Vec<i32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// The recorded output (data, shape) of the node currently under test.
fn current_output(_index: usize) -> (Vec<f32>, Vec<i32>) {
    CURRENT_OUTPUT.with(|c| c.borrow().clone())
}

/// Recover a uniform multiplier `out/in` from the element with the largest
/// input magnitude (the most numerically stable ratio). If every input is
/// exactly zero the output is zero too and any scalar reproduces it, so we
/// return 1.0.
fn recover_ratio(input: &[f32], index: usize) -> Option<f32> {
    let (out, _) = current_output(index);
    let argmax = (0..input.len()).max_by(|&a, &b| input[a].abs().total_cmp(&input[b].abs()))?;
    if input[argmax] == 0.0 {
        return Some(1.0);
    }
    Some(out[argmax] / input[argmax])
}

/// Find the axis permutation that turns `in_shape` data into the recorded
/// output, among permutations whose resulting shape matches `out_shape`.
fn recover_transpose_perm(
    input: &[f32],
    in_shape: &[i32],
    expected: &[f32],
    out_shape: &[i32],
) -> Option<Vec<usize>> {
    let rank = in_shape.len();
    for perm in permutations(rank) {
        let candidate_shape: Vec<i32> = perm.iter().map(|&p| in_shape[p]).collect();
        if &candidate_shape != out_shape {
            continue;
        }
        let (out, _) = ops::transpose(input, in_shape, &perm);
        if out
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-6 + 1e-4 * b.abs())
        {
            return Some(perm);
        }
    }
    None
}

/// Find the contiguous offset where `input[offset..offset+out.len()]` equals the
/// recorded slice output.
fn recover_slice_offset(input: &[f32], expected: &[f32]) -> Option<usize> {
    let len = expected.len();
    (0..=input.len().saturating_sub(len)).find(|&off| {
        input[off..off + len]
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-6 + 1e-4 * b.abs())
    })
}

/// All permutations of `0..rank` (rank is small, ≤ 4 in these traces).
fn permutations(rank: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current: Vec<usize> = (0..rank).collect();
    permute(&mut current, 0, &mut result);
    result
}

fn permute(arr: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == arr.len() {
        out.push(arr.clone());
        return;
    }
    for i in k..arr.len() {
        arr.swap(k, i);
        permute(arr, k + 1, out);
        arr.swap(k, i);
    }
}
