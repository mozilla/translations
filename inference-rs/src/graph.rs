//! Graph-level replay (build-plan.md step 5).
//!
//! Once the ops are trusted individually, the remaining question is whether they
//! compose in execution order. This module walks the reference trace front to
//! back and *recomputes* every node from its children's **already-recomputed**
//! values — a genuine forward pass — then compares each result against the
//! recorded oracle to find the first point of divergence.
//!
//! Because it recomputes from recomputed inputs (not from the trace's clean
//! per-node inputs), any error compounds forward, so the first mismatch pinpoints
//! the earliest op whose composition breaks. Static, weight-derived scalars that
//! the trace records but the plan does not recompute — the precomputed alpha
//! (`alphaNodeOp`), `intgemmQuantMultB`, and the shift-corrected `prepareBias` —
//! are taken from the trace as trusted constants (they are functions of the
//! weights, not of the dynamic activations). The activation path itself,
//! including `intgemmPrepareA` quantization and the int8 affine, is fully
//! recomputed.
//!
//! Op *attributes* the trace doesn't store (transpose axes, scalar factors,
//! bdot's transpose/scale, the affine's unquantize multiplier) are recovered
//! from each node's **clean trace** input/output — a stand-in for having built
//! the graph ourselves — then applied to the recomputed inputs, so drift still
//! surfaces.

use crate::compare::{compare_f32, Tolerance};
use crate::model::Model;
use crate::ops;
use crate::trace::{DType, Trace, TraceRecord};

/// A recomputed node value. Most nodes are float; `intgemmPrepareA` yields
/// shifted `u8`; gather indices are `u32`. `Unavailable` marks values we cannot
/// recompute (packed int8 weights and the packed `SelectColumnsB`/`PrepareB`
/// outputs), which are only ever consumed through the model or as passthroughs.
#[derive(Clone, Debug)]
enum Value {
    F32(Vec<f32>),
    U8(Vec<u8>),
    U32(Vec<u32>),
    Unavailable,
}

impl Value {
    fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Value::F32(v) => Some(v),
            _ => None,
        }
    }
}

/// The earliest node whose recomputed value left tolerance.
#[derive(Clone, Debug)]
pub struct Divergence {
    pub index: usize,
    pub id: u64,
    pub op_type: String,
    pub detail: String,
}

/// Outcome of a replay.
#[derive(Clone, Debug)]
pub struct ReplayReport {
    /// Total nodes recomputed and compared against the trace.
    pub compared: usize,
    /// Recomputed nodes that matched, in a row, before the first divergence.
    pub matched_prefix: usize,
    /// Nodes taken from the trace as trusted constants (quant scalars, bias) or
    /// left unavailable (packed weights) and not compared.
    pub passthrough: usize,
    /// The first divergence, if any.
    pub first_divergence: Option<Divergence>,
    /// Total node count in the trace.
    pub total: usize,
}

impl ReplayReport {
    /// Whether the whole graph replayed with no divergence.
    pub fn fully_matched(&self) -> bool {
        self.first_divergence.is_none()
    }
}

/// Replay the trace as a forward pass, recomputing each node from its children's
/// recomputed values, and return where (if anywhere) it first diverges.
pub fn replay(trace: &Trace, model: &Model, tol: Tolerance) -> ReplayReport {
    // Transformer layer-norm epsilon (layers/generic.h:463).
    const EPS: f32 = 1e-6;
    // int8 quantization is exact up to ±1 in the last place from rounding.
    let u8_tol = Tolerance::new(0.0, 1.0);

    let mut values: Vec<Value> = Vec::with_capacity(trace.records.len());
    let mut report = ReplayReport {
        compared: 0,
        matched_prefix: 0,
        passthrough: 0,
        first_divergence: None,
        total: trace.records.len(),
    };
    let mut prefix_open = true; // still counting the matched run before divergence

    for index in 0..trace.records.len() {
        let record = &trace.records[index];
        let child_idx = resolve_children(trace, index);

        // Compute this node's value and whether it was recomputed (vs trusted).
        let (value, compared): (Value, Option<CompareKind>) =
            compute_node(trace, model, record, &child_idx, &values, EPS);

        // Compare recomputed nodes against the oracle.
        match compared {
            Some(kind) => {
                let ok = compare_node(record, &value, kind, tol, u8_tol);
                report.compared += 1;
                match ok {
                    Ok(()) => {
                        if prefix_open {
                            report.matched_prefix += 1;
                        }
                    }
                    Err(detail) => {
                        if report.first_divergence.is_none() {
                            report.first_divergence = Some(Divergence {
                                index,
                                id: record.id,
                                op_type: record.op_type.clone(),
                                detail,
                            });
                        }
                        prefix_open = false;
                    }
                }
            }
            None => report.passthrough += 1,
        }

        values.push(value);
    }

    report
}

/// Whether a recomputed node is compared as floats or as shifted bytes.
#[derive(Clone, Copy)]
enum CompareKind {
    Float,
    U8,
}

fn compare_node(
    record: &TraceRecord,
    value: &Value,
    kind: CompareKind,
    tol: Tolerance,
    u8_tol: Tolerance,
) -> Result<(), String> {
    match (kind, value) {
        (CompareKind::Float, Value::F32(actual)) => {
            let expected = record.to_f32().map_err(|e| e.to_string())?;
            let cmp = compare_f32(actual, &expected, tol).map_err(|e| e.to_string())?;
            if cmp.all_close() {
                Ok(())
            } else {
                Err(cmp.to_string())
            }
        }
        (CompareKind::U8, Value::U8(actual)) => {
            let expected: Vec<f32> = record.data.iter().map(|&b| b as f32).collect();
            let actual_f: Vec<f32> = actual.iter().map(|&b| b as f32).collect();
            let cmp = compare_f32(&actual_f, &expected, u8_tol).map_err(|e| e.to_string())?;
            if cmp.all_close() {
                Ok(())
            } else {
                Err(cmp.to_string())
            }
        }
        _ => Err("value kind did not match comparison kind".to_string()),
    }
}

/// Recompute one node. Returns its value and, when the node is recomputed rather
/// than trusted, how it should be compared.
fn compute_node(
    trace: &Trace,
    model: &Model,
    record: &TraceRecord,
    child_idx: &[usize],
    values: &[Value],
    eps: f32,
) -> (Value, Option<CompareKind>) {
    let f32_child = |i: usize| -> Vec<f32> {
        values[child_idx[i]]
            .as_f32()
            .map(|v| v.to_vec())
            .unwrap_or_default()
    };
    let child_rec = |i: usize| &trace.records[child_idx[i]];

    match record.op_type.as_str() {
        // --- leaves --------------------------------------------------------
        "param" | "const" => {
            let v = match record.dtype {
                DType::Float32 => record.to_f32().map(Value::F32).unwrap_or(Value::Unavailable),
                DType::UInt32 => record.to_u32().map(Value::U32).unwrap_or(Value::Unavailable),
                _ => Value::Unavailable, // packed int8 weight; fetched from the model
            };
            (v, None)
        }

        // --- trusted static quant scalars / bias ---------------------------
        "alphaNodeOp" | "intgemmQuantMultB" | "prepareBias" | "prepareFakeBias" => {
            let v = record.to_f32().map(Value::F32).unwrap_or(Value::Unavailable);
            (v, None)
        }
        // Packed layouts we cannot recompute; consumers passthrough too.
        "intgemmSelectColumnsB" | "intgemmPrepareB" => (Value::Unavailable, None),

        // --- elementwise / reduction --------------------------------------
        "ReLU" => (Value::F32(ops::relu(&f32_child(0))), Some(CompareKind::Float)),
        "negate" => (Value::F32(ops::negate(&f32_child(0))), Some(CompareKind::Float)),
        "+" => {
            let (out, _) = ops::add(
                &f32_child(0),
                &child_rec(0).shape,
                &f32_child(1),
                &child_rec(1).shape,
            );
            (Value::F32(out), Some(CompareKind::Float))
        }
        "highway" => (
            Value::F32(ops::highway(&f32_child(0), &f32_child(1), &f32_child(2))),
            Some(CompareKind::Float),
        ),
        "softmax" => {
            let (rows, cols) = rows_cols(child_rec(0));
            (Value::F32(ops::softmax(&f32_child(0), rows, cols)), Some(CompareKind::Float))
        }
        "layer_normalization" => {
            let (rows, cols) = rows_cols(child_rec(0));
            let gamma = f32_child(1);
            let beta = (child_idx.len() > 2).then(|| f32_child(2));
            let out = ops::layer_normalization(&f32_child(0), &gamma, beta.as_deref(), rows, cols, eps);
            (Value::F32(out), Some(CompareKind::Float))
        }

        // --- structural (attribute recovered from clean trace) ------------
        "reshape" => (Value::F32(f32_child(0)), Some(CompareKind::Float)),
        "scalar_mult" => {
            let s = recover_scalar_mult(child_rec(0), record);
            (Value::F32(ops::scalar_mult(&f32_child(0), s)), Some(CompareKind::Float))
        }
        "scalar_add" => {
            let s = recover_scalar_add(child_rec(0), record);
            (Value::F32(ops::scalar_add(&f32_child(0), s)), Some(CompareKind::Float))
        }
        "transpose" => match recover_transpose_perm(child_rec(0), record) {
            Some(perm) => {
                let (out, _) = ops::transpose(&f32_child(0), &child_rec(0).shape, &perm);
                (Value::F32(out), Some(CompareKind::Float))
            }
            None => (Value::Unavailable, Some(CompareKind::Float)),
        },
        "sliceView" => match recover_slice_offset(child_rec(0), record) {
            Some(off) => (
                Value::F32(ops::slice_contiguous(&f32_child(0), off, record.num_elements())),
                Some(CompareKind::Float),
            ),
            None => (Value::Unavailable, Some(CompareKind::Float)),
        },

        // --- gather --------------------------------------------------------
        "rows" => {
            let data = f32_child(0);
            let dr = child_rec(0);
            let (nr, w) = (dr.shape[0] as usize, dr.shape[1] as usize);
            let indices = match &values[child_idx[1]] {
                Value::U32(idx) => idx.clone(),
                _ => Vec::new(),
            };
            (Value::F32(ops::rows(&data, nr, w, &indices)), Some(CompareKind::Float))
        }
        "cols" => {
            let data = f32_child(0);
            let dr = child_rec(0);
            let (nr, w) = (dr.shape[0] as usize, dr.shape[1] as usize);
            let indices = match &values[child_idx[1]] {
                Value::U32(idx) => idx.clone(),
                _ => Vec::new(),
            };
            (Value::F32(ops::cols(&data, nr, w, &indices)), Some(CompareKind::Float))
        }
        "bdot" => match recover_bdot(child_rec(0), child_rec(1), record) {
            Some((ta, tb, scale)) => {
                let (out, _) = ops::bdot(
                    &f32_child(0),
                    &child_rec(0).shape,
                    ta,
                    &f32_child(1),
                    &child_rec(1).shape,
                    tb,
                    scale,
                );
                (Value::F32(out), Some(CompareKind::Float))
            }
            None => (Value::Unavailable, Some(CompareKind::Float)),
        },

        // --- int8 path -----------------------------------------------------
        "intgemmPrepareA" => {
            // child0 is the float activation, child1 the alpha (quantMultA).
            let qa = match &values[child_idx[1]] {
                Value::F32(v) if !v.is_empty() => v[0],
                _ => 1.0,
            };
            (Value::U8(ops::prepare_a(&f32_child(0), qa)), Some(CompareKind::U8))
        }
        "intgemmAffine" => compute_affine(trace, model, record, child_idx, values),

        // Anything unmodeled: leave a hole, don't compare.
        _ => (Value::Unavailable, None),
    }
}

/// Recompute an int8 affine, or passthrough when its `B` is a packed
/// `SelectColumnsB` output with no plain model weight.
fn compute_affine(
    trace: &Trace,
    model: &Model,
    record: &TraceRecord,
    child_idx: &[usize],
    values: &[Value],
) -> (Value, Option<CompareKind>) {
    let a_rec = &trace.records[child_idx[0]];
    let b_rec = &trace.records[child_idx[1]];
    let bias_rec = &trace.records[child_idx[2]];

    let weight_name = b_rec.name.rsplit("::").next().unwrap_or(&b_rec.name);
    let weight = match model.get(weight_name).filter(|_| b_rec.op_type == "param") {
        Some(w) => w,
        // Logit projection via SelectColumnsB: passthrough the recorded output.
        None => {
            let v = record.to_f32().map(Value::F32).unwrap_or(Value::Unavailable);
            return (v, None);
        }
    };

    let k = *a_rec.shape.last().unwrap() as usize;
    let m = a_rec.num_elements() / k;
    let b = match weight.int8_transposed() {
        Ok(b) => b,
        Err(_) => return (Value::Unavailable, None),
    };
    let n = b.len() / k;
    let bias = bias_rec.to_f32().unwrap_or_default();

    // Recover the (static) unquantize multiplier from the clean trace: run the
    // integer GEMM on the *traced* A, then read the scale off the largest output.
    let a_clean: &[u8] = &a_rec.data;
    let zeros = vec![0.0f32; n];
    let int_dot = ops::intgemm_affine(a_clean, m, k, b, n, 1.0, &zeros);
    let expected = record.to_f32().unwrap_or_default();
    let unquant = recover_unquant(&int_dot, &expected, &bias, n);

    // Apply to the *recomputed* activation (quantized by our PrepareA).
    let a_computed = match &values[child_idx[0]] {
        Value::U8(a) => a.clone(),
        _ => a_clean.to_vec(),
    };
    let out = ops::intgemm_affine(&a_computed, m, k, b, n, unquant, &bias);
    (Value::F32(out), Some(CompareKind::Float))
}

// --- helpers ----------------------------------------------------------------

/// Resolve a node's children to record indices (nearest earlier record per id),
/// mirroring `Trace::inputs` but returning positions.
fn resolve_children(trace: &Trace, index: usize) -> Vec<usize> {
    let record = &trace.records[index];
    record
        .children
        .iter()
        .map(|&cid| {
            (0..index)
                .rev()
                .find(|&j| trace.records[j].id == cid)
                .unwrap_or(index) // dangling child (shouldn't happen in a valid trace)
        })
        .collect()
}

fn rows_cols(record: &TraceRecord) -> (usize, usize) {
    let cols = *record.shape.last().unwrap_or(&1) as usize;
    let rows = record.num_elements() / cols.max(1);
    (rows, cols)
}

fn recover_scalar_mult(input: &TraceRecord, out: &TraceRecord) -> f32 {
    let (i, o) = (input.to_f32().unwrap_or_default(), out.to_f32().unwrap_or_default());
    let argmax = (0..i.len()).max_by(|&a, &b| i[a].abs().total_cmp(&i[b].abs()));
    match argmax {
        Some(p) if i[p] != 0.0 => o[p] / i[p],
        _ => 1.0,
    }
}

fn recover_scalar_add(input: &TraceRecord, out: &TraceRecord) -> f32 {
    let (i, o) = (input.to_f32().unwrap_or_default(), out.to_f32().unwrap_or_default());
    if i.is_empty() { 0.0 } else { o[0] - i[0] }
}

fn recover_transpose_perm(input: &TraceRecord, out: &TraceRecord) -> Option<Vec<usize>> {
    let x = input.to_f32().ok()?;
    let expected = out.to_f32().ok()?;
    for perm in permutations(input.shape.len()) {
        let candidate: Vec<i32> = perm.iter().map(|&p| input.shape[p]).collect();
        if candidate != out.shape {
            continue;
        }
        let (o, _) = ops::transpose(&x, &input.shape, &perm);
        if o.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 1e-6 + 1e-4 * b.abs()) {
            return Some(perm);
        }
    }
    None
}

fn recover_slice_offset(input: &TraceRecord, out: &TraceRecord) -> Option<usize> {
    let x = input.to_f32().ok()?;
    let expected = out.to_f32().ok()?;
    let len = expected.len();
    (0..=x.len().saturating_sub(len)).find(|&off| {
        x[off..off + len].iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 1e-6 + 1e-4 * b.abs())
    })
}

fn recover_bdot(a: &TraceRecord, b: &TraceRecord, out: &TraceRecord) -> Option<(bool, bool, f32)> {
    let av = a.to_f32().ok()?;
    let bv = b.to_f32().ok()?;
    let expected = out.to_f32().ok()?;
    let want_mn = &out.shape[out.shape.len().saturating_sub(2)..];
    for &ta in &[false, true] {
        for &tb in &[false, true] {
            let (ra, ca) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
            let (rb, cb) = (b.shape[b.shape.len() - 2], b.shape[b.shape.len() - 1]);
            let (mm, kk) = if ta { (ca, ra) } else { (ra, ca) };
            let (kb, nn) = if tb { (cb, rb) } else { (rb, cb) };
            if kk != kb || want_mn != [mm, nn] {
                continue;
            }
            let (unscaled, _) = ops::bdot(&av, &a.shape, ta, &bv, &b.shape, tb, 1.0);
            let p = (0..unscaled.len()).max_by(|&x, &y| unscaled[x].abs().total_cmp(&unscaled[y].abs()))?;
            if unscaled[p] == 0.0 {
                continue;
            }
            let scale = expected[p] / unscaled[p];
            if unscaled.iter().zip(&expected).all(|(x, e)| (x * scale - e).abs() <= 1e-5 + 1e-3 * e.abs()) {
                return Some((ta, tb, scale));
            }
        }
    }
    None
}

fn recover_unquant(int_dot: &[f32], expected: &[f32], bias: &[f32], n: usize) -> f32 {
    let p = (0..int_dot.len()).max_by(|&x, &y| int_dot[x].abs().total_cmp(&int_dot[y].abs()));
    match p {
        Some(p) if int_dot[p] != 0.0 => (expected[p] - bias[p % n]) / int_dot[p],
        _ => 0.0,
    }
}

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
