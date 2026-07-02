//! CPU op implementations, validated against the reference trace
//! (build-plan.md, step 3: "op-level parity, float ops first").
//!
//! Each op is a pure function over row-major `f32` slices — no graph, no
//! ordering. The parity harness (`tests/ops_parity.rs`) pulls a node's exact
//! input and output tensors from a recorded trace, feeds the inputs here, and
//! asserts the output matches within tolerance. That closes the loop the whole
//! validation strategy rests on: real reference data in, Rust op out, compared
//! against the oracle.

/// Layer normalization over the last dimension, matching marian's
/// `LayerNormalizationImpl` (`tensors/cpu/tensor_operators.cpp:1103`).
///
/// For each of `rows` rows of `cols` elements:
///
/// ```text
/// mean  = sum(x) / cols
/// sigma = sqrt(sum((x - mean)^2) / cols + eps)
/// out   = gamma * (x - mean) / sigma + beta
/// ```
///
/// `gamma` and `beta` broadcast over rows. Each may be either `cols` long
/// (per-column) or length 1 (a shared scalar), matching marian's `alphaStride`
/// logic. `beta` is optional: `None` is the bias-less variant. The transformer
/// uses `eps = 1e-6` (`layers/generic.h:463`).
///
/// # Panics
/// If `input.len() != rows * cols`, or `gamma`/`beta` is neither `cols` nor `1`
/// long. These are fixture-shape errors, not runtime conditions.
pub fn layer_normalization(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols, "input length must be rows * cols");
    assert!(
        gamma.len() == cols || gamma.len() == 1,
        "gamma must be `cols` or 1 long, got {}",
        gamma.len()
    );
    if let Some(beta) = beta {
        assert!(
            beta.len() == cols || beta.len() == 1,
            "beta must be `cols` or 1 long, got {}",
            beta.len()
        );
    }

    // Broadcast index: 0 for a shared scalar, `i` for a per-column vector.
    let g = |i: usize| if gamma.len() == 1 { gamma[0] } else { gamma[i] };
    let b = |i: usize| match beta {
        Some(beta) if beta.len() == 1 => beta[0],
        Some(beta) => beta[i],
        None => 0.0,
    };

    let cols_f = cols as f32;
    let mut out = vec![0.0f32; input.len()];

    for row in 0..rows {
        let src = &input[row * cols..(row + 1) * cols];
        let dst = &mut out[row * cols..(row + 1) * cols];

        let mean = src.iter().sum::<f32>() / cols_f;
        let sq_sum: f32 = src.iter().map(|&x| (x - mean) * (x - mean)).sum();
        let sigma = (sq_sum / cols_f + eps).sqrt();

        for i in 0..cols {
            dst[i] = g(i) * ((src[i] - mean) / sigma) + b(i);
        }
    }

    out
}

/// ReLU: `max(0, x)`, elementwise (`node_operators_unary.h` `ReLUNodeOp`).
pub fn relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.max(0.0)).collect()
}

/// Unary negation: `-x`, elementwise (marian `negate` / `-` operator).
pub fn negate(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| -x).collect()
}

/// Multiply every element by a scalar (`ScalarMultNodeOp`).
pub fn scalar_mult(input: &[f32], scalar: f32) -> Vec<f32> {
    input.iter().map(|&x| x * scalar).collect()
}

/// Add a scalar to every element (`ScalarAddNodeOp`).
pub fn scalar_add(input: &[f32], scalar: f32) -> Vec<f32> {
    input.iter().map(|&x| x + scalar).collect()
}

/// Highway gate, matching marian's `HighwayForward`
/// (`tensor_operators.cpp:1593`): `out = σ(t)·a + (1 − σ(t))·b`, elementwise.
/// All three inputs must be the same length (marian applies it without
/// broadcasting).
///
/// # Panics
/// If the three slices differ in length.
pub fn highway(a: &[f32], b: &[f32], t: &[f32]) -> Vec<f32> {
    assert!(
        a.len() == b.len() && b.len() == t.len(),
        "highway inputs must be equal length: {}, {}, {}",
        a.len(),
        b.len(),
        t.len()
    );
    a.iter()
        .zip(b)
        .zip(t)
        .map(|((&a, &b), &t)| {
            let g = 1.0 / (1.0 + (-t).exp());
            g * a + (1.0 - g) * b
        })
        .collect()
}

/// Softmax over the last dimension, matching marian's CPU `Softmax`
/// (`tensor_operators.cpp`): subtract the row max for numerical stability, then
/// normalize `exp` by the row sum.
///
/// # Panics
/// If `input.len() != rows * cols`.
pub fn softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols, "input length must be rows * cols");
    let mut out = vec![0.0f32; input.len()];

    for row in 0..rows {
        let src = &input[row * cols..(row + 1) * cols];
        let dst = &mut out[row * cols..(row + 1) * cols];

        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..cols {
            let ex = (src[i] - max).exp();
            dst[i] = ex;
            sum += ex;
        }
        for v in dst.iter_mut() {
            *v /= sum;
        }
    }

    out
}

/// Elementwise binary add with numpy-style broadcasting, matching marian's
/// `PlusNodeOp` (Element-wise `_1 = _2 + _3` over broadcast shapes).
///
/// Shapes are right-aligned; a dimension broadcasts when the two sizes are
/// equal or one of them is 1. Returns the result data together with the
/// broadcast output shape.
///
/// # Panics
/// On incompatible shapes, or if a slice length disagrees with its shape.
pub fn add(a: &[f32], a_shape: &[i32], b: &[f32], b_shape: &[i32]) -> (Vec<f32>, Vec<i32>) {
    broadcast_binary(a, a_shape, b, b_shape, |x, y| x + y)
}

/// Right-align two shapes and compute the broadcast output shape, or `None` if
/// they are incompatible.
fn broadcast_shape(a: &[i32], b: &[i32]) -> Option<Vec<i32>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0i32; rank];
    for i in 0..rank {
        // Missing (left-padded) dims are treated as size 1.
        let da = a.get(a.len().wrapping_sub(rank - i)).copied().unwrap_or(1);
        let db = b.get(b.len().wrapping_sub(rank - i)).copied().unwrap_or(1);
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return None;
        };
    }
    Some(out)
}

/// Generic broadcasting elementwise binary op over row-major tensors.
fn broadcast_binary(
    a: &[f32],
    a_shape: &[i32],
    b: &[f32],
    b_shape: &[i32],
    f: impl Fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<i32>) {
    let numel = |s: &[i32]| s.iter().map(|&d| d as usize).product::<usize>();
    assert_eq!(a.len(), numel(a_shape), "a length disagrees with a_shape");
    assert_eq!(b.len(), numel(b_shape), "b length disagrees with b_shape");

    let out_shape = broadcast_shape(a_shape, b_shape)
        .unwrap_or_else(|| panic!("incompatible shapes {a_shape:?} and {b_shape:?}"));
    let rank = out_shape.len();

    // Row-major strides for a shape, with a stride of 0 on any dimension that is
    // being broadcast (size 1 against a larger output dim) so that index maps
    // back onto the single source element.
    let strides = |shape: &[i32]| -> Vec<usize> {
        let mut s = vec![0usize; rank];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            let out_dim = out_shape[rank - shape.len() + i];
            s[rank - shape.len() + i] = if shape[i] == 1 && out_dim != 1 { 0 } else { acc };
            acc *= shape[i] as usize;
        }
        s
    };
    let sa = strides(a_shape);
    let sb = strides(b_shape);

    let total: usize = out_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![0.0f32; total];
    let mut idx = vec![0usize; rank];
    for o in out.iter_mut() {
        let (mut ia, mut ib) = (0usize, 0usize);
        for d in 0..rank {
            ia += idx[d] * sa[d];
            ib += idx[d] * sb[d];
        }
        *o = f(a[ia], b[ib]);
        // Increment the mixed-radix index (row-major, last dim fastest).
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] as usize {
                break;
            }
            idx[d] = 0;
        }
    }

    (out, out_shape)
}

/// Reshape: reinterpret the same row-major data under a new shape. marian's
/// `reshape` is a metadata-only view, so the element order is unchanged.
///
/// # Panics
/// If `new_shape`'s element count differs from `input.len()`.
pub fn reshape(input: &[f32], new_shape: &[i32]) -> Vec<f32> {
    let n: usize = new_shape.iter().map(|&d| d as usize).product();
    assert_eq!(n, input.len(), "reshape must preserve element count");
    input.to_vec()
}

/// Permute tensor axes, matching marian's `TransposeND`. `perm` lists the input
/// axis that supplies each output axis, so `out_shape[i] = in_shape[perm[i]]`
/// (e.g. `perm = [0, 2, 1, 3]` swaps axes 1 and 2). Returns the permuted data
/// and the output shape.
///
/// # Panics
/// If `perm` is not a permutation of `0..in_shape.len()`, or `input`'s length
/// disagrees with `in_shape`.
pub fn transpose(input: &[f32], in_shape: &[i32], perm: &[usize]) -> (Vec<f32>, Vec<i32>) {
    let rank = in_shape.len();
    assert_eq!(perm.len(), rank, "perm rank must match shape rank");
    let mut seen = vec![false; rank];
    for &p in perm {
        assert!(p < rank && !seen[p], "perm must be a permutation of 0..{rank}");
        seen[p] = true;
    }
    let numel: usize = in_shape.iter().map(|&d| d as usize).product();
    assert_eq!(input.len(), numel, "input length disagrees with in_shape");

    let in_strides = row_major_strides(in_shape);
    let out_shape: Vec<i32> = perm.iter().map(|&p| in_shape[p]).collect();

    let mut out = vec![0.0f32; numel];
    let mut oidx = vec![0usize; rank];
    for slot in out.iter_mut() {
        // The output coordinate maps to input coordinate in_coord[perm[i]] = oidx[i].
        let mut in_off = 0usize;
        for i in 0..rank {
            in_off += oidx[i] * in_strides[perm[i]];
        }
        *slot = input[in_off];
        for d in (0..rank).rev() {
            oidx[d] += 1;
            if oidx[d] < out_shape[d] as usize {
                break;
            }
            oidx[d] = 0;
        }
    }

    (out, out_shape)
}

/// A memory-consecutive slice, matching marian's `sliceView` ("view a slice,
/// must be memory-consecutive"): the output is `len` contiguous elements of the
/// input starting at `offset`.
///
/// # Panics
/// If `offset + len` exceeds the input length.
pub fn slice_contiguous(input: &[f32], offset: usize, len: usize) -> Vec<f32> {
    assert!(offset + len <= input.len(), "slice out of range");
    input[offset..offset + len].to_vec()
}

/// Row-major (C-order) strides for a shape: the number of flat elements between
/// successive indices along each axis.
fn row_major_strides(shape: &[i32]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{assert_close, Tolerance};

    #[test]
    fn matches_hand_computed_single_row() {
        // x = [1, 2, 3, 4]; mean = 2.5; var = 1.25; sigma = sqrt(1.25) (eps~0).
        // normalized = (x - 2.5) / sqrt(1.25).
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let inv = 1.0 / 1.25f32.sqrt();
        let expected = [-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv];

        let out = layer_normalization(&x, &[1.0], None, 1, 4, 0.0);
        assert_close(&out, &expected, Tolerance::default());
    }

    #[test]
    fn applies_gamma_and_beta_per_column() {
        // With gamma=2 and beta=10 applied uniformly, the normalized row is
        // scaled by 2 and shifted by 10.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let inv = 1.0 / 1.25f32.sqrt();
        let norm = [-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv];
        let expected: Vec<f32> = norm.iter().map(|&n| 2.0 * n + 10.0).collect();

        let gamma = [2.0f32; 4];
        let beta = [10.0f32; 4];
        let out = layer_normalization(&x, &gamma, Some(&beta), 1, 4, 0.0);
        assert_close(&out, &expected, Tolerance::default());
    }

    #[test]
    fn normalizes_each_row_independently() {
        // Two rows with different scales must each end up zero-mean, unit-var.
        let x = [1.0f32, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        let out = layer_normalization(&x, &[1.0], None, 2, 4, 0.0);

        for row in 0..2 {
            let r = &out[row * 4..row * 4 + 4];
            let mean: f32 = r.iter().sum::<f32>() / 4.0;
            let var: f32 = r.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "row {row} mean {mean}");
            assert!((var - 1.0).abs() < 1e-4, "row {row} var {var}");
        }
    }

    #[test]
    #[should_panic(expected = "rows * cols")]
    fn rejects_wrong_input_length() {
        layer_normalization(&[1.0, 2.0, 3.0], &[1.0], None, 1, 4, 0.0);
    }

    #[test]
    fn relu_negate_scalars() {
        assert_eq!(relu(&[-2.0, -0.0, 3.0, -1.5]), vec![0.0, 0.0, 3.0, 0.0]);
        assert_eq!(negate(&[1.0, -2.0, 0.0]), vec![-1.0, 2.0, -0.0]);
        assert_eq!(scalar_mult(&[1.0, 2.0, 3.0], 2.5), vec![2.5, 5.0, 7.5]);
        assert_eq!(scalar_add(&[1.0, 2.0], 10.0), vec![11.0, 12.0]);
    }

    #[test]
    fn highway_gates_between_inputs() {
        // t = 0 → σ = 0.5 → exact midpoint of a and b.
        let out = highway(&[10.0], &[0.0], &[0.0]);
        assert_close(&out, &[5.0], Tolerance::default());
        // large +t → σ ≈ 1 → out ≈ a; large −t → σ ≈ 0 → out ≈ b.
        let out = highway(&[10.0, 10.0], &[0.0, 0.0], &[20.0, -20.0]);
        assert_close(&out, &[10.0, 0.0], Tolerance::new(1e-3, 1e-6));
    }

    #[test]
    fn softmax_uniform_and_peaked() {
        // Equal logits → uniform distribution.
        let out = softmax(&[0.0, 0.0, 0.0, 0.0], 1, 4);
        assert_close(&out, &[0.25, 0.25, 0.25, 0.25], Tolerance::default());
        // Each row sums to 1, independent of a large additive offset (stability).
        let out = softmax(&[1000.0, 1001.0, 1002.0], 1, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    #[test]
    fn add_same_shape() {
        let (out, shape) = add(&[1.0, 2.0, 3.0], &[3], &[10.0, 20.0, 30.0], &[3]);
        assert_eq!(shape, vec![3]);
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_broadcasts_bias_row() {
        // [2,3] + [1,3] broadcasts the bias across both rows.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) = add(&a, &[2, 3], &[10.0, 20.0, 30.0], &[1, 3]);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_broadcasts_scalar_and_ranks() {
        // [2,2] + [1] scalar, and rank mismatch [1,1,2] + [2].
        let (out, _) = add(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[100.0], &[1]);
        assert_eq!(out, vec![101.0, 102.0, 103.0, 104.0]);
        let (out, shape) = add(&[1.0, 2.0], &[1, 1, 2], &[10.0, 20.0], &[2]);
        assert_eq!(shape, vec![1, 1, 2]);
        assert_eq!(out, vec![11.0, 22.0]);
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn add_rejects_incompatible_shapes() {
        add(&[1.0, 2.0, 3.0], &[3], &[1.0, 2.0], &[2]);
    }

    #[test]
    fn reshape_preserves_order() {
        assert_eq!(reshape(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transpose_swaps_last_two_axes() {
        // [2,3] -> [3,2] with perm [1,0]: standard matrix transpose.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) = transpose(&x, &[2, 3], &[1, 0]);
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity_and_4d() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let (out, shape) = transpose(&x, &[1, 1, 2, 2], &[0, 1, 2, 3]);
        assert_eq!(shape, vec![1, 1, 2, 2]);
        assert_eq!(out, x);
        // Swap the two middle axes of a [1,2,3,1] tensor.
        let y: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let (out, shape) = transpose(&y, &[1, 2, 3, 1], &[0, 2, 1, 3]);
        assert_eq!(shape, vec![1, 3, 2, 1]);
        assert_eq!(out, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn slice_contiguous_extracts_block() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(slice_contiguous(&x, 2, 3), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
    }
}
