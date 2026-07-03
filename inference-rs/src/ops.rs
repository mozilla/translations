//! CPU op implementations.
//!
//! Each op is a pure function over row-major `f32` slices — no graph, no
//! ordering. Correctness is checked in `tests/ops_parity.rs`, which feeds each
//! op the exact input tensors recorded from the reference engine and compares
//! the output within tolerance.

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
            s[rank - shape.len() + i] = if shape[i] == 1 && out_dim != 1 {
                0
            } else {
                acc
            };
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
        assert!(
            p < rank && !seen[p],
            "perm must be a permutation of 0..{rank}"
        );
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

/// Gather rows of a 2-D `[num_rows, width]` tensor, matching marian's
/// `RowsNodeOp`/`CopyRows`: `out[i] = data[indices[i]]` (used for embedding
/// lookup). Returns `[indices.len(), width]` data.
///
/// # Panics
/// If `data.len() != num_rows * width`, or an index is out of range.
pub fn rows(data: &[f32], num_rows: usize, width: usize, indices: &[u32]) -> Vec<f32> {
    assert_eq!(
        data.len(),
        num_rows * width,
        "data length disagrees with shape"
    );
    let mut out = vec![0.0f32; indices.len() * width];
    for (i, &idx) in indices.iter().enumerate() {
        let idx = idx as usize;
        assert!(idx < num_rows, "row index {idx} out of range {num_rows}");
        out[i * width..(i + 1) * width].copy_from_slice(&data[idx * width..(idx + 1) * width]);
    }
    out
}

/// Gather columns of a 2-D `[num_rows, width]` tensor, matching marian's
/// `ColsNodeOp`/`CopyCols`: `out[r, j] = data[r, indices[j]]`. Returns
/// `[num_rows, indices.len()]` data.
///
/// # Panics
/// If `data.len() != num_rows * width`, or an index is out of range.
pub fn cols(data: &[f32], num_rows: usize, width: usize, indices: &[u32]) -> Vec<f32> {
    assert_eq!(
        data.len(),
        num_rows * width,
        "data length disagrees with shape"
    );
    let k = indices.len();
    let mut out = vec![0.0f32; num_rows * k];
    for r in 0..num_rows {
        for (j, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            assert!(idx < width, "col index {idx} out of range {width}");
            out[r * k + j] = data[r * width + idx];
        }
    }
    out
}

/// Batched matrix product, matching marian's `ProdBatched`/`bdot`:
/// `C[i] = scalar · op(A[i % batchA]) · op(B[i % batchB])` over `batchC =
/// max(batchA, batchB)` batches, where `op` transposes the last two dims when
/// the corresponding `trans` flag is set. The last two dims of each shape are
/// the matrix; all leading dims form the batch. Batches broadcast by modulo, as
/// in marian.
///
/// Returns the output data and its shape (batch dims from the larger operand,
/// then `m × n`).
///
/// # Panics
/// On rank < 2, mismatched contraction dims, or incompatible batch counts.
pub fn bdot(
    a: &[f32],
    a_shape: &[i32],
    transa: bool,
    b: &[f32],
    b_shape: &[i32],
    transb: bool,
    scalar: f32,
) -> (Vec<f32>, Vec<i32>) {
    assert!(
        a_shape.len() >= 2 && b_shape.len() >= 2,
        "bdot needs rank >= 2"
    );
    let (ra, ca) = (
        a_shape[a_shape.len() - 2] as usize,
        a_shape[a_shape.len() - 1] as usize,
    );
    let (rb, cb) = (
        b_shape[b_shape.len() - 2] as usize,
        b_shape[b_shape.len() - 1] as usize,
    );

    // Logical (post-transpose) dims: op(A) is m×k, op(B) is k×n.
    let (m, k) = if transa { (ca, ra) } else { (ra, ca) };
    let (kb, n) = if transb { (cb, rb) } else { (rb, cb) };
    assert_eq!(k, kb, "bdot inner dims must match: {k} vs {kb}");

    let mat_a = ra * ca;
    let mat_b = rb * cb;
    let batch_a = a.len() / mat_a;
    let batch_b = b.len() / mat_b;
    let batch_c = batch_a.max(batch_b);
    assert!(
        batch_c % batch_a == 0 && batch_c % batch_b == 0,
        "incompatible batch counts {batch_a} and {batch_b}"
    );

    let mut out = vec![0.0f32; batch_c * m * n];
    for i in 0..batch_c {
        let a_off = (i % batch_a) * mat_a;
        let b_off = (i % batch_b) * mat_b;
        let c_off = i * m * n;
        // op(A)[p,q] and op(B)[q,j] index into the physical row-major blocks.
        let opa = |p: usize, q: usize| {
            if transa {
                a[a_off + q * ca + p]
            } else {
                a[a_off + p * ca + q]
            }
        };
        let opb = |q: usize, j: usize| {
            if transb {
                b[b_off + j * cb + q]
            } else {
                b[b_off + q * cb + j]
            }
        };
        for p in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for q in 0..k {
                    acc += opa(p, q) * opb(q, j);
                }
                out[c_off + p * n + j] = scalar * acc;
            }
        }
    }

    // Output shape: leading batch dims of the larger operand, then m, n.
    let base = if batch_b >= batch_a { b_shape } else { a_shape };
    let mut out_shape: Vec<i32> = base[..base.len() - 2].to_vec();
    out_shape.push(m as i32);
    out_shape.push(n as i32);
    (out, out_shape)
}

/// Quantize a float activation into the *shifted* unsigned int8 domain,
/// matching gemmology's `Shift::PrepareA`/`QuantizeU`
/// (`gemmology.h:QuantizeU`, `TileU`): round to nearest (ties to even), clamp to
/// `[-127, 127]` (banning -128), then add 127 to land in `[0, 254]`.
///
/// `quant_mult` is the activation's `quantMultA` (the precomputed alpha).
pub fn prepare_a(input: &[f32], quant_mult: f32) -> Vec<u8> {
    let mut out = Vec::new();
    prepare_a_into(input, quant_mult, &mut out);
    out
}

/// [`prepare_a`] into a caller-owned buffer, reused across calls to avoid a fresh
/// allocation per affine (the buffer is resized to `input.len()`).
pub fn prepare_a_into(input: &[f32], quant_mult: f32, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(input.len());
    out.extend(input.iter().map(|&x| {
        let q = (x * quant_mult).round_ties_even();
        let clamped = q.clamp(-127.0, 127.0);
        (clamped as i32 + 127) as u8
    }));
}

/// Compute the *prepared* bias for the shifted int8 affine, matching marian's
/// `prepareBias` (`intgemm_interface.h:351`, `Int8Shift::PrepareBias`).
///
/// Because `PrepareA` shifts the activation by +127, the integer GEMM computes
/// `Σ_k (A_true[k]+127)·W[k,n]`, i.e. an extra `127·Σ_k W[k,n]` per output. The
/// prepared bias folds the cancelling term into the bias so the affine just adds
/// it:
///
/// ```text
/// prepared[n] = raw_bias[n] − 127·unquant · Σ_k W[k,n],   unquant = 1/(qA·qB)
/// ```
///
/// `b_transposed` is the logical weight as `[N, K]` (so `W[k,n] =
/// b_transposed[n*k_dim + k]`). `raw_bias` is `[N]`; pass zeros for the
/// bias-less "fake bias" affines (the `All` in `int8shiftAlphaAll`).
///
/// # Panics
/// If `b_transposed.len() != n * k` or `raw_bias.len() != n`.
pub fn prepare_bias(
    b_transposed: &[i8],
    n: usize,
    k: usize,
    raw_bias: &[f32],
    unquant_mult: f32,
) -> Vec<f32> {
    assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
    assert_eq!(raw_bias.len(), n, "raw_bias length must be n");
    (0..n)
        .map(|col| {
            let colsum: i32 = b_transposed[col * k..(col + 1) * k]
                .iter()
                .map(|&w| w as i32)
                .sum();
            raw_bias[col] - 127.0 * unquant_mult * colsum as f32
        })
        .collect()
}

/// The shifted int8 affine — the `int8shiftAlphaAll` GEMM, matching marian's
/// `AffineNodeOp` + `Int8Shift::Multiply` with an `UnquantizeAndAddBiasAndWrite`
/// callback (`intgemm_interface.h:524`).
///
/// `a` is the *shifted* activation: `PrepareA` added 127 to move it into the
/// unsigned `u8` domain, so `a[m,k] ∈ [0, 255]`. `b_transposed` is the logical
/// int8 weight as read from the model — a weight that is logically `[K, N]` is
/// stored transposed as `[N, K]`, so `W[k,n] == b_transposed[n*k_dim + k]`.
/// `bias` is the *prepared* bias (`prepareBias`), which already folds in the
/// `-127` shift-correction term, so the affine just adds it.
///
/// ```text
/// out[m,n] = unquant_mult · Σ_k a[m,k] · W[k,n] + bias[n]
/// ```
///
/// where the caller supplies `unquant_mult = scalar / (quantMultA · quantMultB)`
/// (the accumulation itself is exact integer arithmetic).
///
/// # Panics
/// If the operand lengths disagree with `m·k`, `n·k`, or `n`.
pub fn intgemm_affine(
    a: &[u8],
    m: usize,
    k: usize,
    b_transposed: &[i8],
    n: usize,
    unquant_mult: f32,
    bias: &[f32],
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A length must be m * k");
    assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
    assert_eq!(bias.len(), n, "bias length must be n");

    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        for col in 0..n {
            let b_col = &b_transposed[col * k..(col + 1) * k];
            // Exact integer dot: A is unsigned (0..255), B is signed (-127..127).
            let mut acc: i32 = 0;
            for i in 0..k {
                acc += a_row[i] as i32 * b_col[i] as i32;
            }
            out[row * n + col] = unquant_mult * acc as f32 + bias[col];
        }
    }
    out
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
        assert_eq!(
            reshape(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            vec![1.0, 2.0, 3.0, 4.0]
        );
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

    #[test]
    fn rows_gathers_embeddings() {
        // 3 rows of width 2; gather rows [2, 0, 2].
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = rows(&data, 3, 2, &[2, 0, 2]);
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn cols_gathers_columns() {
        // [2,3] tensor, gather columns [2, 0].
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = cols(&data, 2, 3, &[2, 0]);
        assert_eq!(out, vec![2.0, 0.0, 5.0, 3.0]);
    }

    #[test]
    fn bdot_plain_matmul() {
        // [1,2,3] @ [1,3,2] -> [1,2,2], single batch, no transpose, scalar 1.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let (out, shape) = bdot(&a, &[1, 2, 3], false, &b, &[1, 3, 2], false, 1.0);
        assert_eq!(shape, vec![1, 2, 2]);
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn bdot_transb_and_scale() {
        // A[2x2] @ B^T where B is [2x2]; scale 0.5. transB swaps B's dims.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0]; // identity; B^T is identity too
        let (out, shape) = bdot(&a, &[2, 2], false, &b, &[2, 2], true, 0.5);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn intgemm_affine_matches_reference() {
        // 1x2 shifted A, 3x2 weight stored transposed as [N=3, K=2].
        // W (logical [K=2, N=3]) columns are the rows of b_transposed.
        let a = [10u8, 20];
        let b_transposed = [1i8, 2, 3, 4, 5, 6]; // W[:,0]=[1,2], W[:,1]=[3,4], W[:,2]=[5,6]
        let bias = [100.0f32, 200.0, 300.0];
        let out = intgemm_affine(&a, 1, 2, &b_transposed, 3, 0.5, &bias);
        // dot0 = 10*1+20*2=50 -> 0.5*50+100=125
        // dot1 = 10*3+20*4=110 -> 0.5*110+200=255
        // dot2 = 10*5+20*6=170 -> 0.5*170+300=385
        assert_eq!(out, vec![125.0, 255.0, 385.0]);
    }

    #[test]
    fn prepare_bias_cancels_shift() {
        // With unquant and B chosen simply, prepared = raw - 127*unquant*colsum.
        // B transposed [N=2, K=2]: col0 = [1,2] (sum 3), col1 = [3,4] (sum 7).
        let b = [1i8, 2, 3, 4];
        let out = prepare_bias(&b, 2, 2, &[10.0, 20.0], 0.5);
        assert_eq!(
            out,
            vec![10.0 - 127.0 * 0.5 * 3.0, 20.0 - 127.0 * 0.5 * 7.0]
        );
    }

    #[test]
    fn prepare_a_shifts_and_clamps() {
        // x*q: [0, 1.4->1, 2.6->3(ties even n/a), -300->clamp -127] then +127.
        let out = prepare_a(&[0.0, 1.4, 2.6, -300.0], 1.0);
        assert_eq!(out, vec![127, 128, 130, 0]);
        // ties-to-even: 0.5 -> 0, 1.5 -> 2, then +127.
        let out = prepare_a(&[0.5, 1.5], 1.0);
        assert_eq!(out, vec![127, 129]);
        // large positive clamps to 127 -> 254.
        assert_eq!(prepare_a(&[1000.0], 1.0), vec![254]);
    }

    #[test]
    #[should_panic(expected = "B length must be n * k")]
    fn intgemm_affine_rejects_bad_b() {
        intgemm_affine(&[1, 2], 1, 2, &[1, 2, 3], 3, 1.0, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn bdot_broadcasts_batch() {
        // batchA=1 broadcasts against batchB=2.
        let a = [1.0, 1.0, 1.0, 1.0]; // 1 batch, 2x2 of ones
        let b = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2 batches, 2x2
        let (out, shape) = bdot(&a, &[1, 2, 2], false, &b, &[2, 2, 2], false, 1.0);
        assert_eq!(shape, vec![2, 2, 2]);
        // ones @ M sums columns of each batch.
        assert_eq!(out, vec![4.0, 6.0, 4.0, 6.0, 40.0, 60.0, 40.0, 60.0]);
    }
}
