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
}
