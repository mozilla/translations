#![cfg(feature = "gemmology")]
//! Cheat-proof parity for the gemmology i8mm kernel.
//!
//! The gemmology-backed GEMM must match the scalar [`ops::intgemm_affine`] — the
//! kernel already validated against the marian oracle (`tests/int8_parity.rs`,
//! `tests/ops_parity.rs`). Two independent kernels agreeing, with an external
//! oracle at the base, so the gemmology path is validated transitively without a
//! tautology. Covered shapes include the transformer's inner dims and output
//! widths that are *not* multiples of 8 (the shim's zero-padding path).

use fxtranslate::gemm::PreparedB;
use fxtranslate::ops;

/// A small linear-congruential PRNG (deterministic, no dev-dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
    fn signed(&mut self) -> i8 {
        (self.next() >> 33) as i8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn argmax_row(v: &[f32], row: usize, n: usize) -> usize {
    let r = &v[row * n..(row + 1) * n];
    (0..n)
        .max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap())
        .unwrap()
}

fn check(m: usize, k: usize, n: usize, seed: u64) {
    let mut r = Lcg(seed);
    let a: Vec<u8> = (0..m * k).map(|_| r.byte()).collect();
    let b: Vec<i8> = (0..n * k).map(|_| r.signed()).collect();
    let bias: Vec<f32> = (0..n).map(|_| r.unit() * 10.0).collect();
    let unquant = 0.000_7_f32;

    let scalar = ops::intgemm_affine(&a, m, k, &b, n, unquant, &bias);
    // No SIMD kernel compiled for this target (non-aarch64, `portable`, or no C++
    // compiler): nothing to compare against, so skip rather than fail.
    let Some(prepared) = PreparedB::new(&b, n, k) else {
        eprintln!("skip m={m} k={k} n={n}: SIMD kernel not built for this target");
        return;
    };
    let gem = prepared.matmul(&a, m, unquant, &bias);

    assert_eq!(scalar.len(), gem.len(), "output length");
    let mut max_diff = 0.0f32;
    for (i, (&s, &g)) in scalar.iter().zip(&gem).enumerate() {
        let d = (s - g).abs();
        max_diff = max_diff.max(d);
        // Same integer accumulation and same unquant+bias formula → essentially
        // exact; allow a hair for f32 reassociation.
        assert!(
            d <= 1e-2 + 1e-4 * s.abs(),
            "value mismatch m={m} k={k} n={n} idx={i}: scalar={s} gemmology={g} diff={d}"
        );
    }
    // Greedy decode only reads the per-row argmax, so pin that exactly.
    for row in 0..m {
        assert_eq!(
            argmax_row(&scalar, row, n),
            argmax_row(&gem, row, n),
            "argmax mismatch m={m} k={k} n={n} row={row}"
        );
    }
    eprintln!("ok m={m} k={k} n={n} max_diff={max_diff:.2e}");
}

#[test]
fn matches_scalar_across_shapes() {
    check(1, 384, 32000, 1); // output projection, vocab-scale N (mult of 8)
    check(1, 384, 512, 2); // single-row affine
    check(8, 384, 384, 3); // batched, attention-shaped
    check(4, 1536, 384, 4); // FFN second layer (k=1536)
    check(2, 384, 1536, 5); // FFN first layer (n=1536)
    check(1, 384, 7, 6); // tiny N, not a multiple of 8 → padding path
    check(3, 384, 251, 7); // N not a multiple of 8, batched
    check(6, 16, 40, 8); // minimal k (16), small N
}

#[test]
fn rejects_k_not_multiple_of_16() {
    // gemmology's NEON int8 register is 16 wide; k=24 is unsupported and the
    // wrapper reports None so the caller keeps the scalar kernel.
    assert!(PreparedB::new(&vec![0i8; 3 * 24], 3, 24).is_none());
}
