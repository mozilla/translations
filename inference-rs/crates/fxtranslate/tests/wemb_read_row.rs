#![cfg(feature = "gemmology")]
//! Round-trip pin for reading rows back out of gemmology's packed layout.
//!
//! `PreparedB::read_row` inverts `PrepareBQuantizedTransposed`, so the raw int8
//! `Wemb` copy can be freed and embedding lookups served from the single packed
//! projection buffer. This asserts the inverse is exact for every row — pinning
//! the pack layout the way `gemm_parity` pins the multiply. If gemmology ever
//! changes its packing, this fails loudly rather than silently corrupting
//! embeddings.

use fxtranslate::gemm::PreparedB;

#[test]
fn read_row_roundtrips_every_row() {
    // `k` must be a multiple of 16 (the NEON int8 register); `n = 37` is
    // deliberately not a multiple of 8, to exercise the shim's row padding. The
    // second shape is vocab-scale, matching a real `Wemb`.
    for (n, k) in [(37usize, 64usize), (32000usize, 512usize)] {
        // Deterministic, wide-ranging int8 fill (no dev-dependency).
        let raw: Vec<i8> = (0..n * k)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 5) & 0xff) as i8)
            .collect();

        let Some(pb) = PreparedB::new(&raw, n, k) else {
            // No SIMD kernel compiled for this target (scalar stub) — nothing to
            // round-trip; the raw copy is kept on such builds.
            return;
        };

        let mut row = vec![0i8; k];
        for id in 0..n {
            pb.read_row(id, &mut row);
            assert_eq!(
                &row[..],
                &raw[id * k..(id + 1) * k],
                "row {id} of {n}x{k} must read back bit-identical"
            );
        }
    }
}
