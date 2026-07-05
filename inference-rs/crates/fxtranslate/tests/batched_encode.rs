//! Batched encoder parity (batch-invariance).
//!
//! The padded/masked batch path must reproduce the single-sentence encoder for
//! every sentence's valid rows: batching is invariant, and padding/masking must
//! not leak across sentences. The single-sentence `encode()` is itself validated
//! against the marian reference trace (graph `replay`, zero node divergence), so
//! anchoring the batch path on it is cheat-proof — a mask/padding bug shows up as
//! a per-sentence divergence here, not a comparison against our own batch output.
//!
//! Skips (rather than fails) when the en-fr model isn't downloaded.

use fxtranslate::engine::Engine;

const MODEL: &str = "../../../data/models/enfr/model.enfr.intgemm.alphas.bin";
const VOCAB: &str = "../../../data/models/enfr/vocab.enfr.spm";

fn engine() -> Option<Engine> {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping batched-encode parity: {MODEL} absent");
        return None;
    }
    Some(Engine::load(MODEL, VOCAB, VOCAB).expect("engine loads"))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn batched_encode_matches_single_per_sentence() {
    let Some(eng) = engine() else { return };
    // Deliberately different lengths so the batch pads → exercises the key mask.
    let texts = ["The cat sat.", "Dogs run very fast today.", "Birds fly."];
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| eng.src_ids(t)).collect();

    let ctx = eng.encode_batch(&ids);
    for (b, sid) in ids.iter().enumerate() {
        let single = eng.encode(sid);
        let diff = max_abs_diff(&single, ctx.sentence(b));
        eprintln!("sentence {b} (len {}): max abs diff {diff:e}", sid.len());
        assert!(
            diff < 1e-3,
            "sentence {b}: batched encoder diverges from single-sentence (max abs {diff})"
        );
    }
}

#[test]
fn batch_of_one_equals_single() {
    let Some(eng) = engine() else { return };
    let ids = eng.src_ids("Hello world.");
    let ctx = eng.encode_batch(&[ids.clone()]);
    let diff = max_abs_diff(&eng.encode(&ids), ctx.sentence(0));
    assert!(
        diff < 1e-6,
        "batch-of-one must equal single-sentence (max abs {diff})"
    );
}
