//! Batched decoder parity (batch-invariance).
//!
//! `greedy_batch` / `translate_batch` must produce, for each sentence, exactly
//! what the single-sentence `greedy` / `translate` produces — decoder rows are
//! independent (SSRU cell per row, cross-attention to each sentence's own masked
//! context, no decoder self-attention), so batching a block must not change any
//! sentence's tokens. The single path is validated against the marian reference
//! trace, so this is cheat-proof: a cross-attention-mask or per-row EOS bug shows
//! up as a per-sentence divergence, not a comparison against our own batch output.
//!
//! Skips (rather than fails) when the en-fr model isn't downloaded.

use inference_rs::engine::Engine;

const MODEL: &str = "../data/models/enfr/model.enfr.intgemm.alphas.bin";
const VOCAB: &str = "../data/models/enfr/vocab.enfr.spm";

fn engine() -> Option<Engine> {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping batched-decode parity: {MODEL} absent");
        return None;
    }
    Some(Engine::load(MODEL, VOCAB, VOCAB).expect("engine loads"))
}

#[test]
fn greedy_batch_matches_single_per_sentence() {
    let Some(eng) = engine() else { return };
    // Mixed lengths so the block pads (source) and sentences finish at different
    // steps (target) — exercises the cross-attention mask and per-row EOS.
    let texts = [
        "The cat sat on the mat.",
        "Dogs run.",
        "Scientists carefully explained the experiment to the students.",
        "Birds fly south.",
    ];
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| eng.src_ids(t)).collect();

    let batched = eng.greedy_batch(&ids);
    for (b, sid) in ids.iter().enumerate() {
        let single = eng.greedy(sid);
        eprintln!(
            "sentence {b}: single {} toks, batched {} toks",
            single.len(),
            batched[b].len()
        );
        assert_eq!(
            batched[b], single,
            "sentence {b}: batched greedy diverges from single-sentence greedy"
        );
    }
}

#[test]
fn translate_batch_matches_single() {
    let Some(eng) = engine() else { return };
    let texts = [
        "Hello world.",
        "The quick brown fox jumps.",
        "Good morning.",
    ];
    let batched = eng.translate_batch(&texts);
    for (b, t) in texts.iter().enumerate() {
        assert_eq!(
            batched[b],
            eng.translate(t),
            "sentence {b}: translate_batch diverges from translate"
        );
    }
}
