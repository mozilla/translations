//! SentencePiece tokenizer parity (finalize-plan.md §1 validation).
//!
//! Anchored on the traced `en→fr` run: the source ids recorded in the trace are
//! `[17169, 564, 264]` for "Hello world." (then EOS=0), and the greedy target
//! the model emits is `▁Bonjour ▁le ▁monde .` = `[16060, 280, 514, 264]` →
//! "Bonjour le monde." Skips when the vocab is absent.

use inference_rs::spm::SpmVocab;

const VOCAB_PATH: &str = "../data/models/enfr/vocab.enfr.spm";

fn vocab() -> Option<SpmVocab> {
    std::path::Path::new(VOCAB_PATH)
        .exists()
        .then(|| SpmVocab::load(VOCAB_PATH).expect("vocab parses"))
}

#[test]
fn encodes_source_like_the_trace() {
    let Some(v) = vocab() else {
        eprintln!("skipping: {VOCAB_PATH} absent");
        return;
    };
    assert_eq!(v.len(), 32000);
    assert_eq!(v.eos_id(), 0);

    assert_eq!(v.encode("Hello world."), vec![17169, 564, 264]);
    assert_eq!(v.encode_with_eos("Hello world."), vec![17169, 564, 264, 0]);
}

#[test]
fn decodes_target_to_text() {
    let Some(v) = vocab() else { return };
    // The greedy output tokens from the trace.
    assert_eq!(v.decode(&[16060, 280, 514, 264]), "Bonjour le monde.");
    // EOS is skipped in detokenization.
    assert_eq!(v.decode(&[16060, 280, 514, 264, 0]), "Bonjour le monde.");
}

#[test]
fn round_trips_a_few_sentences() {
    let Some(v) = vocab() else { return };
    for s in ["The cat sat.", "It works!", "One two three."] {
        let ids = v.encode(s);
        // Detokenizing the encode should recover the original (Latin, happy path).
        assert_eq!(v.decode(&ids), s, "round-trip for {s:?}");
    }
}
