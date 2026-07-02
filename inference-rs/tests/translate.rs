//! End-to-end greedy translation (finalize-plan.md §Validation).
//!
//! The anchor: the traced en→fr run translates "Hello world." to
//! "Bonjour le monde." (target tokens [16060, 280, 514, 264]). This drives the
//! whole pipeline — tokenize → encode → SSRU greedy decode → detokenize — with
//! nothing read from the trace. Skips when the model/vocab are absent.

use inference_rs::engine::Engine;

const MODEL: &str = "../data/models/enfr/model.enfr.intgemm.alphas.bin";
const VOCAB: &str = "../data/models/enfr/vocab.enfr.spm";

fn engine() -> Option<Engine> {
    if !std::path::Path::new(MODEL).exists() || !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping translate: model or vocab absent");
        return None;
    }
    Some(Engine::load(MODEL, VOCAB, VOCAB).expect("engine loads"))
}

#[test]
fn translates_hello_world() {
    let Some(engine) = engine() else { return };

    // Greedy token ids must match the traced run.
    let src = engine_src_ids();
    let out = engine.greedy(&src);
    eprintln!("greedy ids: {out:?}");
    assert_eq!(out, vec![16060, 280, 514, 264], "greedy token ids");

    // And the detokenized text.
    let text = engine.translate("Hello world.");
    eprintln!("translation: {text:?}");
    assert_eq!(text, "Bonjour le monde.");
}

/// Source ids for "Hello world." + EOS, matching the trace.
fn engine_src_ids() -> Vec<u32> {
    vec![17169, 564, 264, 0]
}
