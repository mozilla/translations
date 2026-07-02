//! SentencePiece tokenizer oracle: diff our `spm::encode` against `spm_encode`.
//!
//! The golden id sequences in `corpora/*.ids` are produced by the upstream
//! `spm_encode` binary (`task inference-rs:spm-goldens`) and committed, so this
//! runs offline. We re-tokenize the same corpora with `SpmVocab` and compare
//! full id sequences per line — an independent oracle, not a self-check.
//!
//! Both corpora must match exactly: the `precompiled_charsmap` normalizer and
//! byte fallback are implemented, so tokenization is bit-identical to
//! `spm_encode` on the diverse NLLB corpus too.

use inference_rs::spm::SpmVocab;

const VOCAB: &str = "../data/models/enfr/vocab.enfr.spm";

fn vocab() -> Option<SpmVocab> {
    if !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping spm oracle: {VOCAB} absent");
        return None;
    }
    Some(SpmVocab::load(VOCAB).expect("vocab parses"))
}

/// (matched, total) full-id-sequence agreement between `SpmVocab::encode` and
/// the committed `spm_encode` goldens for a corpus.
fn match_rate(vocab: &SpmVocab, corpus: &str, goldens: &str) -> (usize, usize) {
    let corpus = std::fs::read_to_string(corpus).expect("corpus present");
    let goldens = std::fs::read_to_string(goldens).expect("goldens present");
    let lines: Vec<&str> = corpus.lines().collect();
    let golden_lines: Vec<&str> = goldens.lines().collect();
    assert_eq!(lines.len(), golden_lines.len(), "corpus/goldens line count");

    let mut matched = 0;
    for (line, gold) in lines.iter().zip(&golden_lines) {
        let expected: Vec<u32> = gold
            .split_whitespace()
            .map(|t| t.parse().expect("golden id"))
            .collect();
        if vocab.encode(line) == expected {
            matched += 1;
        }
    }
    (matched, lines.len())
}

#[test]
fn ascii_corpus_matches_exactly() {
    let Some(vocab) = vocab() else { return };
    let (m, t) = match_rate(&vocab, "corpora/dev-en.txt", "corpora/dev-en.ids");
    eprintln!("dev-en tokenizer oracle: {m}/{t}");
    // Pure ASCII — normalization is identity, so this must be exact and stay so.
    assert_eq!(m, t, "ASCII tokenization must match spm_encode exactly");
}

#[test]
fn nllb_corpus_matches_exactly() {
    let Some(vocab) = vocab() else { return };
    let (m, t) = match_rate(&vocab, "corpora/nllb-en-fr.txt", "corpora/nllb-en-fr.ids");
    eprintln!("nllb-en-fr tokenizer oracle: {m}/{t}");
    // charsmap normalization + byte fallback make this bit-exact vs spm_encode.
    assert_eq!(
        m, t,
        "diverse-corpus tokenization must match spm_encode exactly"
    );
}
