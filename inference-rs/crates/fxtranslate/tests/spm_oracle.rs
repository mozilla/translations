//! SentencePiece tokenizer oracle: diff our `spm::encode` against `spm_encode`.
//!
//! The golden id sequences in `corpora/*.ids` are produced by the upstream
//! `spm_encode` binary (`task rs:spm-goldens`) and committed, so this
//! runs offline. We re-tokenize the same corpora with `SpmVocab` and compare
//! full id sequences per line — an independent oracle, not a self-check.
//!
//! Both corpora must match exactly: the `precompiled_charsmap` normalizer and
//! byte fallback are implemented, so tokenization is bit-identical to
//! `spm_encode` on the diverse NLLB corpus too.

use fxtranslate::spm::SpmVocab;

const VOCAB: &str = "../../../data/models/enfr/vocab.enfr.spm";
// Split-vocab (CJK) pair: distinct source/target SentencePiece models.
const ENJA_SRC: &str = "../../../data/models/enja/srcvocab.enja.spm";
const ENJA_TRG: &str = "../../../data/models/enja/trgvocab.enja.spm";

fn vocab() -> Option<SpmVocab> {
    if !std::path::Path::new(VOCAB).exists() {
        eprintln!("skipping spm oracle: {VOCAB} absent");
        return None;
    }
    Some(SpmVocab::load(VOCAB).expect("vocab parses"))
}

fn load_if_present(path: &str) -> Option<SpmVocab> {
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping split-vocab oracle: {path} absent");
        return None;
    }
    Some(SpmVocab::load(path).expect("vocab parses"))
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
    let (m, t) = match_rate(&vocab, "../../corpora/dev-en.txt", "../../corpora/dev-en.ids");
    eprintln!("dev-en tokenizer oracle: {m}/{t}");
    // Pure ASCII — normalization is identity, so this must be exact and stay so.
    assert_eq!(m, t, "ASCII tokenization must match spm_encode exactly");
}

#[test]
fn nllb_corpus_matches_exactly() {
    let Some(vocab) = vocab() else { return };
    let (m, t) = match_rate(&vocab, "../../corpora/nllb-en-fr.txt", "../../corpora/nllb-en-fr.ids");
    eprintln!("nllb-en-fr tokenizer oracle: {m}/{t}");
    // charsmap normalization + byte fallback make this bit-exact vs spm_encode.
    assert_eq!(
        m, t,
        "diverse-corpus tokenization must match spm_encode exactly"
    );
}

// --- split-vocab (CJK) oracle -------------------------------------------------
//
// The en-ja model ships two distinct SentencePiece models — an English source
// vocab and a Japanese target vocab. Both must tokenize bit-identically to
// `spm_encode`, against their own committed goldens (`task rs:spm-goldens`):
// dev-en through the source vocab, a Japanese corpus through the target vocab.

#[test]
fn split_source_vocab_matches_exactly() {
    let Some(src) = load_if_present(ENJA_SRC) else {
        return;
    };
    let (m, t) = match_rate(&src, "../../corpora/dev-en.txt", "../../corpora/dev-en.enja-src.ids");
    eprintln!("enja src tokenizer oracle: {m}/{t}");
    assert_eq!(
        m, t,
        "source-vocab tokenization must match spm_encode exactly"
    );
}

#[test]
fn split_target_vocab_matches_exactly() {
    let Some(trg) = load_if_present(ENJA_TRG) else {
        return;
    };
    let (m, t) = match_rate(&trg, "../../corpora/dev-ja.txt", "../../corpora/dev-ja.enja-trg.ids");
    eprintln!("enja trg tokenizer oracle: {m}/{t}");
    assert_eq!(
        m, t,
        "target-vocab tokenization must match spm_encode exactly"
    );
}

/// The source and target vocabs must be genuinely different models — otherwise
/// the split-vocab code paths (separate `encoder_Wemb`/`decoder_Wemb`, untied
/// output projection) would be silently exercising a shared vocab. Tokenizing
/// the same text through each must disagree.
#[test]
fn split_vocabs_are_distinct() {
    let (Some(src), Some(trg)) = (load_if_present(ENJA_SRC), load_if_present(ENJA_TRG)) else {
        return;
    };
    let text = "The quick brown fox jumps over the lazy dog.";
    assert_ne!(
        src.encode(text),
        trg.encode(text),
        "source and target vocabs must tokenize differently"
    );
}
