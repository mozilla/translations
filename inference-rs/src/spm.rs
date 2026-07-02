//! SentencePiece tokenization in Rust.
//!
//! The Firefox vocab (`vocab.*.spm`) is a SentencePiece **unigram** model stored
//! as a protobuf `ModelProto`. This module reads that protobuf (only the handful
//! of fields we need) and implements the two directions:
//!
//! - **encode**: normalize text → escape whitespace as `▁` (U+2581) → Viterbi
//!   segmentation maximizing the sum of piece log-scores → token ids, then append
//!   EOS (marian appends `</s>` to the source, `sentencepiece_vocab.cpp:216`).
//! - **decode**: ids → pieces → replace `▁` with space and strip, skipping
//!   control/unknown pieces.
//!
//! Normalization currently does whitespace-escaping + a dummy prefix only, which
//! reproduces the reference tokenization for Latin text but not for accented/CJK
//! input.
//! TODO: implement the `precompiled_charsmap` normalizer — see
//! `issues/04-tokenizer-normalization.md`.

use std::collections::HashMap;

/// The whitespace marker SentencePiece substitutes for spaces (U+2581).
const SPACE: char = '\u{2581}';

/// SentencePiece piece types, from the `ModelProto` (`type` enum).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

impl PieceType {
    fn from_raw(v: u64) -> PieceType {
        match v {
            2 => PieceType::Unknown,
            3 => PieceType::Control,
            4 => PieceType::UserDefined,
            6 => PieceType::Byte,
            5 => PieceType::Unused,
            _ => PieceType::Normal, // 1 (NORMAL) and default
        }
    }
}

/// A loaded SentencePiece unigram vocabulary.
pub struct SpmVocab {
    pieces: Vec<String>,
    types: Vec<PieceType>,
    scores: Vec<f32>,
    by_piece: HashMap<String, u32>,
    eos_id: u32,
    unk_id: u32,
    max_piece_bytes: usize,
}

impl SpmVocab {
    /// Load and parse a `.spm` file.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<SpmVocab> {
        let bytes = std::fs::read(path)?;
        Ok(SpmVocab::from_bytes(&bytes))
    }

    /// Parse a `ModelProto` from bytes. Reads the repeated `pieces` (field 1),
    /// each a `SentencePiece { piece:1 (string), score:2 (float), type:3 (enum) }`.
    pub fn from_bytes(data: &[u8]) -> SpmVocab {
        let mut pieces = Vec::new();
        let mut types = Vec::new();
        let mut scores = Vec::new();

        let mut r = Pb::new(data);
        while let Some((field, wire)) = r.tag() {
            match (field, wire) {
                // repeated SentencePiece pieces = 1 (length-delimited message)
                (1, 2) => {
                    let msg = r.bytes();
                    let (piece, score, ptype) = parse_piece(msg);
                    pieces.push(piece);
                    scores.push(score);
                    types.push(ptype);
                }
                _ => r.skip(wire),
            }
        }

        let mut by_piece = HashMap::with_capacity(pieces.len());
        let mut max_piece_bytes = 1;
        for (id, p) in pieces.iter().enumerate() {
            by_piece.insert(p.clone(), id as u32);
            max_piece_bytes = max_piece_bytes.max(p.len());
        }

        // EOS / UNK by their canonical piece strings (marian uses the SPM ids).
        let eos_id = *by_piece.get("</s>").unwrap_or(&0);
        let unk_id = *by_piece.get("<unk>").unwrap_or(&1);

        SpmVocab {
            pieces,
            types,
            scores,
            by_piece,
            eos_id,
            unk_id,
            max_piece_bytes,
        }
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Normalize `text` into the SentencePiece input form: collapse runs of
    /// ASCII whitespace to single spaces, trim, prepend a dummy space, then map
    /// every space to `▁`.
    fn normalize(text: &str) -> String {
        let mut out = String::new();
        let mut prev_space = true; // leading true => dummy prefix, and trims leading ws
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_space {
                    out.push(SPACE);
                    prev_space = true;
                }
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        // A leading dummy prefix even when the text starts non-space.
        if !out.starts_with(SPACE) {
            let mut prefixed = String::with_capacity(out.len() + SPACE.len_utf8());
            prefixed.push(SPACE);
            prefixed.push_str(&out);
            out = prefixed;
        }
        // Drop a trailing marker left by trailing whitespace.
        while out.ends_with(SPACE) && out.chars().count() > 1 {
            out.pop();
        }
        out
    }

    /// Encode text to token ids (no EOS). Uses unigram Viterbi over the
    /// normalized string.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = Self::normalize(text);
        self.viterbi(&norm)
    }

    /// Encode text and append EOS, as the source pipeline does.
    pub fn encode_with_eos(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text);
        ids.push(self.eos_id);
        ids
    }

    /// Viterbi max-score segmentation of `norm` over the piece vocabulary.
    /// Positions are byte offsets at UTF-8 char boundaries. Characters with no
    /// covering piece fall back to a single `<unk>`.
    fn viterbi(&self, norm: &str) -> Vec<u32> {
        let n = norm.len();
        // best[i] = (score, prev_byte, piece_id) for the best path ending at byte i.
        let neg_inf = f32::NEG_INFINITY;
        let mut best_score = vec![neg_inf; n + 1];
        let mut back = vec![(usize::MAX, u32::MAX); n + 1];
        best_score[0] = 0.0;

        // Char boundaries for unknown fallback stepping.
        let is_boundary = |i: usize| i == n || norm.is_char_boundary(i);

        for i in 0..n {
            if !is_boundary(i) || best_score[i] == neg_inf {
                continue;
            }
            // Try every piece that is a prefix of norm[i..].
            let hi = (i + self.max_piece_bytes).min(n);
            let mut matched_any = false;
            let mut j = i + 1;
            while j <= hi {
                if !is_boundary(j) {
                    j += 1;
                    continue;
                }
                if let Some(&id) = self.by_piece.get(&norm[i..j]) {
                    // User-defined and normal pieces participate; control pieces
                    // (</s>, <unk>) are not matched from text.
                    let t = self.types[id as usize];
                    if t == PieceType::Normal || t == PieceType::UserDefined || t == PieceType::Byte
                    {
                        let cand = best_score[i] + self.scores[id as usize];
                        if cand > best_score[j] {
                            best_score[j] = cand;
                            back[j] = (i, id);
                            matched_any = true;
                        }
                    }
                }
                j += 1;
            }
            // Unknown fallback: consume one char as <unk> with a heavy penalty so
            // it is only used when nothing else covers the char.
            if !matched_any || best_score[i + next_char_len(norm, i)] == neg_inf {
                let nj = i + next_char_len(norm, i);
                let cand = best_score[i] - 10.0;
                if cand > best_score[nj] {
                    best_score[nj] = cand;
                    back[nj] = (i, self.unk_id);
                }
            }
        }

        // Backtrack.
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (prev, id) = back[pos];
            if prev == usize::MAX {
                break;
            }
            ids.push(id);
            pos = prev;
        }
        ids.reverse();
        ids
    }

    /// The piece string for an id.
    pub fn piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Decode token ids back to text: concatenate pieces, turn `▁` into spaces,
    /// strip the leading space, and skip control/unknown pieces.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let t = self
                .types
                .get(id as usize)
                .copied()
                .unwrap_or(PieceType::Normal);
            if t == PieceType::Control || t == PieceType::Unknown {
                continue;
            }
            out.push_str(self.piece(id));
        }
        out.replace(SPACE, " ").trim_start().to_string()
    }
}

/// Byte length of the UTF-8 char starting at `i` (>= 1, clamped to string end).
fn next_char_len(s: &str, i: usize) -> usize {
    let bytes = s.as_bytes();
    let mut j = i + 1;
    while j < s.len() && (bytes[j] & 0xC0) == 0x80 {
        j += 1;
    }
    j - i
}

/// Parse a `SentencePiece` sub-message into `(piece, score, type)`.
fn parse_piece(msg: &[u8]) -> (String, f32, PieceType) {
    let mut piece = String::new();
    let mut score = 0.0f32;
    let mut ptype = PieceType::Normal;
    let mut r = Pb::new(msg);
    while let Some((field, wire)) = r.tag() {
        match (field, wire) {
            (1, 2) => piece = String::from_utf8_lossy(r.bytes()).into_owned(),
            (2, 5) => score = f32::from_le_bytes(r.fixed32()),
            (3, 0) => ptype = PieceType::from_raw(r.varint()),
            _ => r.skip(wire),
        }
    }
    (piece, score, ptype)
}

/// A minimal protobuf reader (varint / length-delimited / fixed32 only — all the
/// SentencePiece `ModelProto` uses for the fields we read).
struct Pb<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Pb<'a> {
    fn new(b: &'a [u8]) -> Pb<'a> {
        Pb { b, pos: 0 }
    }

    /// Next `(field_number, wire_type)`, or `None` at end.
    fn tag(&mut self) -> Option<(u64, u64)> {
        if self.pos >= self.b.len() {
            return None;
        }
        let key = self.varint();
        Some((key >> 3, key & 7))
    }

    fn varint(&mut self) -> u64 {
        let mut result = 0u64;
        let mut shift = 0;
        while self.pos < self.b.len() {
            let byte = self.b[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }

    fn bytes(&mut self) -> &'a [u8] {
        let len = self.varint() as usize;
        let end = (self.pos + len).min(self.b.len());
        let slice = &self.b[self.pos..end];
        self.pos = end;
        slice
    }

    fn fixed32(&mut self) -> [u8; 4] {
        let mut out = [0u8; 4];
        if self.pos + 4 <= self.b.len() {
            out.copy_from_slice(&self.b[self.pos..self.pos + 4]);
        }
        self.pos += 4;
        out
    }

    /// Skip a field of the given wire type.
    fn skip(&mut self, wire: u64) {
        match wire {
            0 => {
                self.varint();
            }
            1 => self.pos += 8,
            2 => {
                let len = self.varint() as usize;
                self.pos += len;
            }
            5 => self.pos += 4,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_escapes_and_prefixes() {
        assert_eq!(
            SpmVocab::normalize("Hello world."),
            "\u{2581}Hello\u{2581}world."
        );
        assert_eq!(SpmVocab::normalize("  a  b  "), "\u{2581}a\u{2581}b");
    }
}
