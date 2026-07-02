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
//! Normalization applies the model's `precompiled_charsmap` (NFKC-style, via a
//! darts double-array trie) plus whitespace escaping, and encoding uses byte
//! fallback (`<0xNN>` pieces) for out-of-vocabulary characters — so tokenization
//! is bit-identical to the reference `spm_encode` (see `tests/spm_oracle.rs`).

use std::collections::HashMap;

/// The whitespace marker SentencePiece substitutes for spaces (U+2581).
const SPACE: char = '\u{2581}';
/// UTF-8 bytes of `SPACE` (U+2581).
const SPACE_BYTES: &[u8] = "\u{2581}".as_bytes();

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
    /// Minimum piece score; the unknown-node score is `min_score - 10`.
    min_score: f32,
    normalizer: Normalizer,
    /// `<0xNN>` byte-piece ids for byte fallback: `byte_pieces[b]` is the id of
    /// the piece for raw byte `b`, when the model ships byte pieces.
    byte_pieces: Box<[Option<u32>; 256]>,
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

        let mut normalizer_msg: &[u8] = &[];
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
                // NormalizerSpec normalizer_spec = 3
                (3, 2) => normalizer_msg = r.bytes(),
                _ => r.skip(wire),
            }
        }
        let normalizer = Normalizer::from_spec(normalizer_msg);

        let mut by_piece = HashMap::with_capacity(pieces.len());
        let mut max_piece_bytes = 1;
        let mut byte_pieces = Box::new([None; 256]);
        for (id, p) in pieces.iter().enumerate() {
            by_piece.insert(p.clone(), id as u32);
            max_piece_bytes = max_piece_bytes.max(p.len());
            // Byte pieces look like `<0xNN>`; map the raw byte value to this id.
            if types[id] == PieceType::Byte {
                if let Some(b) = parse_byte_piece(p) {
                    byte_pieces[b as usize] = Some(id as u32);
                }
            }
        }

        // EOS / UNK by their canonical piece strings (marian uses the SPM ids).
        let eos_id = *by_piece.get("</s>").unwrap_or(&0);
        let unk_id = *by_piece.get("<unk>").unwrap_or(&1);
        let min_score = scores.iter().copied().fold(f32::MAX, f32::min);

        SpmVocab {
            pieces,
            types,
            scores,
            by_piece,
            eos_id,
            unk_id,
            max_piece_bytes,
            min_score,
            normalizer,
            byte_pieces,
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

    /// Encode text to token ids (no EOS): SentencePiece normalization (the
    /// `precompiled_charsmap` + whitespace handling) then unigram Viterbi.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalizer.normalize(text);
        self.viterbi(&norm)
    }

    /// Encode text and append EOS, as the source pipeline does.
    pub fn encode_with_eos(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text);
        ids.push(self.eos_id);
        ids
    }

    /// Viterbi max-score segmentation of `norm`, matching SentencePiece's
    /// unigram lattice (`unigram_model.cc` `PopulateNodes`/`Encode`). Normal and
    /// user-defined pieces are matched as substrings at UTF-8 char boundaries; a
    /// character with no covering piece takes an unknown node scored
    /// `min_score - 10`. On output, an unknown character is expanded to its raw
    /// `<0xNN>` byte pieces (byte fallback), or emitted as `<unk>` if the model
    /// ships no byte pieces.
    fn viterbi(&self, norm: &str) -> Vec<u32> {
        let n = norm.len();
        let neg_inf = f32::NEG_INFINITY;
        let mut best_score = vec![neg_inf; n + 1];
        // back[j] = (prev byte offset, piece id chosen to reach j).
        let mut back = vec![(usize::MAX, u32::MAX); n + 1];
        best_score[0] = 0.0;

        let is_boundary = |i: usize| i == n || norm.is_char_boundary(i);
        // Unknown nodes cost less than any real piece, so they only win where no
        // piece covers the character.
        let unk_score = self.min_score - 10.0;

        for i in 0..n {
            if !is_boundary(i) || best_score[i] == neg_inf {
                continue;
            }

            // Normal / user-defined pieces: substring match at char boundaries.
            let hi = (i + self.max_piece_bytes).min(n);
            let mut j = i + 1;
            while j <= hi {
                if is_boundary(j) {
                    if let Some(&id) = self.by_piece.get(&norm[i..j]) {
                        let t = self.types[id as usize];
                        if t == PieceType::Normal || t == PieceType::UserDefined {
                            let cand = best_score[i] + self.scores[id as usize];
                            if cand > best_score[j] {
                                best_score[j] = cand;
                                back[j] = (i, id);
                            }
                        }
                    }
                }
                j += 1;
            }

            // Unknown fallback: one character, expanded to bytes on output.
            let nj = i + next_char_len(norm, i);
            let cand = best_score[i] + unk_score;
            if cand > best_score[nj] {
                best_score[nj] = cand;
                back[nj] = (i, self.unk_id);
            }
        }

        // Backtrack, expanding unknown characters into byte pieces.
        let bytes = norm.as_bytes();
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (prev, id) = back[pos];
            if prev == usize::MAX {
                break;
            }
            if id == self.unk_id {
                // Byte fallback: emit `<0xNN>` for each byte of the character,
                // or `<unk>` if the model has no byte pieces. Pushed reversed;
                // the final `ids.reverse()` restores order.
                let mut any_byte = false;
                for &b in bytes[prev..pos].iter().rev() {
                    if let Some(bid) = self.byte_pieces[b as usize] {
                        ids.push(bid);
                        any_byte = true;
                    }
                }
                if !any_byte {
                    ids.push(self.unk_id);
                }
            } else {
                ids.push(id);
            }
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

/// The SentencePiece normalizer: the `precompiled_charsmap` (a darts-clone
/// double-array trie mapping input byte sequences to normalized replacements)
/// plus the whitespace-handling flags. Ports `Normalizer::Normalize` from
/// `sentencepiece/src/normalizer.cc`.
struct Normalizer {
    /// darts-clone double-array units.
    trie: Vec<u32>,
    /// Concatenated NUL-terminated normalized replacement strings; trie values
    /// are byte offsets into this blob.
    normalized: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_ws: bool,
    escape_ws: bool,
}

impl Normalizer {
    /// Build from a serialized `NormalizerSpec` message (field 2 =
    /// precompiled_charsmap bytes; fields 3/4/5 = the whitespace flags, all
    /// defaulting to true in the proto).
    fn from_spec(msg: &[u8]) -> Normalizer {
        let (mut charsmap, mut add_dummy_prefix, mut remove_extra_ws, mut escape_ws) =
            (Vec::new(), true, true, true);
        let mut r = Pb::new(msg);
        while let Some((field, wire)) = r.tag() {
            match (field, wire) {
                (2, 2) => charsmap = r.bytes().to_vec(),
                (3, 0) => add_dummy_prefix = r.varint() != 0,
                (4, 0) => remove_extra_ws = r.varint() != 0,
                (5, 0) => escape_ws = r.varint() != 0,
                _ => r.skip(wire),
            }
        }
        // charsmap blob: u32 trie-byte-size, then the trie units, then the blob.
        let (trie, normalized) = if charsmap.len() > 4 {
            let trie_bytes =
                u32::from_le_bytes([charsmap[0], charsmap[1], charsmap[2], charsmap[3]]) as usize;
            let end = (4 + trie_bytes).min(charsmap.len());
            let trie = charsmap[4..end]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (trie, charsmap[end..].to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        Normalizer {
            trie,
            normalized,
            add_dummy_prefix,
            remove_extra_ws,
            escape_ws,
        }
    }

    /// darts-clone `commonPrefixSearch`, returning the longest matching key at
    /// the start of `key` as `(value, matched_len)`. `value` is a byte offset
    /// into `normalized`.
    fn longest_prefix(&self, key: &[u8]) -> Option<(u32, usize)> {
        if self.trie.is_empty() {
            return None;
        }
        // Unit accessors (darts.h): has_leaf, value, label, offset.
        let has_leaf = |u: u32| (u >> 8) & 1 == 1;
        let value = |u: u32| u & 0x7fff_ffff;
        let label = |u: u32| u & (0x8000_0000 | 0xff);
        let offset = |u: u32| (u >> 10) << ((u & 0x200) >> 6);

        let mut node_pos = (offset(self.trie[0]) as usize) ^ 0;
        let mut best = None;
        for (i, &b) in key.iter().enumerate() {
            node_pos ^= b as usize;
            let unit = *self.trie.get(node_pos)?;
            if label(unit) != b as u32 {
                return best;
            }
            node_pos ^= offset(unit) as usize;
            if has_leaf(unit) {
                let leaf = *self.trie.get(node_pos)?;
                best = Some((value(leaf), i + 1)); // longer matches overwrite
            }
        }
        best
    }

    /// The normalized replacement for the longest rule at the start of `input`,
    /// and how many input bytes it consumes. With no rule, copies one valid
    /// UTF-8 char (or the replacement char for a malformed byte).
    fn normalize_prefix(&self, input: &[u8]) -> (Vec<u8>, usize) {
        if let Some((val, len)) = self.longest_prefix(input) {
            let start = val as usize;
            let rel_end = self.normalized[start..]
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(self.normalized.len() - start);
            return (self.normalized[start..start + rel_end].to_vec(), len);
        }
        match utf8_char_len(input) {
            Some(len) => (input[..len].to_vec(), len),
            None => (vec![0xEF, 0xBF, 0xBD], 1), // U+FFFD, consume one byte
        }
    }

    /// SentencePiece `Normalize`: charsmap normalization + dummy prefix +
    /// whitespace escaping (`▁`) + extra-whitespace removal, per the flags.
    fn normalize(&self, text: &str) -> String {
        let sp = SPACE_BYTES;
        let input = text.as_bytes();
        let mut i = 0;

        // Ignore leading whitespace.
        if self.remove_extra_ws {
            while i < input.len() {
                let (rep, consumed) = self.normalize_prefix(&input[i..]);
                if rep != b" " {
                    break;
                }
                i += consumed;
            }
        }
        if i >= input.len() {
            return String::new();
        }

        let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
        let add_ws = |out: &mut Vec<u8>| {
            if self.escape_ws {
                out.extend_from_slice(sp);
            } else {
                out.push(b' ');
            }
        };
        if self.add_dummy_prefix {
            add_ws(&mut out);
        }

        let mut is_prev_space = self.remove_extra_ws;
        while i < input.len() {
            let (rep, consumed) = self.normalize_prefix(&input[i..]);
            let mut piece = &rep[..];
            while is_prev_space && piece.first() == Some(&b' ') {
                piece = &piece[1..];
            }
            if !piece.is_empty() {
                for &c in piece {
                    if self.escape_ws && c == b' ' {
                        out.extend_from_slice(sp);
                    } else {
                        out.push(c);
                    }
                }
                is_prev_space = piece.last() == Some(&b' ');
            }
            i += consumed;
            if !self.remove_extra_ws {
                is_prev_space = false;
            }
        }

        // Ignore trailing whitespace.
        if self.remove_extra_ws {
            let tail: &[u8] = if self.escape_ws { sp } else { b" " };
            while out.ends_with(tail) {
                out.truncate(out.len() - tail.len());
            }
        }

        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Parse a `<0xNN>` byte-piece string into its raw byte value.
fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// Byte length of the valid UTF-8 char at the start of `b`, or `None` if it is
/// malformed (bad lead byte or truncated/invalid continuation).
fn utf8_char_len(b: &[u8]) -> Option<usize> {
    let lead = *b.first()?;
    let len = match lead {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return None,
    };
    if b.len() < len || b[1..len].iter().any(|&c| c & 0xc0 != 0x80) {
        return None;
    }
    Some(len)
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
        // With no charsmap, normalization reduces to whitespace handling:
        // dummy prefix, escape spaces as ▁, collapse/trim extra whitespace.
        let n = Normalizer::from_spec(&[]);
        assert_eq!(n.normalize("Hello world."), "\u{2581}Hello\u{2581}world.");
        assert_eq!(n.normalize("  a  b  "), "\u{2581}a\u{2581}b");
    }

    #[test]
    fn utf8_char_len_basics() {
        assert_eq!(utf8_char_len(b"a"), Some(1));
        assert_eq!(utf8_char_len("é".as_bytes()), Some(2));
        assert_eq!(utf8_char_len("€".as_bytes()), Some(3));
        assert_eq!(utf8_char_len(&[0x80]), None); // stray continuation byte
    }
}
