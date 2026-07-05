//! Lexical shortlist reader.
//!
//! The `lex.*.s2t.bin` file restricts the output vocabulary to a per-sentence
//! candidate set: the model only projects (and argmaxes over) those columns of
//! the tied embedding — this is what the reference's `intgemmSelectColumnsB`
//! feeds. Using it is required for exact parity: the full-vocab argmax can pick a
//! near-synonym the reference never considers.
//!
//! Binary layout (little-endian), from `marian-fork/src/data/shortlist.cpp`:
//!
//! ```text
//! header : magic:u64 | checksum:u64 | firstNum:u64 | bestNum:u64
//!          wordToOffsetSize:u64 | shortListsSize:u64
//! wordToOffset : wordToOffsetSize × u64   (per-source-word offset into shortLists)
//! shortLists   : shortListsSize   × u32   (target candidate ids)
//! ```
//!
//! The candidate set for a sentence (`getShortlist`, shortlist.cpp:118) is: the
//! first `firstNum` target ids (most frequent), plus — for a shared vocab — each
//! source token id itself, plus every lexical translation of each source token,
//! padded up to a multiple of 8.

/// Header field count (u64s): magic, checksum, firstNum, bestNum,
/// wordToOffsetSize, shortListsSize.
const HEADER_U64: usize = 6;

/// A loaded binary lexical shortlist.
pub struct Shortlist {
    first_num: u32,
    word_to_offset: Vec<u64>,
    short_lists: Vec<u32>,
}

impl Shortlist {
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Shortlist> {
        let bytes = std::fs::read(path)?;
        Ok(Shortlist::from_bytes(&bytes))
    }

    pub fn from_bytes(b: &[u8]) -> Shortlist {
        let u64_at = |i: usize| {
            let o = i * 8;
            u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
        };
        let first_num = u64_at(2) as u32;
        let word_to_offset_size = u64_at(4) as usize;
        let short_lists_size = u64_at(5) as usize;

        let word_to_offset: Vec<u64> = (0..word_to_offset_size)
            .map(|i| u64_at(HEADER_U64 + i))
            .collect();

        let sl_start = (HEADER_U64 + word_to_offset_size) * 8;
        let short_lists: Vec<u32> = (0..short_lists_size)
            .map(|i| {
                let o = sl_start + i * 4;
                u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
            })
            .collect();

        Shortlist {
            first_num,
            word_to_offset,
            short_lists,
        }
    }

    /// The candidate target ids for a source sentence, sorted. `shared` is true
    /// when source and target share a vocabulary (then source tokens are also
    /// candidates). Mirrors `LexicalShortlistGenerator::getShortlist`.
    pub fn candidates(&self, src_ids: &[u32], shared: bool) -> Vec<u32> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<u32> = BTreeSet::new();

        // The firstNum most-frequent target ids are always included.
        for i in 0..self.first_num {
            set.insert(i);
        }

        // Per source token: itself (shared vocab) + its lexical translations.
        for &w in src_ids {
            if shared {
                set.insert(w);
            }
            let w = w as usize;
            if w + 1 < self.word_to_offset.len() {
                let start = self.word_to_offset[w] as usize;
                let end = self.word_to_offset[w + 1] as usize;
                for &cand in &self.short_lists[start..end] {
                    set.insert(cand);
                }
            }
        }

        // Pad to a multiple of 8 with sequential ids (intgemm needs it), matching
        // the reference so the candidate column set is identical.
        let mut i = self.first_num;
        while set.len() % 8 != 0 {
            set.insert(i);
            i += 1;
        }

        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_include_first_num_and_translations() {
        // Header: magic, checksum, firstNum=2, bestNum=1, wordToOffsetSize=3,
        // shortListsSize=2. wordToOffset=[0,0,2] (word 0: none; word 1: ids at 0..2).
        let mut b = Vec::new();
        for v in [0u64, 0, 2, 1, 3, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u64, 0, 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in [100u32, 200] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let sl = Shortlist::from_bytes(&b);
        // Source token 1 -> translations {100, 200}; firstNum -> {0,1}; shared -> {1}.
        let cands = sl.candidates(&[1], true);
        // {0,1,100,200} then padded to a multiple of 8 with 2,3,4,5.
        assert!(cands.contains(&0) && cands.contains(&1));
        assert!(cands.contains(&100) && cands.contains(&200));
        assert_eq!(cands.len() % 8, 0);
    }
}
