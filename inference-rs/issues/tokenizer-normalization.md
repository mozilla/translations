# Tokenizer Normalization + Byte Fallback

The Rust SentencePiece tokenizer (`src/spm.rs`) currently skips the model's normalizer and
byte fallback: it does whitespace-escaping + a dummy prefix only. That is identity for
Latin text (so it's invisible on the en→es/en→fr samples) and silently wrong for accented
/ CJK / unusual Unicode input.

## Work

- Implement the `precompiled_charsmap` normalization from the `.spm` `normalizer_spec`
  (the NFKC-based Trie), matching SentencePiece.
- Add byte-fallback pieces for out-of-vocabulary characters instead of collapsing to
  `<unk>`.

## Why sequence this early

Unlike the numeric parity work, tokenizer id-parity is *bit-exact achievable* and stays
that way once correct. It's also the highest-correctness-stakes gap for non-Latin pairs,
and it gates any honest CJK evaluation.

## Validation

Via the corpus oracle in [spm-oracle.md](./spm-oracle.md) (NLLB sample + unicode /
byte-fallback edge cases). That oracle is the verification vehicle; this issue is the
implementation behind it. Split-vocab (CJK) tokenization is exercised through
[split-vocab-oracle.md](./split-vocab-oracle.md).
