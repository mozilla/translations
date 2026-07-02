# Sentence Piece Oracle

We need really good sentence piece verification. Let's design a durable oracle that can iterate through edge cases with tokenization. It should load in some kind of substantial real-world corpus, maybe by sampling and committing from NLLB.

I would assume a diverse set of 10,000 examples would give me some confidence. There's some things like unicode byte fallbacking that would be good test.

I believe we're missing normalization behavior as well.

> Tokenizer normalization gaps → will bite on accented/CJK, invisible on my Latin sample.

## Cheat-proof design

Follows the principles in [02-parity-harness.md](./02-parity-harness.md). The oracle diffs
our `spm::encode` ids against an **independent** SentencePiece implementation, per example,
full id-sequence equality.

**Reference = `spm_encode` (decided).** Build the upstream `sentencepiece` `spm_encode`
binary (it's already vendored as a marian dependency, so it should build the way
`translator-cli` did) and run it on the model's own `.spm`. Use it to **generate goldens
once** — commit `(corpus subset, ids, sha256, seed)` — so CI diffs offline and `spm_encode`
is only needed to (re)generate, never in the test loop. Corpus subset comes from
[02-parity-harness.md](./02-parity-harness.md) (the cached NLLB sample).

## Edge-case buckets the corpus must cover (not just NLLB sampling)

Sampling a corpus gives breadth; these give the *sharp* cases and should be hand-committed
alongside:

- **Byte fallback**: emoji (👍), rare/unassigned codepoints, raw bytes with no piece.
- **NFKC normalization**: full-width digits `４２`→`42`, ligature `ﬁ`→`fi`, compatibility
  forms, combining vs. precomposed accents (`é` two ways).
- **Whitespace/control**: leading/trailing/multiple spaces, tabs, newlines, zero-width chars.
- **Scripts**: CJK, Arabic (RTL), Devanagari, mixed-script tokens.

Each bucket is where the current whitespace-only normalizer (see
[04-tokenizer-normalization.md](./04-tokenizer-normalization.md)) is expected to fail today —
so the oracle should fail loudly now and pass once normalization lands.

## Acceptance criteria

- 100% id-sequence match vs. the reference over the committed corpus + edge-case set
  (tokenizer parity is bit-exact achievable, so anything less is a bug).
- Round-trip: `decode(encode(x))` recovers `x` for the normalizable subset (document the
  known lossy cases, e.g. whitespace collapse).

## Open question

- Per-language split of the ~10k sample across pairs (the total and source are decided in
  02; just how to divide it).
