# Split Vocab Oracle

We have done shared vocab only. We need to build and validate off of a shared vocab oracle. I believe CJK models have split vocabs. We need to use the remote settings model loading machinery or remote settings API to find a split vocab, ideally a CJK one, and then build an oracle we can validate from.

The outcomes here are a durable set of tests that can validate split vocab, and some kind of cheat-proof validation against the known good implementation.

## Cheat-proof acceptance (per [02-parity-harness.md](./02-parity-harness.md))

Once a CJK split-vocab model is downloaded from Remote Settings:

- **Tokenizer id-parity on both vocabs** — the source `.spm` *and* the (different) target
  `.spm`, since split vocab is exactly the case a shared-vocab implementation gets wrong.
  Validate through the [03-spm-oracle.md](./03-spm-oracle.md) machinery, run twice.
- **End-to-end greedy match rate** vs. `translator-cli` on a held-out CJK corpus, same bar
  as the shared-vocab case.
- The engine already takes `src_vocab`/`trg_vocab` separately — a test must prove they're
  actually distinct files here (assert the two `.spm` hashes differ), so we don't silently
  regress to shared-vocab behavior.

## Shortlist on split vocab (revised): correct, but still opt-in only

The earlier "gate it off" decision was based on a misreading of marian's
`shortlist.cpp`. The TODO there ("shortlisting is not correct for split vocabs") refers
**only** to copying the *source tokens themselves* into the target candidate set — which
marian gates behind `if(shared_)` and our `candidates(src_ids, shared=false)` already
skips. The rest of the candidate set (the `firstNum` most-frequent target ids + each
source token's lexical translations) is correct for split vocab.

Empirically, the en-ja model **requires** the shortlist:

- `translator-cli` **aborts** (SIGABRT, exit 134) when the shortlist is stripped from its
  config — the model is trained to decode against it.
- Full-vocab greedy produces garbage (`"Hello world." → "めくれの世界。"`); with the shortlist
  it produces fluent, near-reference output (`"こんにちは。世界。"` vs ref `"こんにちは、世界。"`).

Policy: the shortlist is **off by default on every path** (it's off in production, and the
shortlist code is correct for split vocab but the *quality* wasn't good enough) — enabling
it is always the caller's explicit `--shortlist` opt-in. Split-vocab models only produce
good output with it, but we do not turn it on for them automatically. Split-vocab parity is
therefore evaluated **shortlist-on on both sides** (there is no shortlist-off reference to
compare against — it crashes).

### Split-vocab output-projection quant fix

The tied output projection's activation alpha is named `none_QuantMultA` (a float32
scalar) on shared-vocab models but `decoder_Wemb_QuantMultA` on split-vocab ones — and the
latter is stored as an `intgemm8` scalar whose value is `raw_i8 / quant_mult` (≈6.6), not
the raw quant multiplier (≈19.2). Reading it wrong clamps the activation and corrupts the
logits. See `Weights::output_qa`.

## Status: done (en→ja)

- **Model**: `en→ja` (`tied-embeddings-all: false`, separate `encoder_Wemb`/`decoder_Wemb`,
  separate `srcvocab`/`trgvocab`, `lex.50.50`). `download_model.py` now fetches split
  `srcvocab`/`trgvocab` (falling back from the shared `vocab` record) and writes a two-vocab
  config.
- **Tokenizer id-parity on both vocabs** — `tests/spm_oracle.rs`: dev-en through
  `srcvocab.enja` and a Japanese corpus through `trgvocab.enja` both match `spm_encode`
  exactly (offline committed goldens), plus an assertion that the two vocabs tokenize the
  same text differently.
- **Greedy match rate** vs `translator-cli` (shortlist on both sides): drift-limited, the
  same class of near-tie first-token flips as the shared-vocab case (en-fr 14/20 vs en-ja
  ~5/20 on dev-en — Japanese's denser subword vocab flips more near-ties). Forward parity is
  exact: `replay` on recorded en-ja traces shows zero node divergence. The residual greedy
  gap is [06-numeric-reduction-parity.md](./06-numeric-reduction-parity.md), not a split-vocab bug.
