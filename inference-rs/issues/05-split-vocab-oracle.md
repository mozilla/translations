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

## Shortlist on split vocab (decided): gate it off

marian's own `shortlist.cpp` notes shortlisting **is not correct for split vocabs** (source
tokens can't be copied into the target candidate set without retokenizing). Policy: the
shortlist **only works where it works** — enable it on shared vocab, and **gate it off on
split vocab** with a clear reason. The engine should refuse/auto-disable when `src != trg`
rather than produce wrong candidates.

This also matches production, where the shortlist is **off entirely** today (the quality
wasn't good enough — see [02-parity-harness.md](./02-parity-harness.md)). So split-vocab
parity is evaluated **shortlist-off on both sides**; there's no need to reproduce
`translator-cli`'s split-vocab shortlist behavior.

## Open questions

- Which pair / model? (e.g. en→ja, en→ko, en→zh — is there a shipped split-vocab model, and
  which is easiest to pull from Remote Settings?)
- How do we locate a split-vocab model programmatically (RS query by field), vs. hardcoding
  a known one for the test fixture?
