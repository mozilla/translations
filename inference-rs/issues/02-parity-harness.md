# Parity Measurement Harness

Foundational for the parity follow-up work — nothing else is worth starting until
divergence from `translator-cli` is both *measurable* and *localizable*. Two tools:

## Corpus A/B match rate

A `task inference-rs:parity` that runs a corpus through both `translator-cli` and the Rust
engine and reports the exact-match rate plus the diff list. Turns "roughly 7/10 on a
handful of sentences" into a tracked number that every later change has to move. Feed it
pre-split single sentences — splitting is not a parity concern here (see
[15-sentence-splitting.md](./15-sentence-splitting.md)).

## Per-node first-divergence bisector

For a divergent sentence, record the reference trace (`MARIAN_TRACE`) and run
`graph::replay` against it with a *tight* tolerance to find the first node that exceeds it.
That immediately says whether a failure is tokenization, layernorm, attention, or the
projection, instead of guessing from the final text.

## First action

Bisect the current worst divergence ("Thank you very much" → "Je vous remercie beaucoup"
vs. the reference "Merci beaucoup.") to confirm it is a near-tie and not a latent bug.
Don't assume every divergence is a near-tie until the bisector says so — a non-tie node
divergence is a correctness bug and jumps the queue.

## What "parity" should mean (proposed target — refine during prioritization)

- **Tokenizer id-parity**: bit-exact and achievable → aim for 100% (see
  [03-spm-oracle.md](./03-spm-oracle.md)).
- **Per-node numeric parity**: within a tight tolerance.
- **Greedy output**: a high match-rate threshold on the corpus, accepting rare near-tie
  flips as inherent to f32 non-associativity (see
  [06-numeric-reduction-parity.md](./06-numeric-reduction-parity.md)). Bit-identical greedy on
  *every* input is explicitly not the bar.

## Out of scope for parity

- **Beam search** — the shipped config is `beam-size: 1`; only needed if we target beam>1
  models.
- **Batching / threading** — performance, not parity. Correct masking makes batched output
  identical to the single-sentence path, so it cannot change the match rate.

Broader feature-coverage parity with the Wasm build is tracked separately in
[08-plan-audit-wasm-parity.md](./08-plan-audit-wasm-parity.md); this issue is specifically about
numeric/output parity of the translation itself.
