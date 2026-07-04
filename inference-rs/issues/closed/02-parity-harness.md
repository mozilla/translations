# Parity Measurement Harness

Foundational for the parity follow-up work — nothing else is worth starting until
divergence from `translator-cli` is both *measurable* and *localizable*. Two tools:

## Corpus A/B match rate

A `task rs:parity` that runs a corpus through both `translator-cli` and the Rust
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

## Cheat-proof testing principles (canonical — other oracle issues reference this)

The acceptance criteria across the parity/oracle issues follow these rules so a stubbed or
overfit implementation can't pass:

1. **Diff against an independent oracle, never our own output.** The expected values come
   from `translator-cli`, the reference SentencePiece, or the recorded trace — code we did
   not write. A test that compares the engine to itself proves nothing.
2. **Held-out data, committed with a hash + fixed seed.** Use a public corpus
   (candidate: FLORES-200 devtest, or an NLLB sample) checked in with its sha256 and the
   sampling seed, so results are reproducible and nobody can tune the implementation to the
   test set.
3. **Whole-artifact equality, not spot checks.** Assert the *entire* token-id sequence /
   the *entire* per-node tensor within tolerance — not one element a stub could hardcode.
4. **No recover-from-output-then-check-output loops.** The oracle must supply inputs *and*
   expected outputs independently. (Some existing op-parity tests recover an attribute from
   the node's own output and then re-check that output — fine for a multi-element op like
   transpose, weak for a 1-scalar op. Prefer trace-supplied expectations.)
5. **Fail loud on empty.** If a selector matches zero cases, that's a failure, not a pass
   (a broken harness must not read as green).

## Goal (not a 100% gate)

The point is to **quantify** parity and give later work a number to move — not to require a
perfect match. So:

- `task rs:parity` reports greedy exact-match rate over the corpus and lists every
  mismatch with its first-divergence node (from the bisector).
- Each mismatch is *classified* — near-tie (small logit gap on the flipped token) vs. an
  actual node bug — so we know what we're looking at.
- Confirmed bugs get filed as follow-up issues; near-ties are recorded as the accepted
  residual. Nothing here blocks on reaching 100%.
- The bisector runs `graph::replay` at a fixed rtol/atol and names the first node over
  tolerance.

## Shortlist off = the production baseline (decided)

Production runs with the lexical shortlist **off** today — its quality wasn't good enough.
So parity is measured **shortlist-off on both sides** (our default is already opt-in
`--shortlist`), which is also the apples-to-apples comparison against what ships. The
shortlist is a "works where it works" optional capability: valid on shared vocab, gated off
on split vocab (see [05-split-vocab-oracle.md](./05-split-vocab-oracle.md)).

## Corpus (decided)

**NLLB**, for diversity — but it's huge and *order-biased* (can't just take the head). Plan:
download up to ~500 MB for the pair, then cache a **random subset** with a committed seed +
sha256 so runs are reproducible and offline.

- Sampling: reservoir-sample the stream as it downloads so the subset is uniform over what
  we read (not just the first N lines). Caveat, worth noting honestly: a 500 MB prefix of an
  order-sorted corpus is still prefix-biased. If the host supports HTTP range requests,
  prefer sampling line offsets spread across the whole file instead — better sample, no full
  download. Good-enough fallback is the prefix + reservoir.
- The same cached subset feeds [03-spm-oracle.md](./03-spm-oracle.md) and
  [10-perf-harness.md](./10-perf-harness.md).

## Tolerances (decided defaults — tunable)

- Bisector: reuse `compare::Tolerance::default` (rtol 1e-3 / atol 1e-5).
- Near-tie: the flipped token's logit is within ~1% of the winner (≈ the 0.14/14 gap we
  already saw). Adjust once real data lands.

## Example diff report (shape, for review)

```
corpus: nllb-enes (subset sha256 abc123…, seed 7), 500 sentences
greedy exact match: 468/500 (93.6%)
mismatches:
  #14  "…"  first divergence: decoder_l2_context softmax (Δ 2e-3)  -> near-tie (gap 0.08)
  #52  "…"  first divergence: encoder_l0 layernorm  (Δ 4e-1)       -> BUG: file follow-up
```
