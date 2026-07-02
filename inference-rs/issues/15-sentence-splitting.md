# Sentence Splitting

We don't really need to handle sentence splitting yet, as that step is going to be handled in the eventual integration back into Gecko. However, for the inference-rs/issues/14-rust-only-package.md issue, we may want to sequence work afterwards to provide a cfg feature for a sentence splitter, which will bring in an additional dependency of `icu_segmenter`. This can then handle arbitrarily large text sizes. Consider existing fallbacking behavior if a sentence split still busts the context size budget.

## Not a parity concern

Because production splitting lives upstream (Gecko, or the `icu_segmenter` cfg feature for
the rust-only package), the engine's contract is **pre-split input, one sentence in →
one translation out**. Parity should therefore be evaluated per-sentence (see
[02-parity-harness.md](./02-parity-harness.md)); the `translator-cli` `¡Hola mundo!` difference
we saw is only because it runs its own `ssplit` on multi-sentence input, not a numeric gap.
