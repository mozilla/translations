# Sentence Splitting

We don't really need to handle sentence splitting yet, as that step is going to be handled in the eventual integration back into Gecko. However, for the inference-rs/issues/rust-only-package.md issue, we may want to sequence work afterwards to provide a cfg feature for a sentence splitter, which will bring in an additional dependency of `icu_segmenter`. This can then handle arbitrarily large text sizes. Consider existing fallbacking behavior if a sentence split still busts the context size budget.
