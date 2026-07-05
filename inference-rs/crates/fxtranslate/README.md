# fxtranslate - A Rust port of the translations engine in Firefox

[Firefox uses high quality](https://support.mozilla.org/en-US/kb/website-translation), lightweight, CPU-only [translation models](https://mozilla.github.io/translations/firefox-models/) with a focus for on-device translations. This project is a Rust rewrite of the translations engine using the same [gemmology](https://github.com/mozilla/gemmology) matrix math library as Firefox. The `fxtranslate` crate is the underlying library, but there is also the [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) tool, which is a batteries-included command line tool that lets you download the Firefox models, and run them through either an interactive terminal, or through piping through stdout.

## Usage

The [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) lets you quickly try out the translations system.

```
[example cargo commands to install it]
[3 examples of translations grounded in reality using a wide variety of languages. Select phrases with solid translations, and nothing sketchy]
```

[insert developer-focused docs on how to use. Explain lightly the model store, remote settings, and strong recommendations to cache and re-host the models to not rely on Firefox hosting for downstream projects. Do a full example of translation using just this library. Ground this example in a rustdoc example. The rustdoc examples are great because they are actually checked.]

## Acknowledgements

 * [Firefox Translations](https://github.com/mozilla/translations) - The project this work was based off of.
 * [Project Bergamot](https://browser.mt/) for the overall translations recipe, CPU optimized models, and kicking all of this off.
 * [Marian](https://github.com/marian-nmt/marian-dev/) for the model training and reference inference implementation.
 * [OPUS - Open Parallel Corpora](https://opus.nlpl.eu/) for collecting and unifying so many parallel and monolingual datasets.
 * [No Language Left Behind](https://arxiv.org/abs/2207.04672) for the largest collection of diverse web-scaped training data.
 * [Bicleaner AI](https://github.com/bitextor/bicleaner-ai) for the ability to clean the data for higher quality models.
 * Plus many more folks in the translations and researcher community!
