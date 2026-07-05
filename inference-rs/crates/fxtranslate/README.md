# fxtranslate - A Rust port of the translations engine in Firefox

[Firefox uses high quality](https://support.mozilla.org/en-US/kb/website-translation), lightweight, CPU-only [translation models](https://mozilla.github.io/translations/firefox-models/) with a focus for on-device translations. This project is a Rust rewrite of the translations engine using the same [gemmology](https://github.com/mozilla/gemmology) matrix math library as Firefox. The `fxtranslate` crate is the underlying library, but there is also the [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) tool, which is a batteries-included command line tool that lets you download the Firefox models, and run them through either an interactive terminal, or through piping through stdout.

## Usage

The [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) lets you quickly try out the translations system.

```
[example cargo commands to install it]
[3 examples of translations grounded in reality using a wide variety of languages. Select phrases with solid translations, and nothing sketchy]
```

[insert developer-focused docs on how to use. Explain lightly the model store, remote settings, and strong recommendations to cache and re-host the models to not rely on Firefox hosting for downstream projects. Do a full example of translation using just this library. Ground this example in a rustdoc example. The rustdoc examples are great because they are actually checked.]

## Cargo features

The inference engine is the default; model management is opt-in so a plain
dependency (and wasm) stays lean — the default build pulls only `memmap2`.

| feature | adds | pulls |
|---|---|---|
| *(default)* `fast` | the native SIMD kernel where wired (aarch64 i8mm), scalar fallback elsewhere, plus `mmap` | build-time `cc`, `memmap2` |
| `portable` | forces the scalar kernel (no C++ toolchain) | — |
| `mmap` | `Engine::load_mmapped` — model tensors are views into a memory mapping (file-backed pages) instead of owned heap copies; on under `fast` | `memmap2` |
| `download` | Remote Settings discovery (`remote`), verified local cache (`cache`), the pluggable `fetch::Fetch` client + retry/resume, language display names (`lang`), and the `src→trg`→`Engine` convenience (`loader`) | `tinyjson`, `ruzstd`, `sha2`, `dirs` (pure-Rust, no TLS/C) |
| `net` | the built-in `fetch::NetworkFetch` (HTTPS via rustls); implies `download` | `ureq` |

An embedder that already has its own HTTP client (e.g. Firefox) enables just
`download` and implements `fetch::Fetch` over its own stack, reusing discovery and
the verified cache without pulling in a TLS stack. `loader::load_engine` is the
one-call path (discover → download+verify → build the engine) when `net` is on.

## Implementation Methodology

This project was built using AI assisted code generation. A cheat-proof harness was built to use the backing Marian inference as an oracle. An agentic loop was created to go through each opcode in the graph to match the underlying reference implementation using numeric similarity, rather than byte similarity. The agent was given access to the underlying Marian implementation. This implementation was the vendored fork of [browsermt/marian-dev](https://github.com/browsermt/marian-dev) that has been maintained by [mozilla/translations](https://github.com/mozilla/translations) for the Firefox translations program. A domain-expert in translations ([gregtatum](https://github.com/gregtatum/)) directed the code generation using the harness, and validated the implementation results for correctness.

The project was implemented in the following phases:

 * Get the `mozilla/translations` project building on macOS ARM outside of Docker.
 * Build the oracle harness so that the model's internal memory can be introspected.
 * Run the agent in a loop to implement the opcodes.
 * Continue with cheat-proof tests for implementing the full copy of the Marian expression graph, including building the SSRU network.
 * Bypass tokenization, and verify translations produce sensical and compatible translations.
 * Integrate gemmology into the operations and verify correctness.
 * Do memory and performance optimizations to get performance equivalent with the reference implementation.
 * To do this a cheat-proof perf harness was built with [dhat-rs](https://github.com/nnethercote/dhat-rs), RSS measurements, [samply](https://lib.rs/crates/samply), and https://www.npmjs.com/package/@firefox-devtools/profiler-cli. The harness was able to do it's own perf analysis through natural language.
 * Status quo performance improvements were applied, e.g. key-value caching and batching. This heavily used the reference implementation to find gaps in the initial implementation.
 * Novel memory improvements were applied by leveraging Rust's zero-cost memory abstractions, and choosing the appropriate allocators to improve actual RSS memory usage. Vanity-only memory metrics like eliminating small allocation churn were discarded as not actually improving the cheat proof metrics.
 * Cross-platform gemmology support was added and verified where possible using CI resources. These are documented in [gemm-backends.md](https://github.com/gregtatum/translations/blob/inference-rs/inference-rs/gemm-backends.md).
 * A batteries included translator-cli crate was built to provide ergonomic access to translations.

## Performance and memory

Care was taken to make this implementation as light and performant as possible, as the goal is for these to be embeddable in on-device situations where memory and performance is critical especially in a CPU-only environment.

[Insert salient tables including the RSS baselines comparing Marian, Firefox's Wasm inference process, and fxtranslate. No need to do broad self-talk about the process. Includes relevant perf metrics like tokens per second. inference-rs/notes contains the markdown analysis from the process, with citable data points. 08-10 are the relevant final pieces of the process. Only the final state is important to share here.]

## Acknowledgements

 * [Firefox Translations](https://github.com/mozilla/translations) - The project this work was based off of.
 * [Project Bergamot](https://browser.mt/) for the overall translations recipe, CPU optimized models, and kicking all of this off.
 * [Marian](https://github.com/marian-nmt/marian-dev/) for the model training and reference inference implementation.
 * [OPUS - Open Parallel Corpora](https://opus.nlpl.eu/) for collecting and unifying so many parallel and monolingual datasets.
 * [No Language Left Behind](https://arxiv.org/abs/2207.04672) for the largest collection of diverse web-scaped training data.
 * [Bicleaner AI](https://github.com/bitextor/bicleaner-ai) for the ability to clean the data for higher quality models.
 * Plus many more folks in the translations and researcher community!
