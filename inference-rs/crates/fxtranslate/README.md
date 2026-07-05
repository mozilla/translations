# fxtranslate - A Rust port of the translations engine in Firefox

[Firefox uses high quality](https://support.mozilla.org/en-US/kb/website-translation), lightweight, CPU-only [translation models](https://mozilla.github.io/translations/firefox-models/) with a focus for on-device translations. This project is a Rust rewrite of the translations engine using the same [gemmology](https://github.com/mozilla/gemmology) matrix math library as Firefox. The `fxtranslate` crate is the underlying library, but there is also the [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) tool, which is a batteries-included command line tool that lets you download the Firefox models, and run them through either an interactive terminal, or through piping through stdout.

## Usage

The [fxtranslate-cli](https://crates.io/crates/fxtranslate-cli) lets you quickly try out the translations system. It installs an `fxtranslate` binary that uses the native SIMD kernel where one is wired (aarch64 and x86_64) and a portable scalar fallback everywhere else, so the install never needs a C++ toolchain to succeed.

```console
$ cargo install fxtranslate-cli

# Translate a phrase. The model for the pair is discovered, downloaded, and
# cached on first use, then reused from disk on subsequent runs.
$ fxtranslate translate en es "The weather is nice today."
El clima es agradable hoy.

$ fxtranslate translate en fr "I would like a cup of coffee, please."
Je voudrais une tasse de café, s'il vous plaît.

$ fxtranslate translate es en "Buenos días, ¿cómo estás?"
Good morning, how are you?
```

Every Firefox Translations model translates to or from English, so each direction is its own model (`en → ru` and `ru → en` are separate downloads). `fxtranslate list` enumerates the ~100 available pairs.

## Using the library

Add the engine as a dependency. The `net` feature pulls in model management plus a built-in HTTPS client, which is the quickest way to get translating:

```console
$ cargo add fxtranslate --features net
```

With `net` on, the whole flow — discover the model in Remote Settings, download and hash-verify it into a local cache, build the engine, translate — is one call. This is a compile-checked doctest from the crate:

```rust,no_run
use fxtranslate::{cache::Cache, fetch::NetworkFetch, loader::load_engine};

let engine = load_engine(&NetworkFetch::new(), &Cache::locate(), "en", "es")?;
assert_eq!(engine.translate("The weather is nice today."), "El clima es agradable hoy.");
# Ok::<(), String>(())
```

### The model store

Models live as a small triple of files — a marian `.bin` model and the source/target SentencePiece (`.spm`) vocabularies (the two vocab paths are the same file for most pairs; only CJK pairs differ). `Cache` owns the on-disk layout under the platform-native cache directory (`Cache::locate()`), e.g. `~/Library/Caches/fxtranslate/models` on macOS, `$XDG_CACHE_HOME/fxtranslate/models` on Linux, `%LOCALAPPDATA%\fxtranslate\models` on Windows, with one subdirectory per `<src>-<trg>` pair. Pass an explicit root with `Cache::with_root(path)` to keep models alongside your application instead.

`loader::load_engine` downloads on a miss and reuses on a hit. If you'd rather manage the engine yourself, `loader::ensure_files` returns the verified on-disk paths without building the engine, so you can hand them to `Engine::load` (owned) or `Engine::load_mmapped` (weight tensors become views into a memory mapping — lower resident memory; see the feature table):

```rust,no_run
use fxtranslate::{cache::Cache, engine::Engine, fetch::NetworkFetch, loader::ensure_files};

let files = ensure_files(&NetworkFetch::new(), &Cache::locate(), "en", "es")?;
let engine = Engine::load_mmapped(&files.model, &files.src_vocab, &files.trg_vocab)?;
println!("{}", engine.translate("The weather is nice today."));
# Ok::<(), String>(())
```

The `net` feature is the batteries-included download path. If you'd rather use your own HTTP stack (or already have the model files on disk), the download mechanics are all opt-in — see the feature table below.

### Re-host the models — don't rely on Firefox's hosting

Model discovery resolves against Mozilla's Remote Settings, and the model attachments are served from Firefox's CDN. **That hosting is provisioned for Firefox, not for third-party traffic**, and neither the endpoints nor the availability of any particular model are a stable contract for downstream projects. If you ship `fxtranslate` in your own product, mirror the model files you depend on and serve them from infrastructure you control, then load them straight through `Engine::load`.

## Cargo features

The inference engine is the default; model management is opt-in so a plain
dependency (and wasm) stays lean — the default build pulls only `memmap2`.

| feature | adds | pulls |
|---|---|---|
| *(default)* `fast` | the native SIMD kernel where wired (aarch64 i8mm, x86_64 AVX2 — see [gemm-backends.md](https://github.com/gregtatum/translations/blob/inference-rs/inference-rs/gemm-backends.md)), scalar fallback elsewhere, plus `mmap` | build-time `cc`, `memmap2` |
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

Care was taken to make this implementation as light and performant as possible, as the goal is for these models to be embeddable in on-device environments where memory and CPU performance are at a premium.

The benchmarks below compare identical work: the **same 28.4 MiB en→ru `base` model** (dim-emb 512, FFN 2048, 6 encoder / 2 SSRU decoder layers, tied 32k embeddings), the **same corpus** (Firefox's own *Frankenstein* benchmark page: 103 paragraph blocks, 9,513 words), single thread, shortlist off, on Apple Silicon. Although the model occupies only 28 MiB on disk, its in-memory footprint is substantially larger due to the large token embedding ("Wemb") table, intermediate activation buffers, and the transformer's per-layer key/value tensors retained during attention. As a result, a model of this size naturally expands to many times larger than its on-disk size while translating.

| engine                                 | words/s | translate | settled RSS | peak RSS |
| -------------------------------------- | ------: | --------: | ----------: | -------: |
| fxtranslate (Rust)                     |    1267 |     7.5 s |     149 MiB |  164 MiB |
| marian `block-bench` (native C++)      |    1323 |     7.2 s |     298 MiB |  298 MiB |
| Firefox (Wasm, Full-Page Translations) |     419 |    22.9 s |     355 MiB |        – |

Note that the Firefox inference process running Wasm includes other features such as a full SpiderMonkey engine and is only one process among many, so it is not an apples-to-apples comparison, and expected to be larger and slower.

- **Throughput: 0.96× native marian, ~3.0× the shipping Firefox Wasm path.** After the gemmology kernel swap both fxtranslate and marian spend ~80% of their time in the *same* i8mm GEMM kernel, so the remaining ~4% is how much GEMM work each issues, not kernel speed.
- **Memory: the lightest of the three** — 149 MiB settled is half of native marian's and 58% under Firefox's inference process alone. This is the payoff of running the embedding table and output projection in int8 (no retained f32 copy) and adopting a page-returning allocator (jemalloc); memory-mapping the model (`Engine::load_mmapped`) trims settled RSS further still.

## Acknowledgements

 * [Firefox Translations](https://github.com/mozilla/translations) - The project this work was based off of.
 * [Project Bergamot](https://browser.mt/) for the overall translations recipe, CPU optimized models, and kicking all of this off.
 * [Marian](https://github.com/marian-nmt/marian-dev/) for the model training and reference inference implementation.
 * [OPUS - Open Parallel Corpora](https://opus.nlpl.eu/) for collecting and unifying so many parallel and monolingual datasets.
 * [No Language Left Behind](https://arxiv.org/abs/2207.04672) for the largest collection of diverse web-scaped training data.
 * [Bicleaner AI](https://github.com/bitextor/bicleaner-ai) for the ability to clean the data for higher quality models.
 * Plus many more folks in the Firefox, translations, and researcher community!
