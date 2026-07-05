# fxtranslate-cli - A batteries included translation CLI

A command-line front end for [fxtranslate](https://crates.io/crates/fxtranslate), a Rust port of the [translation engine in Firefox](https://mozilla.github.io/translations/firefox-models/). It uses the same high-quality, lightweight, CPU-only models Firefox ships for on-device translation, and handles discovering, downloading, and caching them for you. Installing it gives you an `fxtranslate` binary that uses the native SIMD kernel where one is wired (aarch64 and x86_64) and a portable scalar fallback everywhere else — so the install never needs a C++ toolchain to succeed.

For the engine itself, the developer API, and performance details, see the [main library](https://crates.io/crates/fxtranslate).

## Install

```console
$ cargo install fxtranslate-cli
```

## Usage

```console
# Enumerate the available <src>-<trg> pairs (or filter to a language):
$ fxtranslate list
$ fxtranslate list es

# Translate a phrase. The model for the pair is discovered, downloaded, and
# cached on first use, then reused from disk on subsequent runs.
$ fxtranslate translate en es "The weather is nice today."
El clima es agradable hoy.

$ fxtranslate translate en de "Knowledge is power."
Wissen ist Macht.

$ fxtranslate translate es en "Buenos días, ¿cómo estás?"
Good morning, how are you?
```

Every Firefox Translations model translates to or from English, so each direction is its own model (`en → es` and `es → en` are separate downloads).

With no text argument, `translate` reads from stdin — one translation per line when piped, or an interactive prompt on a terminal:

```console
# Pipe mode: one line in, one translation out.
$ echo "The library opens at nine in the morning." | fxtranslate translate en fr
La bibliothèque ouvre à neuf heures du matin.

# Interactive prompt (Ctrl-D to quit).
$ fxtranslate translate en es
Interactive en→es. Type a sentence and press Enter; Ctrl-D to quit.
en→es» ...
```

Status lines (model resolution, progress) are written to stderr, so piped stdout carries only the translations.

## Model cache

Models are cached under the platform-native cache directory — `~/Library/Caches/fxtranslate/models` on macOS, `$XDG_CACHE_HOME/fxtranslate/models` on Linux, `%LOCALAPPDATA%\fxtranslate\models` on Windows — with one subdirectory per language pair. Override the location with `--cache-dir <DIR>`.

The models are discovered via Mozilla's Remote Settings and downloaded from Firefox's CDN, which is provisioned for Firefox rather than third-party traffic. This CLI is a convenient way to try the engine; if you're building a product on top of it, use the [library](https://crates.io/crates/fxtranslate) and re-host the models you depend on rather than relying on Firefox's hosting.
