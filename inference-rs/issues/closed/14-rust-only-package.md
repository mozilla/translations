# Rust-only `fxtranslate` Package

Probably sequenced at the end of this work. I want a fully-powered `fxtranslate` package built here that has batteries included. Basically if you cargo install it, I want to be able to do the following:

Enumerate the the list of models available in Remote Settings. Let's not worry about langauge display names as that sounds like a bigger dependency.

Download and manage the model files in a local cache of some kind.

Run translations by the stdout piping per-sentence inputs similar to how Marian already does it.

A fully interactive translator that works when you type in it. Enter sends in a translation, and then outputs the translation in your terminal. This is a minimal pretty and ergonomic CLI tool. I want this to be dependency free as possible, just as a nice "kick the tires" kind of feature.

DO NOT PUBLISH ANYTHING, but come up with a publish plan as a durable markdown artifact afterwards.

## Acceptance criteria

- Translation correctness rides on [02-parity-harness.md](./02-parity-harness.md) — the
  package doesn't re-prove the engine; it proves the *packaging* (model discovery,
  download+cache, CLI wiring) works.
- **Tests must not hit the network.** Record/mock the Remote Settings responses and a small
  fixture model so CI is offline and deterministic — a test that depends on a live CDN is
  neither cheat-proof nor reproducible.
- Cache management has explicit tests: cache hit skips download, corrupt/partial file is
  detected (hash) and re-fetched.
- The interactive REPL is a manual/smoke feature (hard to assert on ergonomics); keep its
  logic thin over the tested translate path so there's little untested surface.

## Open questions

- Package name `fxtranslate` — check crates.io availability before we lean on it.
- Remote Settings: which collection/endpoint enumerates models, and what's the offline
  fixture strategy (checked-in RS response snapshot)?
- Cache location convention (XDG dir? `~/.cache/fxtranslate`?) and how model versions/hashes
  key the cache.
- "As dependency-free as possible" vs. the network/TLS + progress-bar reality — where's the
  line? (e.g. is `reqwest`/`ureq` acceptable, or hand-rolled over `std`?)

## Done (commit pending) — `fxtranslate` crate + PUBLISHING.md

Implemented as a workspace member `inference-rs/fxtranslate/` (kept under inference-rs; its
network/CLI deps stay out of the lean engine crate). `cargo run -p fxtranslate` /
`task inference-rs:fxtranslate`:

- **Enumerate** models from Remote Settings (`translations-models-v2`): `fxtranslate list [prefix]`
  (50 en-* pairs live).
- **Download + cache**: verified (zstd-decode + sha256 vs `decompressedHash`) into
  `<platform-cache>/fxtranslate/models/<src>-<trg>/` (via the `dirs` crate), atomic writes, cache-hit skips download,
  corrupt/partial detected and re-fetched.
- **Translate**: one-shot args, stdin/pipe per-line (marian-style), or an interactive TTY REPL,
  over the inference-rs engine. Live check: `echo "The monster was created by a scientist." |
  fxtranslate en es` → "El monstruo fue creado por un científico."

**Acceptance criteria met**: correctness rides the engine/parity harness (not re-proven here);
**tests are offline** (a recorded RS snapshot + a tiny zstd fixture, driven through a mockable
`Http` trait — 9 tests, no network); cache hit / corrupt-refetch / hash-mismatch have explicit
tests; the REPL is a thin shell over the tested translate path.

**Open questions answered:** name `fxtranslate` is free on crates.io (2026-07-03); RS endpoint is
`.../collections/translations-models-v2/records`, offline fixture is a trimmed **real** snapshot;
cache is the platform-native cache dir via the `dirs` crate
(`<platform-cache>/fxtranslate/models/<src>-<trg>/`), verified by `decompressedHash`; the dep line
is `ureq`+`ruzstd`+`tinyjson`+`sha2`+`dirs` (rationale in PUBLISHING.md "Dependency budget").

**Not published** (`publish = false`); the publish plan is [../PUBLISHING.md](../PUBLISHING.md).
Remaining before publish (see that doc): vendored-C++-outside-crate, native-only default features,
path→versioned engine dep.
