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
