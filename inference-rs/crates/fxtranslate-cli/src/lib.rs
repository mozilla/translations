//! `fxtranslate` — batteries-included CLI over the fxtranslate engine.
//!
//! The library half holds the whole CLI surface — `parse`, the `list` renderer,
//! and `run` (parse → dispatch → execute over injected `Fetch`/`Translator` and
//! I/O) — so it can all be driven offline against fakes. `main.rs` is a thin shim
//! that wires the real network, engine, and terminal into `cli::run`.
//!
//! Model management (Remote Settings discovery, attachment download + verified
//! cache, the pluggable `Fetch` client, and the `src→trg`→engine loader) now lives
//! in the `fxtranslate` engine library under its `download`/`net` features; this
//! crate consumes it (`fxtranslate::{remote,cache,fetch,lang,loader}`) and only
//! adds the terminal CLI on top. Translation correctness is the engine's job
//! (validated against the marian oracle); this crate's tests only cover the CLI.

pub mod cli;
pub mod translate;
