//! `fxtranslate` — batteries-included CLI over the fxtranslate engine.
//!
//! The library half holds the *packaging* logic (Remote Settings model discovery,
//! attachment download + verified cache) plus the whole CLI surface — `parse`,
//! the `list` renderer, and `run` (parse → dispatch → execute over injected
//! `Fetch`/`Translator` and I/O) — so it can all be driven offline against
//! checked-in fixtures and fakes. `main.rs` is a thin shim that wires the real
//! network, engine, and terminal into `cli::run`.
//!
//! Translation correctness is the engine's job (validated against the marian
//! oracle in the parent crate); this crate only proves discovery, download+cache,
//! and CLI wiring — see `issues/14-rust-only-package.md`.

pub mod cache;
pub mod cli;
pub mod fetch;
pub mod lang;
pub mod remote;
pub mod translate;
