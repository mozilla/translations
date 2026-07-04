//! `fxtranslate` — batteries-included CLI over the inference-rs engine.
//!
//! The library half holds the *packaging* logic (Remote Settings model discovery,
//! attachment download + verified cache) so it can be tested offline against
//! checked-in fixtures; `main.rs` wires it to the engine and the terminal.
//!
//! Translation correctness is the engine's job (validated against the marian
//! oracle in the parent crate); this crate only proves discovery, download+cache,
//! and CLI wiring — see `issues/14-rust-only-package.md`.

pub mod cache;
pub mod cli;
pub mod http;
pub mod lang;
pub mod remote;
