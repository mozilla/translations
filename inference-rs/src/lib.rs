//! inference-rs: a Rust reimplementation of the Firefox Translations inference
//! engine, validated against the reference C++ engine.
//!
//! The current foundation (build-plan.md, step 2) is the reference-trace
//! tooling: [`trace`] reads the oracle produced by the C++ `TraceRecorder`, and
//! [`compare`] provides the within-tolerance comparator op-level parity tests
//! assert against.

pub mod compare;
pub mod model;
pub mod ops;
pub mod trace;
