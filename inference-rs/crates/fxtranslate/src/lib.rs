//! fxtranslate: a Rust reimplementation of the Firefox Translations inference
//! engine, validated against the reference C++ engine.
//!
//! - [`trace`] reads the oracle produced by the C++ `TraceRecorder`.
//! - [`model`] reads the marian binary model, giving the logical int8 weights
//!   the packed trace can't.
//! - [`ops`] holds the CPU ops — float, structural, gather, batched matmul, and
//!   the shifted int8 affine.
//! - [`weights`] resolves model parameters; [`spm`] tokenizes; [`shortlist`]
//!   reads the lexical shortlist; [`engine`] runs the transformer and greedy decode.
//!
//! The trace-comparison harness (tolerance comparator, graph-replay bisector) and
//! the diagnostic binary live in the separate `fxtranslate-oracle` dev crate.

// Tolerance comparator, built only for `cargo test` so the ops unit tests can
// assert against it. The public comparison harness lives in `fxtranslate-oracle`.
#[cfg(test)]
mod compare;
pub mod engine;
/// FFI wrapper over the vendored gemmology i8mm SIMD kernel (`gemmology` feature).
#[cfg(feature = "gemmology")]
pub mod gemm;
pub mod model;
pub mod ops;
pub mod shortlist;
pub mod spm;
pub mod trace;
pub mod weights;
