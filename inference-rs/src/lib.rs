//! inference-rs: a Rust reimplementation of the Firefox Translations inference
//! engine, validated against the reference C++ engine.
//!
//! - [`trace`] reads the oracle produced by the C++ `TraceRecorder`, and
//!   [`compare`] provides the within-tolerance comparator parity tests assert against.
//! - [`model`] reads the marian binary model, giving the logical int8 weights
//!   the packed trace can't.
//! - [`ops`] holds the CPU ops — float, structural, gather, batched matmul, and
//!   the shifted int8 affine.
//! - [`weights`] resolves model parameters; [`spm`] tokenizes; [`shortlist`]
//!   reads the lexical shortlist; [`engine`] runs the transformer and greedy decode.
//! - [`graph`] replays a recorded trace as a forward pass, recomputing each node
//!   from its children and finding the first divergence.

/// Tolerance comparator for parity checks (`instrumentation` feature; also built
/// for `cargo test` so unit tests can assert against it).
#[cfg(any(feature = "instrumentation", test))]
pub mod compare;
pub mod engine;
/// FFI wrapper over the vendored gemmology i8mm SIMD kernel (`gemmology` feature).
#[cfg(feature = "gemmology")]
pub mod gemm;
/// Trace-replay bisector: recompute a recorded trace to find the first
/// divergence (`instrumentation` feature).
#[cfg(any(feature = "instrumentation", test))]
pub mod graph;
pub mod model;
pub mod ops;
/// Capacity-keyed scratch pool for reusable `f32` activation buffers.
pub mod pool;
pub mod shortlist;
pub mod spm;
pub mod trace;
pub mod weights;
