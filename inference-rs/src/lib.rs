//! inference-rs: a Rust reimplementation of the Firefox Translations inference
//! engine, validated against the reference C++ engine.
//!
//! The pieces, following build-plan.md:
//! - [`trace`] reads the oracle produced by the C++ `TraceRecorder`, and
//!   [`compare`] provides the within-tolerance comparator parity tests assert
//!   against (step 2).
//! - [`model`] reads the marian binary model, giving the logical int8 weights
//!   the packed trace can't.
//! - [`ops`] holds the CPU ops — float, structural, gather, batched matmul, and
//!   the shifted int8 affine — each validated node-by-node against the trace
//!   (steps 3–4).
//! - [`graph`] replays the whole trace as a forward pass, recomputing each node
//!   from its children and finding the first divergence (step 5).

pub mod compare;
pub mod graph;
pub mod model;
pub mod ops;
pub mod spm;
pub mod trace;
pub mod weights;
