//! Marian-oracle validation harness for the `fxtranslate` engine.
//!
//! - [`compare`] is the within-tolerance float comparator the parity tests
//!   assert against.
//! - [`graph`] replays a recorded trace as a forward pass, recomputing each node
//!   from its children and finding the first divergence.

pub mod compare;
pub mod graph;
