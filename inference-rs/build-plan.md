## The build plan

I want to create a more memory safe and rigorous implementation of the marian-fork that
can run the specially quantized Firefox models with `int8shiftAlphaAll` mode. We have a
custom fork of the Marian translation system to run CPU-optimized models.

In the original inference engine there is the
[ExpressionGraph](../inference/marian-fork/src/graph/expression_graph.h) class which builds
up a graph of operations, then executes it. I want to build something with parity in Rust,
with more memory-safe conventions, replicating `ExpressionGraph` incrementally and
validating each piece against the C++ reference.

## Validation strategy

The whole approach hinges on being able to extract the reference engine's internal states
cheaply. Marian already gives us the hook: in `ExpressionGraph::forward()`
(`expression_graph.cpp:114`) every node runs `v->forward()` in topological order, and right
after there is a `marked_for_debug()` branch that calls `v->val()->debug()`.

So the mechanism is small and contained: record each tensor at that point, writing, for
**every** node, a record keyed by node id → `{type, name, shape, value_type, raw bytes}`.
One reference translation then produces a complete **reference trace** of the entire graph —
every intermediate tensor, in execution order — that becomes the oracle for all the Rust
work.

That gives us two validation granularities:

- **Op-level (unit).** For each CPU node op, pull its exact input tensors + output from the
  trace, feed the inputs to the Rust op, and assert the output matches. No graph, no
  ordering — pure functions with golden fixtures. This catches the large majority of parity
  bugs and each op is an independently verifiable unit.
- **Graph-level (integration).** Replay the whole graph in Rust, compare node-by-node
  against the trace, and find the first divergence. This is the "work forward through the
  graph" part, and it becomes a sequencing concern once the ops themselves are trusted.

### Parity bar

Comparisons use a **tight numerical tolerance** (rtol/atol within a small epsilon), not
bit-exactness. The int8 GEMM path and threaded reductions make exact matching brittle, and
tolerance is robust to reduction-order differences. We keep epsilon tight enough that real
bugs still surface.

## The `int8shiftAlphaAll` op surface

The reference `translator-cli` is built and run **natively on Apple Silicon (arm64)** — no
Docker — with the **gemmology** int8 backend (the default on ARM; `task inference-build`).
gemmology reimplements intgemm's kernels on top of xsimd/NEON and preserves intgemm's
exact `int8shiftAlphaAll` numerics (within reduction-order tolerance), so it exercises the
same quantized code path (`intgemmPrepareA/B`, `intgemmAffine`, …) as the shipped WASM
models — not the default Ruy (ARM) path or plain wasm. That makes the native gemmology
build the reference-trace oracle. See [gemm-backends.md](./gemm-backends.md).

`int8shiftAlphaAll` decomposes into four backend flags — `int8 + shifted + shiftedAll +
precomputedAlpha` (see `tensors/cpu/backend.h`) — which light up this set of node ops:

- **`intgemmPrepareA`** — quantize activations to int8, *shifted* variant (add 127 → unsigned
  domain).
- **`intgemmPrepareB`** — quantize + tile-pack weights. Normally precomputed offline and
  shipped in the model.
- **`intgemmSelectColumnsB`** — gather output columns (vocab selection for the final
  projection).
- **`QuantMultNodeOp`** — the quantization multipliers; *precomputedAlpha* means A's
  multipliers ship in the model rather than being derived at runtime.
- The **Multiply / affine** op — the shifted int8 GEMM plus the alpha/bias *correction term*
  that cancels the +127 shift ("All" = apply even to bias-less matmuls).

**Validation boundary for the int8 path:** `PrepareB`'s output is an intgemm-specific
tiled/packed byte blob. Comparing it tensor-for-tensor against a Rust implementation is
painful and buys nothing, since the packing layout is an internal detail. So we treat
prepare → multiply → correct as **one validation unit** and compare the float GEMM output
within tolerance, rather than matching the opaque packed weights byte-for-byte. The
clean-float intermediates (layernorm, softmax, attention scores, elementwise) are where
per-node comparison is both easy and valuable.

## Build order

1. **Reference-trace recorder** (C++) — hook at `expression_graph.cpp:114`, record every
   node's `{id, type, name, shape, dtype, bytes}` in execution order to a trace file. One
   reference run = complete oracle.
2. **Rust trace reader + tolerance comparator** — load a trace, expose per-node fixtures,
   and provide an assert-within-epsilon helper.
3. **Op-level parity, float ops first** — layernorm, softmax, elementwise, plain affine.
   Each is a pure function tested against its fixture. Builds out the tensor/shape plumbing.
4. **The int8 GEMM unit** — shifted quantize + multiply + alpha correction, validated at the
   float-output boundary.
5. **Graph assembly** — wire ops in execution order, self-attention block first, diff
   node-by-node against the trace to find the first divergence, then walk forward to full
   end-to-end translation.

Steps 1–2 are the foundation everything else leans on, and they are small and
self-contained.

## Working mode

Interactive and step by step — no orchestrated agent fan-out. The first concrete step is
the reference-trace recorder, since it unblocks all of the Rust work.
