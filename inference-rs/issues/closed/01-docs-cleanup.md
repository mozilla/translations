# Docs Cleanup

For high quality code generation, we need human-focused documentation. Your generative loop for building out the original files left comment artifacts strewn across.

## What to remove:

Tautological comments - Comments that repeat the implementation details and don't really add value. Comments should explain why things happen, not just repeat details. I didn't see a ton of these in the code comments, but it's something to keep an eye out for. I'm less concerned here for bigger comment rewriting but have it loaded in your context that this is important.

Self talk, and process comments - We don't need durable comments about _how_ this project was built. There is a lot of referencing of code gen process. Anytime it references a 01-*.md markdown file, this should be a red flag for something to clean up. It's fine to reference future clean-ups or design directions with TODO and a reference link. It's also OK to reference oracle implementations.

## Follow-up issue:

This should be sequenced at the top of the agentic workflow, but I want a new issue for you to file that handles this as a final validation step of an agentic flow.

## Examples (for review)

Concrete instances found in the current tree — the pattern, not an exhaustive list.

**Process references to the numbered design docs** (~20 across `src/` and `tests/`). Every
module doc opens by citing where in the build it came from:

- `src/ops.rs:2` — `//! (01-build-plan.md, step 3: "op-level parity, float ops first").`
- `src/lib.rs:4` — `//! The pieces, following 01-build-plan.md:`
- `src/graph.rs:1` — `//! Graph-level replay (01-build-plan.md step 5).`
- `tests/ops_parity.rs:171` — `// --- Structural ops (01-build-plan.md step 3b) ---`

These are the clearest red flags. A reader of `ops.rs` doesn't care which planning step
produced it. Mechanical to find: `grep -rnE '0[0-9]-[a-z-]+\.md|step [0-9]' src tests`.

**Self-talk about the build narrative:**

- `src/ops.rs:7` — `//! ... That closes the loop the whole validation strategy rests on:
  real reference data in, Rust op out, compared against the oracle.`

**Before → after** (the `ops.rs` header):

```rust
// before
//! CPU op implementations, validated against the reference trace
//! (01-build-plan.md, step 3: "op-level parity, float ops first").
//! ... The parity harness (`tests/ops_parity.rs`) pulls a node's exact
//! input and output tensors ... That closes the loop the whole
//! validation strategy rests on: real reference data in, Rust op out ...

// after
//! CPU op implementations. Each op is a pure function over row-major `f32`
//! slices; correctness is checked in `tests/ops_parity.rs` against recorded
//! reference tensors.
```

**Borderline — keep:** `src/engine.rs:5` "no trace involved" reads like process but is a
real behavioral fact (the engine builds the forward pass from weights, not from a trace).
Rewrite for tone, don't delete the fact. Pointers to oracle impls and `TODO(link)` to
design docs stay, per your note.

## Acceptance criteria

- Mechanical gate: `grep -rnE '0[0-9]-[a-z-]+\.md|step [0-9]' src tests` returns nothing
  (can wire into `scripts/check.py`). This part is cheat-proof — it either matches or not.
- Human review for tautology/tone — inherently not mechanizable; a reviewer pass on the
  diff, which is why it's a *final* validation step rather than a CI blocker.

## Follow-up (filed)

The recurring "run this as the final validation step of every agentic flow" practice is
[17-agentic-docs-validation.md](./17-agentic-docs-validation.md), sequenced at the end. This
issue stays the one-time cleanup that gives it a clean baseline to start from.
