# Use #[cfg(feature = "...")] to create a minimal binary

Here I want to opt-in to the Rust machinery to collect and validate instrumentation. When we built this and when we test this we need to have all kinds of tracing and internal measurement to validate things. I want a more production focused build by excluding things.

## Proposed feature taxonomy (for review)

Default = lean production; everything measurement-related is opt-in:

- **(default, no feature)** — the engine + translate path only.
- **`trace`** — the reference-trace comparison / `graph::replay` instrumentation and any
  timing hooks ([12-task-translate-logging.md](./12-task-translate-logging.md)).
- **`dhat`** — the allocator profiling ([11-dhat-report.md](./11-dhat-report.md)); pulls in
  `dhat-rs` only here.
- **`oracle`** — the parity/oracle test harnesses if they need extra deps (spm reference, etc.).

## Acceptance criteria (cheat-proof)

- Default `cargo build` does **not** link the opt-in crates — assert mechanically via
  `cargo tree --no-default-features` (or a symbol/`cargo bloat` check), not by eyeballing.
- A smoke translation still passes on the default build (feature-gating didn't remove
  needed code) — reuse a golden from [02-parity-harness.md](./02-parity-harness.md).
- Report the default-vs-full binary-size delta so the gating's payoff is visible (feeds
  [13-task-release-build.md](./13-task-release-build.md)).

## Open questions — resolved

- **Granularity: two features, not four.** The proposed `trace`/`dhat`/`oracle` split collapses
  to a **weight-based** one. Only `dhat` is heavy (external crate + `#[global_allocator]`
  override), so it keeps its own gate (`dhat-heap`). Everything else measurement-related is
  pure std Rust with no extra deps, so it groups under one general-purpose `instrumentation`
  flag. The two are **orthogonal and composable** — `--features instrumentation,dhat-heap` is
  the full measurement build; neither implies the other.
- **`oracle` tests are not a feature of their own.** They need the trace reader + comparator,
  which are exactly what `instrumentation` gates, so the parity/oracle test files carry
  `#![cfg(feature = "instrumentation")]` and `task rs:test` runs
  `cargo test --features instrumentation`. No dev-dependencies were needed.
- **`--timing` stays in the default build.** It is std-only operational telemetry (a latency
  span a translation server would legitimately want), not oracle tooling — so it is *not*
  behind `instrumentation`, unlike the trace/replay machinery the taxonomy lumped with it.

## Status: feature split done

`instrumentation` (default off) gates the oracle/trace-comparison surface: the `trace::Trace`
reader + `TraceRecord`, the `compare` tolerance comparator, the `graph::replay` bisector, and
the `replay`/`trace` diagnostic subcommands. `dhat-heap` (default off) is unchanged.

Verified:

- **Default (lean) build** (`lean-embed,gemmology`) compiles warning-clean, translates
  correctly (`"Hello world." → "Bonjour le monde."`), and its `usage` lists only
  `translate`/`encode` — `replay`/`trace` are compiled out.
- **`--features instrumentation`** restores the subcommands; the full parity/oracle suite
  (44 unit + integration tests) compiles and passes. Bare `cargo test` (no feature) compiles
  the gated test files as empty and still builds clean.
- **`instrumentation,dhat-heap`** together build clean (orthogonality confirmed).

### Residual → 13-task-release-build.md

Two acceptance items are reporting/measurement rather than gating, and belong with the release
task:

- Mechanical assertion that the default build links no opt-in **crate** (the `dhat` crate is
  the only one; issue 11 already checks this informally with `nm`). Wire a
  `cargo tree`/symbol check into the release validation.
- The default-vs-full **binary-size delta** report (the payoff number). This is
  [13-task-release-build.md](./13-task-release-build.md)'s `cargo bloat`/size characterization.

Pre-existing, unrelated: `cargo build --no-default-features` warns `unused_mut` at
`weights.rs:251` (the `mut model` is only used when `lean-embed`/`gemmology` drain tensors) —
not touched here.
