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

## Open questions

- Feature names/granularity above — right split, or too fine?
- Should the oracle tests be a feature, or just `#[cfg(test)]` + dev-dependencies (which
  already don't affect the release binary)?
