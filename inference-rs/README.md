# inference-rs

Development workspace for the Rust reimplementation of the Firefox Translations inference engine,
validated against the existing C++ engine (`inference/build/src/app/translator-cli`).

This file is the internal orientation for working *in* the workspace. For using the engine or the
CLI — installation, the library API, performance — see the crate README:
[`crates/fxtranslate/README.md`](./crates/fxtranslate/README.md).

## Crates

Three crates under `crates/`:

- **[`fxtranslate`](./crates/fxtranslate)** — the engine library. Fast by default (native SIMD int8
  kernel where wired, portable scalar fallback everywhere else). Also carries optional model
  management (Remote Settings discovery + verified cache) behind the `download`/`net` features.
- **[`fxtranslate-cli`](./crates/fxtranslate-cli)** — the batteries-included CLI (binary named
  `fxtranslate`): discover → download/cache → translate. A thin shell over the engine.
- **[`fxtranslate-oracle`](./crates/fxtranslate-oracle)** — dev-only validation harness and raw
  diagnostic binary. The trace comparator, replay bisector, and the marian-oracle parity tests live
  here so none of it can leak into the shipped library or CLI. `publish = false`.

## Tasks

Tasks live in `Taskfile.yml`, included by the repo-root Taskfile under the `rs` namespace, so run
them from anywhere in the repo. `task rs:` lists them all; the ones you'll reach for:

| task | what it does |
|---|---|
| `task rs:download-model -- en es` | Fetch a model+vocab+lex by pair from Remote Settings into `data/models/<src><trg>/`. |
| `task rs:translate -- en es --text "…"` | Translate with the Rust engine (via the oracle diagnostic binary). |
| `task rs:translate-reference -- en es --text "…"` | Translate with the reference C++ `translator-cli` — the oracle. |
| `task rs:fxtranslate -- translate en es "…"` | Run the batteries-included CLI (its own model discovery/cache; `list` to enumerate pairs). |
| `task rs:parity` | Greedy exact-match rate of the Rust engine vs. `translator-cli` over a corpus. |
| `task rs:perf` | Engine perf (TTFT + tok/s), or record a samply profile. |
| `task rs:test` / `task rs:check` | Workspace tests / all checks incl. the C++ engine build. |
| `task rs:release` | Release build + binary-size characterization + artifact validation. |

## The oracle: how correctness is validated

The reference C++ engine is the source of truth. It builds and runs natively on Apple Silicon
(arm64) — no Docker:

```bash
task inference-build     # builds inference/build/src/app/translator-cli
```

On ARM that build uses the gemmology int8 backend, which runs the same `int8shiftAlphaAll` algorithm
as the shipped WASM models — so a native build is a faithful reference-trace oracle for the Rust
port. See [gemm-backends.md](./gemm-backends.md) for how the int8 backends line up across
architectures.

The Rust ops are validated two ways: op-level, against recorded intermediate tensors, and
end-to-end, against reference translations (`task rs:parity`).

### Recording a reference trace

The C++ engine can record every intermediate tensor of one translation. Pass `--trace` to
`translate-reference`:

```bash
# Writes artifacts/<src><trg>.trace (+ .trace.txt) by default. Keep --text short and
# --cpu-threads 1 so the trace stays compact and complete.
task rs:translate-reference -- en fr --text "Hello world." --cpu-threads 1 --trace
```

Each run writes two files: `<path>.trace` — the binary trace (one record per graph node in
forward-execution order: `{id, op, name, dtype, shape, child ids, raw bytes}`), consumed by the Rust
reader; and `<path>.trace.txt` — a human-readable manifest of the same nodes, shapes only, no tensor
data. The binary format is documented at the top of
[`inference/marian-fork/src/graph/trace_recorder.h`](../inference/marian-fork/src/graph/trace_recorder.h).

Traces land in `artifacts/` (gitignored) and are large — a single short sentence is ~170 MB, because
static model parameters are re-recorded on every decoding step. Under the hood the recorder keys off
the `MARIAN_TRACE` env var (a no-op for normal runs); `--trace` just sets it for you.

### Reading and comparing a trace (the Rust side)

- **Reader:** [`crates/fxtranslate/src/trace.rs`](./crates/fxtranslate/src/trace.rs) —
  `Trace::load(path)` parses a trace into per-node fixtures with typed views of the tensor bytes
  (`to_f32`/`to_i8`/`to_i32`) and `Trace::inputs(index)` to resolve a node's inputs by child id.
- **Comparator:** [`crates/fxtranslate-oracle/src/compare.rs`](./crates/fxtranslate-oracle/src/compare.rs)
  — `assert_close` / `compare_f32` assert two `f32` slices match within a tight rtol/atol.
- **Inspect from the CLI** (also a smoke check that the reader handles a real, full-size trace):

  ```bash
  cargo run -p fxtranslate-oracle -- trace artifacts/enfr.trace       # record count, op histogram
  cargo run -p fxtranslate-oracle -- trace artifacts/enfr.trace 20    # + first 20 records
  ```

The real-trace integration tests skip when no trace is present, so `task rs:test` passes without one.

## Further reading

- [`crates/fxtranslate/README.md`](./crates/fxtranslate/README.md) — engine + CLI usage, library API, performance.
- [gemm-backends.md](./gemm-backends.md) — the int8 GEMM backends and how they diverge per architecture.
- `notes/` — the design and build-out history (`01`–`10`): the parity bar and plan, the model
  architecture, the memory/perf approach, and the final comparisons.
