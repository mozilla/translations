# inference-rs

A WIP Rust reimplementation of the Firefox Translations inference engine, validated against
the existing C++ engine (`inference/build/src/app/translator-cli`).

The workspace has three crates under `crates/`:

- **`fxtranslate`** — the engine library (portable by default; native SIMD via `--features fast`).
- **`fxtranslate-cli`** — the batteries-included CLI (binary named `fxtranslate`).
- **`fxtranslate-oracle`** — the marian-oracle validation harness and raw diagnostic binary (dev-only).

## Tasks

Tasks live in `Taskfile.yml` and are included by the repo-root Taskfile under the
`rs` namespace (see
[Including other Taskfiles](https://taskfile.dev/docs/guide#including-other-taskfiles)),
so run them from anywhere in the repo:

```bash
# List the tasks
task rs:

# Download a production model by language pair (source target).
# Fetches model/vocab/lex from the Firefox Remote Settings CDN, decompresses, verifies
# hashes, and writes a translator-cli decode config to data/models/<src><trg>/.
task rs:download-model -- en es

# Run the reference C++ engine (defaults to en->es, "Hello" -> "Hola").
task rs:translate-reference
task rs:translate-reference -- en es --text "Hello World"
```

`translate-reference` requires the C++ engine to be built first:

```bash
task inference-build
```

The C++ engine builds and runs natively on Apple Silicon (arm64) — no Docker required. On ARM,
that build uses the gemmology int8 backend, which runs the same `int8shiftAlphaAll` algorithm as
the shipped WASM models, so this native build is the reference-trace oracle for the Rust
reimplementation. See [02-gemm-backends.md](./02-gemm-backends.md).

## Build configuration (Cargo features)

The engine (`fxtranslate`) is **portable pure-Rust scalar by default**, so it builds on any
target (x86, wasm, non-aarch64) with no C++ toolchain. The native, production-shaped speed is
opt-in via `--features fast`. On the en→ru base model the fast config runs at **~0.96× native
marian throughput** (see [08-perf-analysis.md](./08-perf-analysis.md),
[09-final-comparison.md](./09-final-comparison.md), and
[10-decoder-optimizations.md](./10-decoder-optimizations.md)).

| feature (crate) | what it does |
|---|---|
| `fast` = `lean-embed` + `gemmology` (`fxtranslate`) | The native production config; aarch64 + C++17 only (see the two rows below). |
| `lean-embed` (`fxtranslate`) | Drop the resident dequantized-f32 embedding tables: dequantize embedding rows on demand and run the output projection full-vocab in int8. Large load/retained-memory win; pure Rust (portable). |
| `gemmology` (`fxtranslate`) | Route the int8 affine through the vendored gemmology i8mm SIMD kernel (the same kernel marian uses) instead of the scalar Rust loop. Needs a C++17 toolchain + the vendored gemmology/xsimd headers; aarch64-only. Bit-identical to the scalar path (`tests/gemm_parity.rs`). |
| `jemalloc` / `dhat-heap` (`fxtranslate-oracle`) | Global-allocator choices for the diagnostic binary — page-returning jemalloc (settled RSS ~249 → ~149 MiB at ~0% throughput cost) or dhat's heap profiler. A binary concern, so they live in the dev crate, not the library. |

The marian-oracle harness (trace comparator, replay bisector, the raw diagnostic binary, and the
parity tests) lives in the separate `fxtranslate-oracle` dev crate; the engine library carries
none of it.

Common invocations:

```bash
cargo build -p fxtranslate                         # portable pure-Rust scalar engine (any arch / wasm)
cargo build -p fxtranslate --features fast         # native aarch64 config (gemmology SIMD)
cargo test  -p fxtranslate --features fast         # engine tests incl. the gemmology parity test
cargo test  -p fxtranslate-oracle                  # marian-oracle parity tests (skip without a trace)
cargo run   -p fxtranslate-oracle --features fast,dhat-heap -- translate …   # heap profiling
```

## `fxtranslate` — batteries-included CLI

The [`fxtranslate-cli`](./crates/fxtranslate-cli) crate is a self-contained CLI (binary named
`fxtranslate`) that discovers Firefox Translations models from Remote Settings, downloads +
verifies + caches them under the platform-native cache dir + `/fxtranslate/models/<src>-<trg>/`,
and translates via the engine — no manual model wrangling. Not yet published (see
[PUBLISHING.md](./PUBLISHING.md)); run it locally:

```bash
task rs:fxtranslate -- list es            # model pairs for a language (both directions)
echo "Hello world." | task rs:fxtranslate -- en es
task rs:fxtranslate -- en es "Hello world."
task rs:fxtranslate -- en es              # interactive prompt
```

Its packaging logic (discovery, verified download/cache) has offline, deterministic tests
(`crates/fxtranslate-cli/tests/packaging.rs`) against checked-in fixtures; translation correctness
rides the engine's parity harness.

## Release build + validation

```bash
task rs:release                 # build, characterize size, validate the artifacts
task rs:release -- --bloat      # + cargo bloat crate breakdown
task rs:release -- --skip-validation
```

Reports the size of the shippable CLI and the oracle binary, then validates that the product CLI
is lean (the trace/replay diagnostics and dhat live in a separate crate, so they cannot leak into
it) and that the release engine is a faithful build of the oracle-validated engine (release output
== debug output), reporting the `translator-cli` parity rate as a tracking metric. See
[issues/13-task-release-build.md](./issues/13-task-release-build.md).

## Recording a reference trace

The C++ engine can record every intermediate tensor of one translation — the parity oracle
the Rust ops are validated against (see [01-build-plan.md](./01-build-plan.md)). Pass `--trace` to
`translate-reference`:

```bash
# Writes inference-rs/artifacts/<src><trg>.trace by default.
# Use a short --text and --cpu-threads 1 to keep the trace compact and complete.
task rs:translate-reference -- en fr --text "Hello world." --cpu-threads 1 --trace

# Or choose an explicit path:
task rs:translate-reference -- en fr --text "Hello world." --trace /tmp/my.trace
```

Each run writes two files next to each other:

- `<path>.trace` — the binary trace consumed by the Rust reader: one record per graph node
  in forward-execution order, each `{id, op type, name, dtype, shape, child ids, raw bytes}`.
  The format is documented at the top of
  [`inference/marian-fork/src/graph/trace_recorder.h`](../inference/marian-fork/src/graph/trace_recorder.h).
- `<path>.trace.txt` — a human-readable manifest of the same nodes with shapes but **no**
  tensor data, for eyeballing the graph without a parser.

Traces land in `inference-rs/artifacts/` (gitignored). A single short sentence produces a
large trace (~170 MB) because static model parameters are re-recorded on each decoding step.

Under the hood the recorder is driven purely by the `MARIAN_TRACE` environment variable, so
it stays a no-op for normal runs; `--trace` just sets it for you. To record a trace from a
direct `translator-cli` invocation, set `MARIAN_TRACE=<path>` in the environment yourself.

## Reading a trace (the Rust side)

The trace reader lives in the engine ([`crates/fxtranslate/src/trace.rs`](./crates/fxtranslate/src/trace.rs)),
and the tolerance comparator in the oracle crate
([`crates/fxtranslate-oracle/src/compare.rs`](./crates/fxtranslate-oracle/src/compare.rs)) — together
they are the foundation the op-level parity tests build on:

- `fxtranslate::trace::Trace::load(path)` parses a trace into per-node fixtures (`TraceRecord`s) with
  typed views of the tensor bytes (`to_f32`, `to_i8`, `to_i32`) and `Trace::inputs(index)`, which
  resolves a node's input records by matching child ids to the most recent earlier record.
- `fxtranslate_oracle::compare::assert_close(actual, expected, Tolerance::default())` asserts two
  `f32` slices match within a tight rtol/atol (the parity bar from
  [01-build-plan.md](./01-build-plan.md)), and `compare::compare_f32` returns the error statistics.

Inspect a recorded trace from the command line (record count, op histogram, first records) —
this doubles as a smoke check that the reader handles a real, full-size trace:

```bash
cargo run -p fxtranslate-oracle -- trace artifacts/enfr.trace       # from the repo root
cargo run -p fxtranslate-oracle -- trace artifacts/enfr.trace 20    # print the first 20 records
```

Run the tests (the real-trace integration tests skip when no trace is present):

```bash
task rs:test        # cargo test across the three crates
task rs:lint-rust   # cargo fmt --check  (fix with :lint-rust-fix)
task rs:check       # all inference-rs checks + the C++ engine build
```
