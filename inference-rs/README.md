# inference-rs

A WIP Rust reimplementation of the Firefox Translations inference engine, validated against
the existing C++ engine (`inference/build/src/app/translator-cli`).

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

The engine builds and runs natively on Apple Silicon (arm64) — no Docker required. On ARM,
the build defaults to the gemmology int8 backend, which runs the same `int8shiftAlphaAll`
algorithm as the shipped WASM models, so this native build is the reference-trace oracle for
the Rust reimplementation. See [02-gemm-backends.md](./02-gemm-backends.md).

## Build configuration (Cargo features)

The **fast configuration is the default** and is what `cargo build`/`test` and all the harness
scripts use: `default = ["lean-embed", "gemmology", "jemalloc"]`. On the en→ru base model it runs at
**~0.96× native marian throughput** and is the **lightest of the three engines on memory** (~149 MiB
settled, vs marian ~298 and Firefox ~355). It is **native aarch64-only** (gemmology + jemalloc); build
the portable pure-Rust engine with `--no-default-features`. See
[08-perf-analysis.md](./08-perf-analysis.md), [09-final-comparison.md](./09-final-comparison.md), and
[10-decoder-optimizations.md](./10-decoder-optimizations.md).

| feature | default | what it does |
|---|:---:|---|
| `gemmology` | ✅ | Route the int8 affine through the vendored gemmology i8mm SIMD kernel (the same kernel marian uses) instead of the scalar Rust loop. Needs a C++17 toolchain + the gemmology/xsimd submodules; aarch64-only. Bit-identical to the scalar path (`tests/gemm_parity.rs`). |
| `lean-embed` | ✅ | Drop the resident dequantized-f32 embedding tables: dequantize embedding rows on demand and run the output projection full-vocab in int8. Large load/retained-memory win. |
| `jemalloc` | ✅ | Install the jemalloc global allocator — a page-returning allocator that takes settled RSS to the live-memory floor (~249 → ~149 MiB at ~0% throughput cost). Native-only; ceded to `dhat-heap` when both are set. Tune page return with `_RJEM_MALLOC_CONF`. |
| `instrumentation` | ❌ | Reference-trace reader, tolerance comparator, replay bisector, and the `replay`/`trace` subcommands. Required by the parity/oracle tests (`cargo test --features instrumentation`, wired into `task rs:test`). |
| `dhat-heap` | ❌ | Install dhat's heap-profiling allocator and write a dhat report on exit. Measurement only; exclusive with `jemalloc` (dhat wins the `#[global_allocator]`). |

Common invocations:

```bash
cargo build --release                                   # fast config (default)
cargo test --features instrumentation                   # fast config + parity/oracle tests
cargo build --release --no-default-features             # portable pure-Rust scalar engine (any arch / wasm)
cargo build --release --no-default-features --features lean-embed   # portable + the memory win, scalar kernel
cargo build --release --features dhat-heap              # heap profiling (jemalloc auto-ceded to dhat)
```

## `fxtranslate` — batteries-included CLI

The workspace member [`fxtranslate/`](./fxtranslate) is a self-contained CLI that discovers
Firefox Translations models from Remote Settings, downloads + verifies + caches them under the
platform-native cache dir + `/fxtranslate/models/<src>-<trg>/`, and translates via the engine
— no manual model wrangling. Not yet published (see [PUBLISHING.md](./PUBLISHING.md)); run it locally:

```bash
task rs:fxtranslate -- list en            # enumerate en-* model pairs
echo "Hello world." | task rs:fxtranslate -- en es
task rs:fxtranslate -- en es "Hello world."
task rs:fxtranslate -- en es              # interactive prompt
```

Its packaging logic (discovery, verified download/cache) has offline, deterministic tests
(`fxtranslate/tests/packaging.rs`) against checked-in fixtures; translation correctness rides the
engine's parity harness.

## Release build + validation

```bash
task rs:release                 # build, characterize size, validate the artifact
task rs:release -- --bloat      # + cargo bloat crate breakdown
task rs:release -- --skip-validation
```

Reports binary size and the default-vs-`--all-features` delta, and validates the *actual release
binary* is lean (instrumentation/dhat off) and a faithful build of the oracle-validated engine
(release output == debug output), reporting the `translator-cli` parity rate as a tracking metric.
See [issues/13-task-release-build.md](./issues/13-task-release-build.md).

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

The crate is a library plus a small CLI. The library ([`src/trace.rs`](./src/trace.rs) and
[`src/compare.rs`](./src/compare.rs)) is the foundation the op-level parity tests build on:

- `trace::Trace::load(path)` parses a trace into per-node fixtures (`TraceRecord`s) with typed
  views of the tensor bytes (`to_f32`, `to_i8`, `to_i32`) and `Trace::inputs(index)`, which
  resolves a node's input records by matching child ids to the most recent earlier record.
- `compare::assert_close(actual, expected, Tolerance::default())` asserts two `f32` slices
  match within a tight rtol/atol (the parity bar from [01-build-plan.md](./01-build-plan.md)), and
  `compare::compare_f32` returns the error statistics for programmatic use.

Inspect a recorded trace from the command line (record count, op histogram, first records) —
this doubles as a smoke check that the reader handles a real, full-size trace:

```bash
cargo run -- artifacts/enfr.trace          # from the inference-rs/ directory
cargo run -- artifacts/enfr.trace 20       # print the first 20 records
```

Run the Rust tests (the real-trace integration test skips when no trace is present):

```bash
task rs:test        # cargo test
task rs:lint-rust   # cargo fmt --check  (fix with :lint-rust-fix)
task rs:check       # all inference-rs checks + the C++ engine build
```
