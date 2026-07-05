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

The engine (`fxtranslate`) is **fast by default**, and still builds and runs everywhere. The
default requests the SIMD int8 kernel (gemmology); `build.rs` compiles it for the target — aarch64
(i8mm) and x86_64 (AVX2) today, with a C++17 toolchain — and otherwise prints a warning and falls
back to the portable pure-Rust scalar kernel *without failing the build*. So `cargo install` works
on any target; it's just faster where a kernel is wired. On the en→ru base model the SIMD path runs
at **~0.96× native marian throughput** (see [08-perf-analysis.md](./notes/08-perf-analysis.md),
[09-final-comparison.md](./notes/09-final-comparison.md), and
[10-decoder-optimizations.md](./notes/10-decoder-optimizations.md)).

The AVX2 kernel matches the exact scalar/i8mm kernels in the normal quantized range but is a
saturating approximation on extreme inputs (as is Firefox's WASM engine); the faster, exact x86
path (AVX-VNNI) is not wired yet. See [gemm-backends.md](./gemm-backends.md) for the full backend
map and the remaining x86 work.

| feature (crate) | what it does |
|---|---|
| `fast` = `lean-embed` + `gemmology` (`fxtranslate`) | **On by default.** The native production config, degrading to scalar where no SIMD kernel is built. |
| `portable` (`fxtranslate`) | Force the scalar kernel: no SIMD, no C++ toolchain, even with `fast` on. Opt out of the default without `--no-default-features` (e.g. `cargo build --features portable`). |
| `lean-embed` (`fxtranslate`) | Drop the resident dequantized-f32 embedding tables: dequantize embedding rows on demand and run the output projection full-vocab in int8. Large load/retained-memory win; pure Rust (portable). |
| `gemmology` (`fxtranslate`) | Request the vendored gemmology SIMD kernel for the int8 affine instead of the scalar Rust loop. build.rs compiles it where a kernel exists (aarch64 i8mm, x86_64 AVX2; C++17 toolchain) and emits `--cfg gemmology_simd`; otherwise the scalar kernel is used. Exact vs. scalar on ARM/VNNI; AVX2 matches in the quantized range but saturates on extremes — see [gemm-backends.md](./gemm-backends.md) and `tests/gemm_parity.rs`. |
| `jemalloc` / `dhat-heap` (`fxtranslate-oracle`) | Global-allocator choices for the diagnostic binary — page-returning jemalloc (settled RSS ~249 → ~149 MiB at ~0% throughput cost) or dhat's heap profiler. A binary concern, so they live in the dev crate, not the library. |

The marian-oracle harness (trace comparator, replay bisector, the raw diagnostic binary, and the
parity tests) lives in the separate `fxtranslate-oracle` dev crate; the engine library carries
none of it.

Common invocations:

```bash
cargo build -p fxtranslate                         # fast by default (SIMD where wired, else scalar)
cargo build -p fxtranslate --features portable     # force the portable scalar kernel (no C++)
cargo build -p fxtranslate --no-default-features    # bare scalar (also drops lean-embed)
cargo test  -p fxtranslate                         # engine tests (incl. the gemmology parity test on aarch64)
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
== debug output), reporting the `translator-cli` parity rate as a tracking metric.

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
