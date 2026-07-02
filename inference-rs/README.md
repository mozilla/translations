# inference-rs

A WIP Rust reimplementation of the Firefox Translations inference engine, validated against
the existing C++ engine (`inference/build/src/app/translator-cli`).

## Tasks

Tasks live in `Taskfile.yml` and are included by the repo-root Taskfile under the
`inference-rs` namespace (see
[Including other Taskfiles](https://taskfile.dev/docs/guide#including-other-taskfiles)),
so run them from anywhere in the repo:

```bash
# List the tasks
task inference-rs:

# Download a production model by language pair (source target).
# Fetches model/vocab/lex from the Firefox Remote Settings CDN, decompresses, verifies
# hashes, and writes a translator-cli decode config to data/models/<src><trg>/.
task inference-rs:download-model -- en es

# Run the reference C++ engine (defaults to en->es, "Hello" -> "Hola").
task inference-rs:translate-reference
task inference-rs:translate-reference -- en es --text "Hello World"
```

`translate-reference` requires the C++ engine to be built first:

```bash
task inference-build
```

The engine builds and runs natively on Apple Silicon (arm64) — no Docker required. On ARM,
the build defaults to the gemmology int8 backend, which runs the same `int8shiftAlphaAll`
algorithm as the shipped WASM models, so this native build is the reference-trace oracle for
the Rust reimplementation. See [gemm-backends.md](./gemm-backends.md).

## Recording a reference trace

The C++ engine can record every intermediate tensor of one translation — the parity oracle
the Rust ops are validated against (see [build-plan.md](./build-plan.md)). Pass `--trace` to
`translate-reference`:

```bash
# Writes inference-rs/artifacts/<src><trg>.trace by default.
# Use a short --text and --cpu-threads 1 to keep the trace compact and complete.
task inference-rs:translate-reference -- en fr --text "Hello world." --cpu-threads 1 --trace

# Or choose an explicit path:
task inference-rs:translate-reference -- en fr --text "Hello world." --trace /tmp/my.trace
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
