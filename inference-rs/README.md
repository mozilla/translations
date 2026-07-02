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
algorithm as the shipped WASM models, so this native build is the golden-trace oracle for
the Rust reimplementation. See [gemm-backends.md](./gemm-backends.md).
