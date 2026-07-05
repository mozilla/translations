# fxtranslate-oracle

Internal, dev-only validation harness for the [fxtranslate](../fxtranslate) engine. Not published to crates.io (`publish = false`).

It validates the Rust engine against the reference marian C++ translator by comparing recorded execution traces, and provides a raw diagnostic binary for poking at the engine directly. It holds:

- **`compare`** — the within-tolerance float comparator the parity tests assert against.
- **`graph`** — replays a recorded reference trace as a forward pass, recomputing each node from its children to find the first point of divergence.
- **`tests/`** — op-level, int8, and trace-replay parity tests against tensors recorded from the reference engine.
- **the `fxtranslate-oracle` binary** — translate/encode text, or inspect and replay a trace.

```console
# Translate directly (greedy); vocab may be one shared .spm or split src/trg.
$ cargo run -- translate <model.bin> <vocab.spm> "Hello world."

# Inspect a recorded reference trace.
$ cargo run -- trace <trace-path> [num-records]
```

Usually driven through the repo's task runner (e.g. `task rs:translate -- en es --text "…"`), which resolves the model and vocab from a downloaded config. The `fast`, `jemalloc`, and `dhat-heap` features select the engine's compute path and allocator for the perf and heap-profiling runs.
