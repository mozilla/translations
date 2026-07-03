# Quantify Performance

I want a test harness that can run through a somewhat realistic corpus to measure how fast we compared to the baseline implementation. It should collect time to first token and tokens per second. Care should be taken to statistically reproduct the results. I don't know that we need any big dependencies added here. This should be doable as some vanilla code. I'm also not opinionated on python + rust or just rust here.

## Fair-comparison methodology (so the numbers mean something)

A perf number is only honest if the two engines do the same work on the same hardware:

- **Same corpus, same sentences, pre-split** (reuse the parity corpus so tokenization/length
  are identical). Report per-sentence, not one blended number.
- **Match `translator-cli` config** — crucially `--cpu-threads`. Our engine is single-threaded
  today, so either pin the baseline to 1 thread or clearly label the thread counts; don't
  compare 1-thread Rust to N-thread Marian and call it a regression.
- **Warmup + repeat**: discard the first run (build, page cache), then N runs (≥10). Report
  the **median** (middle run) and **IQR** (interquartile range — the 25th–75th percentile
  spread, i.e. the band the middle half of runs fell in) / min, not mean ± stddev —
  wall-clock is right-skewed (slow outliers, never faster-than-hardware), so the mean is
  dragged up and stddev overstates spread. Exclude the `cargo` build entirely (pre-build,
  or measure the binary directly).
- Metrics: **TTFT** (time to first token — encode + first decode step) and **steady-state
  tok/s** (generated tokens ÷ decode time), kept separate (TTFT is encode-dominated, tok/s
  is decode-dominated — they regress for different reasons).

## Example output (shape, for review)

```
corpus: flores200-devtest (200 sent) | machine: M-series, 1 thread | 10 runs, warmup 1
                     TTFT (ms)         tok/s
                  median   IQR      median   IQR
inference-rs        30     28–34     43     41–45
translator-cli      22     21–24     61     58–63
ratio               1.4x            0.70x
```

(median = middle run; IQR = interquartile range, the 25th–75th percentile spread /
middle-half band; TTFT = time to first token; tok/s = steady-state decode throughput.)

## Second run: direct comparison against production

There's a full e2e production benchmark at `~/dev/taskcluster-tools/src/benchmark/`
(`index.html`) — a "Translations Benchmark" page that translates a Spanish document
(Don Quixote) es→en in Firefox's production Wasm pipeline and reports **words-per-second**
and **tokens-per-second**. **The human runs this** (it needs Firefox); the harness can't.

So this issue produces two comparisons:

1. **Local A/B** (above): inference-rs vs. `translator-cli`, for fast iteration.
2. **Prod comparison** (human-assisted): run inference-rs (es→en model) over the *same*
   benchmark document and report **wps + tps in the same units** as that page, so the
   number can be set side-by-side with what the production page shows in Firefox. Use the
   benchmark's own input text (and `enes.spm`) so tokenization/word-count line up. The
   harness emits our number; the human reads the prod number off the page.

## Acceptance criteria

- Reproducible: two runs on the same machine agree within the reported IQR (interquartile
  range — the 25th–75th percentile spread).
- Emits wps/tps in the production benchmark's units for the prod-comparison run.
- **Tracking, not a gate** (per the "quantify parity/perf" goal — no regression threshold
  for now). Cheat-proof property is the apples-to-apples setup, not a target number.

## Decided

- **Tracking only**, no regression gate yet.
- Baseline: **pin `translator-cli` to 1 thread** to match our single-threaded engine, and
  label it; revisit if/when we parallelize.
- **Python driver** (orchestrates both binaries + stats) consuming the CLI `--timing`
  output ([12-task-translate-logging.md](./12-task-translate-logging.md)); no big deps.

## Status: local harness built

`scripts/perf.py` (`task inference-rs:perf`), stdlib-only:

- **timing mode** — pre-builds the release binary (build excluded from timing), translates a
  corpus `--runs` times single-threaded, parses the CLI's per-sentence `[timing]` spans
  (`--timing`, added to `translate`), reports the **median and IQR** (interquartile range —
  the 25th–75th percentile spread) of **TTFT** (time to first token) and **steady-state
  tok/s**.
  Compare build variants with `--features` (e.g. `lean-embed`).
- **`--samply` mode** — records a Firefox Profiler `.json.gz` under `artifacts/` (corpus
  repeated `--samply-loops` times for enough samples), for flame-graph / call-tree work.

First use (en-fr, dev-en, 10 runs, 1 thread) answered the [lean-embed](../06-memory-approach.md)
perf question:

```
                default        lean-embed
TTFT (ms)         7.1             4.0        (-44%)
tok/s           211.6           825.0        (+290%)
```

lean-embed is faster *and* lighter — the output projection is memory-bound on reading the
whole embedding table each step, and int8 is ¼ the bytes of the dequantized f32.

Still **not built**: the translator-cli local A/B (fair 1-thread wall-clock) and the
human-assisted prod-benchmark comparison — the `--timing` plumbing and stats are in place to
add them.
