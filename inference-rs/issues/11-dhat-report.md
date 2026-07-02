# dhat report

https://github.com/nnethercote/dhat-rs

I want some dhat reports built to understand where memory is being allocated. We can save these out to inference-rs/artifacts/dhat* of some kind. Ultimately I use the Firefox Profiler to visualize these, but this is a human activity. From your side we can collect and report on things, and then let a human see the results. Let's do this as a `--memory-report` flag for inference-rs/scripts/translate.py that accepts no args, and a cfg feature to keep compiles small and targeted

## Acceptance criteria

- `--memory-report` produces a machine-readable dhat file under `inference-rs/artifacts/`
  (dhat's JSON, which the Firefox Profiler already ingests) plus a short human summary line
  (peak bytes, top allocation site).
- Primarily informational — the human reads it. One cheap non-cheatable assertion worth
  adding: peak heap is dominated by the expected `Wemb` table
  ([16-wemb-dequant-memory.md](./16-wemb-dequant-memory.md)); if that ever stops being the
  top site, something changed and the report should surface it.
- The `dhat` cfg feature ([09-cfg-feature-instrumentation.md](./09-cfg-feature-instrumentation.md))
  is off by default so it doesn't bloat normal builds.

## Open question

- A memory **budget** to assert against, or purely a report + the "Wemb is #1" sanity check?
