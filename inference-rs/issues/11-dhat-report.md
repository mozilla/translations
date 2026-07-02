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

## Spike done

Wired end-to-end:

- `dhat` is an **optional** dependency behind a `dhat-heap` cargo feature (Cargo.toml). Default
  builds carry no allocator override and no dhat dependency (`nm` shows 0 dhat symbols); it
  compiles in only under `--features dhat-heap`.
- `main.rs` installs `dhat::Alloc` as the global allocator and a `dhat::Profiler` (both
  `#[cfg(feature = "dhat-heap")]`) that writes its JSON to `$DHAT_OUT` on exit.
- `translate.py --memory-report` builds with the feature, points `DHAT_OUT` at
  `inference-rs/artifacts/dhat-<pair>.json` (gitignored), runs one translation, and prints a
  summary. The dhat JSON is the format the Firefox Profiler ingests.

Measured (en-fr, "Hello world."):

```
[memory] peak heap (t-gmax): 154,455,052 bytes
[memory] top site: 49,152,000 bytes @ inference_rs::weights::Weights::new (weights.rs:132)
[memory] OK: peak dominated by the Wemb table (as expected)
```

The top site is the dequantized `Wemb` clone (32000×384×4 = 49,152,000 bytes exactly), so
the non-cheatable sanity check ("Wemb is #1") is implemented by walking the top allocation
point's stack to the first `inference_rs` frame and asserting it's in `weights` — the summary
exits non-zero if that ever stops holding. Peak is read from dhat's authoritative `At t-gmax`
line (the JSON's per-site maxima don't sum to the global peak).

### Open question — answered

Report + the "Wemb is #1" sanity check, **no hard budget**. The peak is dominated by fixed
model tables (the ~147 MB is mostly the several dequantized-f32 embedding/clone copies — see
[16-wemb-dequant-memory.md](./16-wemb-dequant-memory.md)), so a byte budget would be brittle
across models and would really be re-asserting that known cost. The Wemb-dominates check
catches the meaningful regression (a new unexpected top site) without a magic number.
