# Measuring and fixing memory: dhat, the Firefox Profiler, and profiler-cli

This is the story of how a load-time memory problem in `inference-rs` was measured,
understood, and fixed — and, more interestingly, how an agent drove the whole loop by
reading a heap profile the same way a human would in the Firefox Profiler.

- **Profiler (before):** https://share.firefox.dev/4gk7nAb
- **Profiler (after):** https://share.firefox.dev/4ffxHdJ

The result: shared-vocab peak heap dropped **154.5 MB → 89.4 MB (−42%)**, output
byte-identical, all tests green.

## 1. Measuring: dhat-rs as a heap recorder

[dhat-rs](https://github.com/nnethercote/dhat-rs) turns any Rust program into a heap
profiler by swapping the global allocator. Every allocation and free is recorded with the
call stack that made it, then written out as a JSON report when the profiler drops.

We wire it in behind a cargo feature so normal builds are untouched — no allocator
override, no extra dependency (`nm` shows zero dhat symbols in the default binary). It
compiles in only under `--features dhat-heap`:

```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> ExitCode {
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::builder()
        .file_name(std::env::var("DHAT_OUT").unwrap_or_else(|_| "dhat-heap.json".into()))
        .build(); // writes the JSON on drop, at the end of main
    ...
}
```

Driving it is a one-liner — `translate.py --memory-report` builds with the feature, points
`DHAT_OUT` at `artifacts/dhat-<pair>.json`, runs one translation, and prints dhat's own
summary plus a derived line:

```
$ task inference-rs:translate -- en fr --text "Hello world." --memory-report
dhat: At t-gmax: 154,455,052 bytes in 771 blocks
[memory] peak heap (t-gmax): 154,455,052 bytes
[memory] top site: 49,152,000 bytes @ inference_rs::weights::Weights::new (weights.rs:132)
[memory] OK: peak dominated by the Wemb table (as expected)
```

dhat's JSON isn't just a text dump — it's a full allocation graph (program points, each
with a call stack and byte/block totals at several time points). That structure is what
makes the next two steps possible.

## 2. Visualizing: the Firefox Profiler imports the dhat format

The [Firefox Profiler](https://profiler.firefox.com) is best known for sampling CPU
profiles, but it can import a dhat JSON directly — thanks to a **dhat importer contributed
to the Profiler by Greg Tatum a few years ago**. Dropping a `dhat-*.json` onto
profiler.firefox.com (or opening a share link like the two above) gives the full
call-tree, inverted-stack, and flame-graph UI over *bytes* instead of *samples*.

The importer maps dhat's time points onto four "threads", one per metric:

| Thread | dhat metric | Meaning |
|---|---|---|
| `t-0` | Total Bytes | lifetime allocation churn (everything ever allocated) |
| `t-1` | Maximum Bytes | each site's own local maximum |
| `t-2` | **Bytes at Global Max** | live bytes at the single peak instant — the true peak |
| `t-3` | Bytes at End | still live at exit (leaks / retained) |

For a peak-memory question, `t-2` is the thread that matters. Comparing the two share
links above, the before/after difference is visible at a glance in the call tree under
`Weights::new`.

## 3. Reacting: profiler-cli closes the loop for an agent

The Profiler UI is a human activity. What let *an agent* do this analysis is
[`profiler-cli`](https://github.com/firefox-devtools/profiler) — it loads a profile into a
daemon session and answers the same queries the UI would, over the terminal (with
`--json` for machine consumption). The dhat JSON loads natively, because it goes through
the same importer.

The agent workflow was literally:

```sh
profiler-cli load inference-rs/artifacts/dhat-enfr.json     # start a session
profiler-cli profile info                                    # 4 threads, peak 154 MB
profiler-cli thread functions --thread t-2 --limit 15        # top sites at peak
profiler-cli thread functions --thread t-2 --search inference_rs
profiler-cli thread functions --thread t-2 \
    --search "decode_step|project|multihead|ffn"             # 0 functions -> decode not at peak
profiler-cli thread functions --thread t-3                   # 9.3 KB retained -> no leaks
profiler-cli stop --all
```

Each query returns self/total *bytes* per function with percentages. That is enough for an
agent to reach conclusions and act on them without a human ever opening the GUI: the
profile becomes a queryable data structure, not a picture. The GUI and the CLI are two
front-ends over one imported profile — a human can open the share link to sanity-check
exactly what the agent read.

## 4. What the profile showed

At the peak (`t-2`), inclusive byte attribution:

```
inference_rs::engine::Engine::load        105.3 MB  (68%)
  inference_rs::weights::Weights::load      73.7 MB  (48%)
    Weights::new (src clone, f32+int8)      61.4 MB  (40%)
    load_embedding (Wemb f32 dequant)       49.2 MB  (32%)
    load_embedding (Wemb int8 raw)          12.3 MB  ( 8%)
  inference_rs::model::Model::from_bytes    31.5 MB  (20%)
```

Two facts fell out immediately:

1. **The peak is entirely load-time.** Filtering the decode path (`decode_step`, `project`,
   `multihead`, `ffn`) at `t-2` returned *zero* functions — none of the per-step work is
   live at the peak. Peak memory is "what the loaded model holds", and does not grow with
   input length. (Decode allocations are real but transient; they show up only in `t-0`
   churn.)
2. **The embedding table was held four times.** `en-fr` is a *shared*-vocab model that
   logically needs one ~49 MB table, yet load kept: the dequantized f32 `Wemb`, its raw
   int8, **and a full clone of both** into a separate source slot. `32000 × 384 × 4 =
   49,152,000` and `32000 × 384 = 12,288,000` confirmed the copies exactly.

`t-3` (Bytes at End) was 9.3 KB — no leaks.

## 5. The fix

Two of the four copies were pure redundancy (`src/weights.rs`):

- **Share, don't clone.** `src_wemb` became `Option<Vec<f32>>`, `None` for shared vocab.
  The encoder falls back to `trg_wemb` instead of cloning it — no 49 MB f32 + 12 MB int8
  duplicate. Split-vocab (CJK) models, which genuinely have two different tables, keep
  `Some(encoder_Wemb)`.
- **Read int8 on demand.** The raw int8 embedding had been copied out of the model with
  `to_vec` at load, even though the parsed `Model` already holds it. `output_wemb_int8()`
  now reads it back from the model by param name, keeping no second copy.

A tempting third "win" from the first-pass analysis — *free the parsed model after building
the Weights view* — was **rejected**: `affine()` looks up every layer's weight from the
model by name at each GEMM, so those tensors are load-bearing, not redundant. The profiler
told us *what* was big; reading the code told us *which big thing was actually free to drop*.

## 6. What changed, measured

Re-profiling with the same harness (`t-2` peak):

| Model | Before | After | Saved |
|---|---:|---:|---:|
| en-fr (shared vocab) | 154.5 MB | **89.4 MB** | 62 MB (−42%) |
| en-ja (split vocab)  | ~175 MB  | **150.8 MB** | ~24 MB (−14%) |

Characteristics that changed:

- **Shared vocab loses one whole embedding duplicate** (f32 + int8) — the dominant win.
- **Both vocab types lose the int8 `to_vec` copies** — the raw int8 now lives once, in the
  model.
- **Decode profile is unchanged** — it was already allocation-light and transient; the peak
  was never about decoding.
- **No behavior change** — outputs are byte-identical (`Bonjour le monde.`,
  `こんにちは。世界。`) and all tests pass; this is purely which copies of the same bytes we keep.
- After the fix, en-fr's peak instant shifts to coincide with the SentencePiece vocab load
  (~128 K small blocks). The byte peak is still the embedding + model tables, but the vocab
  is now the largest *block-count* contributor — the next thing the profiler would point at
  if block count ever mattered.

Still on the table (not done): split-vocab keeps the model's `encoder_Wemb` int8 (~12 MB)
that is unused once the source f32 is extracted — dropping just that tensor would trim CJK
peak further. See [issues/16-wemb-dequant-memory.md](./issues/16-wemb-dequant-memory.md).

## 7. Two metrics: peak vs. retained, and the int8-only option

A long-lived translation process has two distinct memory costs:

- **Peak** — the process-lifetime high-water mark (dhat's t-gmax). Sets the OOM ceiling.
- **Retained** — what stays live *between* sentences once the model is loaded. This is what
  a translation server actually holds for hours.

### How each metric is obtained (they come from different places)

**Peak** is read straight from the dhat report — `t-gmax` (thread `t-2`, "Bytes at Global
Max").

**Retained is *not* in the JSON.** The dhat report only stores three time points: `t-gmax`
(peak) and `t-end` (bytes at exit, thread `t-3`). `t-end` is ~9 KB — at process exit
everything has been dropped, so it is *not* the steady state a long-lived server holds.
"Live heap once the model is loaded and idle" is a mid-run instant the report doesn't
capture. So the `dhat-heap` build samples it directly with `dhat::HeapStats::get()` and
prints `curr_bytes` to **stderr** — once after `Engine::load` (before translating) and once
after translating (`heap_snapshot` in `main.rs`, feature-gated):

```
[dhat] after load (retained):      live 88,585,968 bytes in 128,782 blocks (max so far 89,397,300)
[dhat] after translate (retained): live 88,595,312 bytes in 128,786 blocks (max so far 89,397,300)
dhat: At t-gmax: 89,397,300 bytes
```

So the retained figures below come from those stderr lines (`curr_bytes`), not from the
`.json`. Two things stand out: the load transient is tiny (peak 89.4 MB vs retained 88.6 MB
— only ~0.8 MB of the peak is transient), and translating leaves nothing behind (retained is
flat across sentences — no per-sentence leak). So **peak ≈ retained**, and both are dominated
by the same resident tables. For shared-vocab en-fr that ~88.6 MB is roughly: f32 `Wemb`
49.2 MB + model (int8 `Wemb` 12.3 MB + layer weights ~19 MB) + SentencePiece vocab ~8 MB.

The f32 `Wemb` is ~55% of retained and exists only to serve two things: the input-embedding
row lookup (a handful of rows per sentence) and the **full-vocab float output projection**
(all 32000 rows, every decode step, when no shortlist is used).

### The `lean-embed` feature: drop the f32 table, project in int8

The int8 `Wemb` is already `[vocab, dim]` — it can *be* the projection weight matrix. So the
`lean-embed` cargo feature (off by default) holds **no** dequantized f32 tables: embedding
rows are dequantized on demand (cheap — a few rows per sentence) and the output projection
runs full-vocab in int8, with the prepared bias precomputed once at load (it's static). Only
the int8 table already in `model` stays resident. See `weights.rs` (the `#[cfg(feature =
"lean-embed")]` split) and §"Two metrics" above for why it targets both peak and retained.

Numerics were the only risk — does full-vocab int8 argmax match the float path? A prototype
measured it on 60 diverse en-fr sentences, shortlist off:

| Comparison | Result |
|---|---:|
| float vs int8-full (internal agreement) | 58/60 |
| float projection vs `translator-cli` | 32/60 |
| int8-full projection vs `translator-cli` | 32/60 |

**Parity-neutral** — int8-full differs from float on only 2/60 sentences (near-tie flips),
and against the reference the two score identically (same story as the f64 diagnostic in
[issues/06-numeric-reduction-parity.md](./issues/06-numeric-reduction-parity.md): a
different-but-not-worse rounding). Output is byte-identical to default on the sample; en-ja
(split, shortlist) is correct on both builds.

Measured (en-fr, shared vocab, identical 3-sentence input, debug + `dhat-heap`):

| Metric | default | `lean-embed` | change |
|---|---:|---:|---:|
| peak (t-gmax, from JSON) | 89.4 MB | 63.2 MB | −26 MB (−29%) |
| retained after load (stderr `curr_bytes`) | 88.6 MB | 39.6 MB | −49 MB (−55%) |

Retained — the metric a long-lived server actually pays — nearly halves. Peak drops less
because with the f32 tables gone it's now bounded by the **model-load transient** (reading
the file + `Model::from_bytes` parsing), a separate lever (mmap / streaming parse) for later.
Split-vocab (CJK) needs two distinct tables, so it benefits from the int8 projection but not
from source/target sharing.

**Status:** implemented but gated, *not* the default. Build it with `cargo build
--features lean-embed`.

**Perf (measured — the harness now exists, `task inference-rs:perf`).** The concern was that
the full-vocab int8 GEMM replacing the f32 one each step might be slower. It's the opposite —
en-fr, dev-en, 10 runs, 1 thread:

| | default | `lean-embed` |
|---|---:|---:|
| TTFT (ms) | 7.1 | 4.0 (−44%) |
| tok/s | 211.6 | 825.0 (+290%) |

The output projection is memory-bound on reading the whole embedding table each step, and
int8 is ¼ the bytes of the dequantized f32 — so lean-embed is ~4× faster *and* lighter, with
parity-neutral output. That removes the reason it was gated; making it the default is now
just a green-light decision. (`model.get` is still a linear name scan on the hot path, same
as `affine()` today — an orthogonal cleanup.)

## Reproduce

```sh
# measure the default build (peak in the JSON; retained on stderr)
task inference-rs:translate -- en fr --text "Hello world." --memory-report

# side-by-side default vs lean-embed, identical input, into stable artifact paths
printf 'Hello world.\nThe cat sat on the mat.\n' > /tmp/in.txt
cargo build --manifest-path inference-rs/Cargo.toml --features dhat-heap
DHAT_OUT=inference-rs/artifacts/dhat-enfr-default.json \
  ./inference-rs/target/debug/inference-rs translate \
  data/models/enfr/model.enfr.intgemm.alphas.bin data/models/enfr/vocab.enfr.spm < /tmp/in.txt
cargo build --manifest-path inference-rs/Cargo.toml --features dhat-heap,lean-embed
DHAT_OUT=inference-rs/artifacts/dhat-enfr-lean.json \
  ./inference-rs/target/debug/inference-rs translate \
  data/models/enfr/model.enfr.intgemm.alphas.bin data/models/enfr/vocab.enfr.spm < /tmp/in.txt
# the [dhat] after-load lines on stderr are the retained figures; t-gmax is the peak

# analyze (headless)
profiler-cli load inference-rs/artifacts/dhat-enfr-lean.json
profiler-cli thread functions --thread t-2 --search inference_rs
profiler-cli stop --all
# or: drag the JSON onto https://profiler.firefox.com
```

Artifacts (both under gitignored `artifacts/`, so throwaway; debug builds, so absolute
numbers carry allocator/profiler overhead — the default-vs-lean delta is the signal):
`dhat-enfr-default.json` (peak 89.4 MB) and `dhat-enfr-lean.json` (peak 63.2 MB).
