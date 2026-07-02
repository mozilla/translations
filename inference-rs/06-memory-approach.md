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

The dhat report gives peak directly. For retained, the `dhat-heap` build now snapshots
`dhat::HeapStats` after load and after translating (`main.rs`, feature-gated):

```
[dhat] after load (retained):      live 88,585,968 bytes in 128,782 blocks (max so far 89,397,300)
[dhat] after translate (retained): live 88,595,312 bytes in 128,786 blocks (max so far 89,397,300)
dhat: At t-gmax: 89,397,300 bytes
```

Two things stand out: the load transient is tiny (peak 89.4 MB vs retained 88.6 MB — only
~0.8 MB of the peak is transient), and translating leaves nothing behind (retained is flat
across sentences, no per-sentence leak). So **peak ≈ retained**, and both are dominated by
the same resident tables. For shared-vocab en-fr that ~88.6 MB is roughly: f32 `Wemb`
49.2 MB + model (int8 `Wemb` 12.3 MB + layer weights ~19 MB) + SentencePiece vocab ~8 MB.

The f32 `Wemb` is ~55% of retained and exists only to serve two things: the input-embedding
row lookup (a handful of rows per sentence) and the **full-vocab float output projection**
(all 32000 rows, every decode step, when no shortlist is used).

### Option considered: drop the f32 table, project in int8

The int8 `Wemb` is already `[vocab, dim]` — it can *be* the projection weight matrix. If the
output projection runs in int8 over the full vocab (the reference already projects in int8),
and embedding rows are dequantized on demand (cheap — a few rows per sentence), the f32 table
is unnecessary. That removes ~49 MB (shared) / ~98 MB (split) from **both** peak and
retained — retained en-fr would drop from ~88.6 MB to ~40 MB.

The only real risk is numerics: does full-vocab int8 argmax match the current float path?
A prototype (behind `RS_INT8_PROJ`, since reverted) measured it on 60 diverse en-fr
sentences, shortlist off:

| Comparison | Result |
|---|---:|
| float vs int8-full (internal agreement) | 58/60 |
| float projection vs `translator-cli` | 32/60 |
| int8-full projection vs `translator-cli` | 32/60 |

**Parity-neutral.** int8-full differs from float on only 2/60 sentences (near-tie flips),
and against the reference the two score identically — same story as the f64 diagnostic in
[issues/06-numeric-reduction-parity.md](./issues/06-numeric-reduction-parity.md): a
different-but-not-worse rounding. So the f32 table can go with no parity regression.

A real implementation would: (1) not materialize `trg_wemb`/`src_wemb` as f32; (2)
dequantize embedding rows on demand in `embed()`; (3) project full-vocab int8, **precomputing
the prepared bias once at load** (it's static — the spike recomputed it per step, which is
why the prototype ran slow). This is a hot-path change to the default decode, so it's left as
a green-light decision rather than folded in with the load-time cleanups.

## Reproduce

```sh
# measure
task inference-rs:translate -- en fr --text "Hello world." --memory-report
# analyze (headless)
profiler-cli load inference-rs/artifacts/dhat-enfr.json
profiler-cli thread functions --thread t-2 --search inference_rs
profiler-cli stop --all
# or: drag inference-rs/artifacts/dhat-enfr.json onto https://profiler.firefox.com
```
