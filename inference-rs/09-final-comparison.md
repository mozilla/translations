# Final three-way comparison: inference-rs vs native marian vs Firefox Wasm

The apples-to-apples finale: the **same corpus**, the **same model**, and the same
block-shaped work through three engines —

1. **inference-rs** — our Rust engine, default fast config (`lean-embed` + `gemmology`),
2. **marian `block-bench`** — the native marian-fork reference (bergamot),
3. **Firefox Wasm** — Full-Page Translations, the shipping browser path.

All three translate Firefox's own benchmark page (a public-domain *Frankenstein* excerpt)
with the **en→ru `base` model**, block by block (one batched translate per paragraph on a
loaded engine — the production shape).

## Setup

- **Model:** the en→ru **`base`** model from the Firefox perftest (`base.model.enru.intgemm.alphas.bin`,
  shared `base.vocab.enru.spm`), decompressed from `~/.mozfetches`. This is the larger
  architecture — **dim-emb 512, ffn 2048, enc-depth 6, dec-depth 2 (SSRU), 8 heads, tied
  embeddings** — not the small en→fr student used in [08-perf-analysis.md](./08-perf-analysis.md).
  Our engine reads the dims from the embedded `model.yml` and runs it unchanged; spot-checked
  correct ("The monster was created by a scientist." → "монстр был создан ученым.").
- **Corpus:** `corpora/frankenstein-en.blocks.txt`, generated from Firefox's
  `translations-bencher-en.html` by `scripts/make_frankenstein_blocks.py` — one block per
  block-level element (skipping `translate="no"`), sentences split one per line. **103 blocks,
  403 sentences, 9,513 words, 13,340 source tokens** (Firefox counts 9,575 words / 12,955
  tokens on the same page; our tokens include one EOS per sentence — 13,340 − 403 = 12,937 ≈
  12,955, confirming the same tokenization).
- **Metric parity with `TranslationsBencher`:** `words/s = source words ÷ translation seconds`,
  `tokens/s = source tokens ÷ translation seconds`, both with **model load excluded** (Firefox
  measures engine-ready → done; the native tools sum per-block compute). `init ms` is model
  load / engine init, measured separately.
- **Memory — settled vs peak.** We sample each process's RSS every 20 ms (like Firefox's
  `PeakMemorySampler`). **`settled`** is the retained working set *during* translation — the
  median RSS over the run's second half, past the model-load ramp (what Activity Monitor shows
  sitting there); **`peak`** is the max RSS including the load transient. For Firefox these map
  to `stabilized-` and `peak-inference-process-memory-usage`.
- Single-thread, shortlist off, on Apple Silicon (M5 Max). Native numbers are the median of 4
  timed runs (1 warmup); Firefox numbers are the median of the 5-run perftest.
- Harness: `scripts/final_comparison.py`.

## Results

| engine | words/s | tokens/s | translate (s) | init (ms) | settled RSS (MiB) | peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| **inference-rs** (rust, fast) | **467** | 655 | 20.37 | 70 | **251** | 256 |
| marian `block-bench` (native) | **1312** | 1840 | 7.25 | 62 | 298 | 298 |
| Firefox Wasm (Full-Page) | 419 | 567 | 22.85 | 135 | 355¹ | 355 |

¹ Firefox's **inference-process** memory (settled = stabilized, peak = peak); it also runs a
**parent process** (~385 MiB peak) that the native single-process tools have no equivalent of.
The settled/peak gap shows the model-load transient: only ~5 MiB for inference-rs, none
resolvable above steady-state for marian or Firefox at this sampling rate.

**Speedups (words/s):**
- inference-rs vs Firefox Wasm: **1.11×**
- native marian vs Firefox Wasm: **3.13×**
- inference-rs vs native marian: **0.36×**

## What this says

**Throughput.** Our Rust engine (470 words/s) already **edges out the shipping Wasm path**
(419) by ~12% on identical work, and native marian (1323) is ~3.2× the Wasm path. So the field
is: Wasm is the floor, native marian the ceiling (~3×), and inference-rs currently sits just
above Wasm with clear headroom toward native. The gap to native marian (0.36×) is the same
kernel-efficiency gap tracked in 08 — both are GEMM-bound on the identical i8mm kernel; marian's
is more cache/register-tuned and its decode-loop overheads are lower. The per-call overheads
called out in 08 (prepared-bias recompute, activation copies) are the next levers.

**Memory — our strongest result.** inference-rs retains **251 MiB settled** (256 peak), the
**lowest of the three**:
- 16% under native marian (298), and
- 29% under Firefox's inference process alone (355 settled), before even counting Firefox's
  ~385 MiB parent process.

The settled number is the fair one for "how much does it hold while translating": we sample RSS
through the run and take the steady-state plateau, so the model-load transient (a mere ~5 MiB
here) doesn't inflate it. That win is `lean-embed` doing its job (06-memory-approach.md): the
int8 embedding table stays resident and is dequantized on demand instead of holding a resident
f32 copy, and the output projection runs int8 — so the larger `base` model (bigger embeddings +
ffn) still fits in less memory than either reference. The memory-safe Rust rewrite is also the
leanest.

**Init.** inference-rs (70 ms) and native marian (62 ms) load the model from disk in about the
same time; Firefox's 135 ms includes Wasm instantiation + engine/worker spin-up + payload
transfer, which the native tools skip. (These init figures are process wall minus translation
compute, so they also absorb process spawn and the RSS-sampling overhead — treat as approximate.)

## Caveats (why this is a gut check, not a lab benchmark)

- **Wasm vs native, and end-to-end vs kernel.** Firefox's path is the *same Bergamot kernel
  compiled to Wasm*, run in a worker with HTML parsing and process IPC around every block. It is
  the shipping end-to-end path; the native tools measure just the inference. So marian-native ≫
  Firefox is expected (Wasm SIMD < native SIMD, plus overhead), and inference-rs ≈ Firefox is a
  meaningful "our native Rust ≈ their shipping Wasm" data point, not "our kernel ≈ their kernel."
- **Sentence segmentation.** We pre-split paragraphs with a heuristic; Firefox uses ssplit-cpp.
  Boundaries differ slightly, so per-block batch sizes differ — but the words/tokens totals (the
  numerators) do not, and both group by the same paragraphs, so words/s is robust.
- **Counting.** whitespace words (9,513) vs Firefox's ICU (9,575); our source tokens include EOS
  (13,340 vs 12,955). ~1–3% differences, noted above.
- **Language/lengths.** en→ru for all three (same model), so decode lengths are comparable.
- **`translate s`** is summed per-block compute for the native tools and wall (engine-ready→done)
  for Firefox; both exclude model load. On a single thread these are close.

## Reproducing

```
# 1. Get the en→ru base model (from the Firefox perftest fetch cache) into data/models/enru/
for f in model.enru.intgemm.alphas.bin vocab.enru.spm lex.50.50.enru.s2t.bin; do
  zstd -d -f -o data/models/enru/$f ~/.mozfetches/base.$f.zst
done
# write data/models/enru/config.enru.yml (relative-paths, models:, vocabs: [vocab, vocab],
# gemm-precision: int8shiftAlphaAll)  — see the enfr config for the template

# 2. Regenerate the corpus from Firefox's benchmark page (or use the committed one)
inference-rs/scripts/make_frankenstein_blocks.py

# 3. Run the three-way comparison (Firefox column is baked in from the perftest run)
inference-rs/scripts/final_comparison.py
```

`data/` and `artifacts/` are gitignored; the corpus (`corpora/frankenstein-en.blocks.txt`) and
both scripts are committed. To refresh the Firefox column, re-run its perftest
(`./mach perftest browser/components/translations/tests/browser/browser_translations_perf_base.js`,
see 08-perf-analysis.md) and update the `FIREFOX` medians at the top of `final_comparison.py`.
