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

## Profiles (attached)

Generated on this exact workload (en→ru base, Frankenstein blocks). Files land in
`artifacts/` (gitignored — regenerate with the commands below). Both `.json.gz` and the dhat
`.json` load in the **Firefox Profiler** (`samply load <file>`, or drag onto
profiler.firefox.com; the dhat JSON imports via the same importer / `dhat/dh_view.html`).

| profile | file | tool | shared |
|---|---|---|---|
| inference-rs CPU | `artifacts/perf-blocks-rs-enru.json.gz` | samply | https://share.firefox.dev/3Rg5AC0 |
| marian CPU | `artifacts/perf-blocks-marian-enru.json.gz` | samply | https://share.firefox.dev/4wfdgmM |
| inference-rs heap | `artifacts/dhat-frankenstein-enru.json` | dhat | https://share.firefox.dev/4p3s1qA |

**CPU (samply) — both engines are ~80% in the *same* i8mm kernel:**

| % self | inference-rs | marian block-bench |
|--:|---|---|
| ~78–81% | `gemmology::Shift::Multiply<i8mm<neon64>>` | `gemmology::Shift::Multiply<i8mm<neon64>>` |
| next | `multihead_batched` 4%, `step_select` 3%, `affine` 3%, layernorm 2% | `cpu::element` (bias/elementwise) 5%, `partial_sort` (top-k) 3%, PrepareA 1% |

On the larger base model inference-rs is *even more* GEMM-bound (81% vs 73.5% on the en→fr student
in [08](./08-perf-analysis.md)) — same kernel as marian, so the remaining ~3× throughput gap is
**how much kernel work each issues**, not kernel speed: marian packs/amortizes the minibatch more
aggressively (fewer, larger GEMM calls with more B-reuse). Closing it means issuing fewer, larger
matmuls, not a faster inner loop.

**Heap (dhat) — clean, but allocation-heavy:**
- **Peak Rust heap (t-gmax): 88 MB**, dominated by `Model::from_bytes` **43 MB** (the resident
  int8 weights) plus transient full-vocab logits and FFN buffers (~10 MB each). **Zero leaked**
  at exit (1 KB in 2 blocks).
- **29 GB total churn over 3.1M allocations** — nearly all in `PreparedB::matmul` (~17 GB; it
  returns a fresh `Vec<f32>` every call) and `Weights::affine` (~3.5 GB; `prepare_a` + prepared
  bias per call). All short-lived, so it doesn't raise the retained footprint, but it's real
  allocator traffic — the concrete form of the "reduce activation copies" lever from 08 (reuse
  scratch buffers instead of allocating per matmul/affine).
- **Caveat:** dhat sees only the **Rust** heap. The gemmology prepared-B weights are allocated in
  C++ (`std::aligned_alloc`) and are invisible here; they (plus malloc arenas) account for the
  gap between the 88 MB Rust peak and the 251 MB settled RSS measured above.

### Where the 251 MiB actually lives

Combining the dhat heap, a byte counter added to the gemmology shim
(`gemmology_prepared_bytes()`), and an RSS trajectory (sampled every 50 ms), the settled footprint
breaks down as:

| component | ~MiB | how measured | resident? |
|---|--:|---|---|
| int8 model blob — all weights + biases + embeddings, Rust `Model` | 43 | dhat, `Model::from_bytes` t-gmax | yes, whole run |
| **gemmology prepared-B** — packed copies of every affine + the output-proj `Wemb`, C++ | **41** | shim counter (42,598,400 B) — **invisible to dhat** | yes, whole run |
| transient activations — full-vocab logits `[batch×32000]`, FFN `[batch×seq×2048]`, attention | tens | dhat (t-gmax 88 − 43 model) | cycled per block |
| allocator working set + runtime/libs/stacks/binary | remainder (~120) | RSS − the above | grows then plateaus |

RSS trajectory over one run: **46 MiB pre-load → 184 MiB right after load → plateau 252 MiB** by
~25% in, then flat (settled). Two things drive the 251 MiB, and both are explainable:

1. **The weights are held twice (~43 + 41 ≈ 84 MiB).** `Model` keeps the raw int8 file blob
   (needed for the on-demand embedding dequant and biases), and gemmology holds a *second, packed*
   copy of every matrix that goes through a GEMM — the affine weights plus the 16.4 MiB
   output-projection `Wemb` (32000×512). This duplication is the price of the SIMD kernel and is
   **exactly the memory dhat cannot see** (C++ `aligned_alloc`), so the true retained-weight cost
   is ~84 MiB, not the 43 MiB dhat alone suggests. marian pays the same prepared-B cost.
2. **Allocation churn inflates the allocator's resident set.** The 184→252 MiB climb during
   translation tracks the 29 GB / 3.1 M-allocation churn (`matmul` returning a fresh `Vec` per
   call, `affine`'s per-call buffers): libmalloc retains freed pages in its magazines rather than
   returning them to the OS, so RSS rises to the working-set high-water and stays. dhat confirms
   it's *not* a leak (1 KB live at exit) — it's allocator retention. Reusing scratch buffers
   (the "reduce activation copies" lever) would shrink both the churn and this resident overhead.

Even carrying the duplicated weights and the churn working-set, settled RSS (251) is still below
native marian (298) and Firefox's inference process (355) — `lean-embed` (no resident f32
embedding/projection tables) more than pays for the gemmology duplication. The clear next memory
win is cutting the per-call allocations in `matmul`/`affine`.

## Update — memory/churn pass (C1+C2+B, commit `5e748e6e`)

Acting on the breakdown above, the affine path now (a) **holds each weight once**: every affine
weight is packed at load and its raw int8 bytes freed (`prepare_affines`), so the ~26 MiB raw
affine copy is gone at the heap level; (b) **caches the prepared bias** instead of recomputing it
per call; (c) **reuses activation scratch** (`prepare_a_into`, `matmul_into`, and an engine-held
logits buffer for the projection). Token-identical, bit-exact `gemm_parity`, all five feature
configs green.

Measured effect (en→ru base, Frankenstein), vs the table above:

| metric | before | after |
|---|--:|--:|
| words/s | 467 | **480** (+3%) |
| peak RSS (MiB) | 256 | **249** |
| **settled RSS (MiB)** | 251 | **248** |
| dhat live blocks @ gmax | 128,750 | **599** |
| dhat total churn | 29 GB | **22 GB** |

So: throughput up, peak down, and the allocation profile is dramatically cleaner — but **settled
RSS barely moved.** The dedup is real (dhat shows the raw affine `Vec`s freed), but on macOS
libmalloc keeps freed pages resident absent memory pressure, and `Model` load touches a 2×
transient (`fs::read` the whole file + `to_vec` per tensor). So the retained-once win doesn't reach
`ps` RSS in an unpressured run.

### `--mmap` closes it (commit `fa75cc0c`)

The genuine settled-RSS lever from [issue 19](./issues/19-settled-rss-allocator.md) is now
implemented as an opt-in flag: `translate --mmap` maps the model file (memmap2) so each tensor is a
view into the mapping — no `fs::read` whole-file buffer, no per-tensor `to_vec`, and the weight
pages are clean/file-backed (reclaimable) rather than dirty anonymous copies.

| config (en→ru base, Frankenstein) | settled RSS | peak RSS |
|---|--:|--:|
| owned (default) | 250 | 251 |
| **`--mmap`** | **229** | **229** |

A real **−21 MiB** (≈ the raw-weight copies), bit-identical output
(`tests/translate.rs::mmap_matches_owned`). That puts inference-rs at **229 MiB vs marian 298 vs
Firefox 355** — comfortably the lightest. Further headroom (madvise the packed-once affine pages;
promote `--mmap` to default) is noted in issue 19.

## Update — decoder optimization pass: 0.36× → 0.96× marian

A follow-on pass (cross-attention K/V caching + finished-sentence retirement, plus the issue-21
allocation eliminations) took inference-rs from **467 → 1274 words/s (0.36× → 0.96× native marian,
1.11× → 3.04× Firefox)** on this same benchmark, while settled RSS stayed at **249 MiB** (still the
lightest of the three). All changes token-identical. **The throughput figures in the Results table
above are superseded by that pass** — see **[10-decoder-optimizations.md](./10-decoder-optimizations.md)**
for the full write-up (optimizations, design decisions, correctness methodology, and per-stage
results). Commits `df600c01`, `dabfc419`, `ef93017c`, `c4894f46`.

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

# 4. Profiles on the same workload
#    - inference-rs + marian CPU (samply, Firefox Profiler format):
inference-rs/scripts/perf.py en ru --samply --blocks inference-rs/corpora/frankenstein-en.blocks.txt
#    - inference-rs heap (dhat, imports into the Firefox Profiler):
cargo build --release --features dhat-heap --manifest-path inference-rs/Cargo.toml
DHAT_OUT="$PWD/inference-rs/artifacts/dhat-frankenstein-enru.json" \
  inference-rs/target/release/inference-rs translate \
  data/models/enru/model.enru.intgemm.alphas.bin data/models/enru/vocab.enru.spm \
  data/models/enru/vocab.enru.spm --blocks inference-rs/corpora/frankenstein-en.blocks.txt >/dev/null
```

`data/` and `artifacts/` are gitignored; the corpus (`corpora/frankenstein-en.blocks.txt`) and
both scripts are committed. To refresh the Firefox column, re-run its perftest
(`./mach perftest browser/components/translations/tests/browser/browser_translations_perf_base.js`,
see 08-perf-analysis.md) and update the `FIREFOX` medians at the top of `final_comparison.py`.

## en-ru

The en-ru model is the larger 42M model.

```
➤ ls data/models/enru
total 45M
drwxr-xr-x 6 greg staff  192 Jul  2 20:36 .
drwxr-xr-x 6 greg staff  192 Jul  2 18:57 ..
-rw-r--r-- 1 greg staff 2.7M Nov 24  2025 lex.50.50.enru.s2t.bin
-rw-r--r-- 1 greg staff  42M Nov 24  2025 model.enru.intgemm.alphas.bin
-rw-r--r-- 1 greg staff 884K Nov 24  2025 vocab.enru.spm
-rw-r--r-- 1 greg staff  316 Jul  2 18:58 config.enru.yml
```

### How the memory inflates — a closed-form account

The 42 MB on disk is int8. Everything above that is the engine turning those int8
bytes into something the SIMD kernel and the float math can consume. This is a
read-only, first-pass breakdown: the weight numbers are **computed from the
architecture dims** (`dim-emb 512, dim-ffn 2048, enc 6, dec 2, heads 8, vocab
32000, tied-embeddings-all`, read from the embedded `model.yml`) and cross-checked
against the shim's `gemmology_prepared_bytes()` counter and the dhat figures above;
the transient sizes are computed from the same dims × a representative block shape.
Items flagged *(measure)* are second-pass work.

**The int8 weight budget (closed form).** Every GEMM weight, exact from the dims
(1 byte/element):

| group | shape(s) | int8 bytes | MiB |
|---|---|--:|--:|
| encoder affines (×6): Wq/Wk/Wv/Wo `512²`, ffn_W1 `512×2048`, ffn_W2 `2048×512` | 6 × 3,145,728 | 18,874,368 | 18.0 |
| decoder affines (×2): rnn_W/rnn_Wf `512²`, ctx Wq/Wk/Wv/Wo `512²`, ffn_W1/W2 | 2 × 3,670,016 | 7,340,032 | 7.0 |
| **all affines** | | **26,214,400** | **25.0** |
| `Wemb` tied embedding / output projection | `32000×512` | 16,384,000 | 15.6 |
| **all int8 matrices** | | **42,598,400** | **40.6** |

That last total is *exactly* the shim's reported `gemmology_prepared_bytes()`
(42,598,400 B) — confirming gemmology holds a packed copy of **every** int8 matrix:
all affines *plus* the projection `Wemb` (padding is nil since `n`, `vocab` are
already multiples of 8). And 26,214,400 is exactly the "~26 MiB raw affine copy"
freed by the C1 pass. So the accounting closes on itself.

**Four ways int8-on-disk becomes resident RSS.** Mapping the user's mental model
onto what the code (`model.rs`, `weights.rs`, `gemm.rs`, `engine.rs`) actually does:

1. **Raw file → resident weights.** *Default* (`Model::load`) does `fs::read` of the
   whole 42 MB file, then `to_vec` per tensor — a ~2× transient at load (file buffer
   + owned copies), settling to ~43 MB of dirty anonymous heap (int8 matrices + f32
   biases/layernorms + quant-mults + `model.yml`). `--mmap` (`Model::load_mmapped`)
   makes each tensor a view into the mapping: no whole-file buffer, no per-tensor
   copy, and the pages are **clean/file-backed (reclaimable)** rather than dirty — the
   measured −21 MiB settled. So "raw weights are mmapped in" is true only under
   `--mmap`; the default heaps them.

2. **The `Wemb` 4× inflation — this is the lever `lean-embed` exists to kill.** In the
   **default (non-`lean-embed`) build**, `load_embedding` dequantizes the int8 `Wemb`
   into a **resident f32 table**: `32000×512×4 = 62.5 MiB`, a literal 4× blow-up of the
   15.6 MiB int8 table, held for the whole run (shared vocab → one table; split/CJK →
   two). The **fast config (`lean-embed`) never builds it**: `src_embed_row`/
   `trg_embed_row` dequantize *one row on demand* (`dim` floats, transient), and the
   output projection runs full-vocab **int8** through gemmology. So the single biggest
   "inflation" in your mental model is the one the shipping config deliberately avoids —
   it's why inference-rs lands *under* marian and Firefox rather than over. Quantify it
   as **−62.5 MiB vs the naive f32-table engine.**

3. **The gemmology packed copy (SIMD working memory).** `PreparedB::new` repacks each
   int8 matrix into the register-blocked tile layout in **C++ `aligned_alloc`**
   (invisible to dhat): **40.6 MiB** total, resident the whole run. Nuance the current
   doc already half-states, now exact:
   - **Affines are held *once*.** `prepare_affines` packs them at load and then frees
     the raw int8 bytes from `Model` (`data = Bytes::Owned(Vec::new())`). So the 25.0
     MiB of affines lives *only* in gemmology, not twice.
   - **`Wemb` is held *twice* — unavoidably.** The packed projection copy (15.6 MiB) is
     built lazily on first projection (`proj_pb`), but the raw int8 `Wemb` must stay in
     `Model` because embedding *lookups* (`dequant_row`) read it directly. Two uses,
     two layouts → ~31.2 MiB for the embedding alone. This is the one genuine weight
     duplication in the fast config.
   - Net resident weights, fast config ≈ **~17 MiB in `Model`** (raw `Wemb` + biases +
     layernorms + quant-mults + `model.yml`) **+ 40.6 MiB gemmology** ≈ **~58 MiB**.

4. **Transient activations — eager, not an "expression graph."** Correction to the
   mental model: inference-rs does **not** build marian's lazy expression graph. It
   evaluates **eagerly** — every op (`affine`, `ffn`, `multihead`, `softmax`, `relu`,
   `layer_normalization`) returns a **fresh `Vec<f32>`** that drops at end of scope. So
   the transient cost is *churn*, not a large retained graph. Largest live buffers, for
   a representative block (batch `b`, max source len `seq`, `rows = b·seq`):
   - full-vocab logits `[active × 32000]` f32 — the reused `logits_scratch` — e.g. **~1
     MiB** at 8 active rows;
   - FFN hidden `[rows × 2048]` f32 — e.g. **~1.6 MiB** at rows=200;
   - encoder context `[b × seq × 512]`, plus per-op q/k/v/joined `[rows × 512]`.

   All small and short-lived (single-digit MiB live at once), but allocated *per op per
   step* — that's the 22 GB / churn the C-pass targeted. The C1+C2+B pass already
   removed the per-call `prepare_a`/bias/matmul allocations on the hot affine path
   (`prepare_a_into`, `matmul_into`, cached bias); the remaining fresh `Vec`s are the
   elementwise/attention intermediates. **The inflation here is allocator working-set
   retention, not live footprint** — libmalloc holds freed pages resident absent
   pressure, which is why settled RSS sits ~120 MiB above the ~58 MiB of resident
   weights even though almost nothing is leaked (dhat: ~1 KB live at exit).

**Summary — fast config (`lean-embed,gemmology`), the ~248–250 MiB settled owned / 229
`--mmap`:**

| bucket | ~MiB | mechanism | held |
|---|--:|---|---|
| `Model` int8 residue (raw `Wemb` + biases + LN + quant-mults + yaml) | ~17 | fs::read → owned copies (default) | whole run |
| gemmology prepared-B (all affines once + `Wemb`-proj) | 40.6 | C++ `aligned_alloc`, SIMD packing | whole run |
| transient activations (logits/FFN/attn `Vec`s) | single-digit live | eager per-op allocation | cycled |
| allocator working set + runtime/libs/stacks/binary | remainder (~120–180) | libmalloc page retention under churn | plateaus |
| *avoided* by `lean-embed`: resident f32 `Wemb` table | −62.5 | on-demand row dequant instead | — |
| *avoided* by `--mmap`: dirty raw-weight heap copies | −21 | file-backed clean pages | — |

**Second-pass measurements to confirm (need permission to run):**
- Split settled RSS *default vs `lean-embed`* to see the 62.5 MiB f32-table cost land in
  `ps` (expected: `lean-embed` ≫ cheaper; quantifies mechanism #2 directly).
- dhat `Model::from_bytes` t-gmax *after* `prepare_affines` frees the affines, to confirm
  `Model` settles near ~17 MiB (mechanism #3).
- `gemmology_prepared_bytes()` printed at steady state (expect 42,598,400 exactly).
- Peak transient (dhat t-gmax − resident) on the *largest* Frankenstein block to bound
  mechanism #4's live high-water.
