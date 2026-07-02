# Block-path perf profile — inference-rs vs marian-fork

First native, apples-to-apples CPU profile of the two engines on the **block-batched
path** (the production unit of work: a paragraph's sentences translated in one batched
call on an already-loaded model). It answers *where the time goes* and therefore *what to
optimize next*.

## Profiles

Both recorded with `samply` and uploaded to the Firefox Profiler:

- **inference-rs:** https://share.firefox.dev/4aSK3WT — `artifacts/perf-blocks-rs-enfr.json.gz`
- **marian-fork:**  https://share.firefox.dev/4oYk5XA — `artifacts/perf-blocks-marian-enfr.json.gz`

## How they were collected

```
inference-rs/scripts/perf.py en fr --samply --blocks
```

`--samply --blocks` records **both** baselines on the block path (see
[07-batched-inference.md](./07-batched-inference.md) for the block model and the
cheat-proof batching validation):

- **inference-rs** — `translate --blocks` over `corpora/nllb-en-fr.blocks.txt` (307
  blocks / 1000 sentences, en→fr), one batched encode+decode per block.
- **marian-fork** — `block-bench` (bergamot `BlockingService::translateMultiple`, one
  batched call per block) over the identical block file, with `ssplit-mode: sentence`
  (pre-split, no re-splitting) so both engines see exactly the same sentence set.

Fairness basis: single-threaded on both sides (`block-bench` is single-threaded by
construction — `BlockingService`), shortlist **off** on both (production baseline; opt-in
only), model loaded once and excluded from the samples, same input, same machine (arm64).

Both files are standard Firefox Profiler JSON — `samply load <file>` (or drag onto
profiler.firefox.com) symbolicates function names from the binaries on disk. The
per-function numbers below were produced by symbolicating every sample's leaf frame with
`atos` against the release binary / `block-bench` and aggregating self-time.

## Result — the gap is one thing: the int8 GEMM kernel

Block-level throughput (from `--blocks` timing mode, not the profile): inference-rs sits at
**~0.07× marian** — i.e. marian is ~14× faster per block. The profile shows why, and it is
*not* diffuse overhead.

### inference-rs — 97% of self-time is int8 matmul, dominated by the output projection

| % self | function | what it is |
|-------:|----------|------------|
| **61.4%** | `weights::full_logits` | full-**vocab** output projection, run every decode step |
| **35.3%** | `ops::intgemm_affine` | every other int8 matmul (attention Q/K/V/O, FFN) |
| 0.7% | `engine::project_argmax` | argmax over logits |
| 0.4% | `engine::multihead_batched` | attention assembly |
| 0.4% | `weights::affine` | remaining affine glue |
| 0.3% | `ops::layer_normalization` | |
| <0.3% ea. | softmax, postnorm, iterators, allocs | |

Total self-samples: 141,574. The two GEMM functions alone are **96.7%**.

### marian-fork — same operation, but on a vectorized kernel

| ~% self | function | what it is |
|--------:|----------|------------|
| ~50%+ | `gemmology::Engine<i8mm<neon64>>::Shift::Multiply` (`UnquantizeAndAddBiasAndWrite`) | **ARM i8mm SIMD + cache-blocked int8 GEMM** — the same shifted-uint8 × int8 math we do |
| ~4% | `NthElementCPU::getNBestList` (`partial_sort`) | top-k / best-token over logits |
| ~3% | `cpu::element` (`Plus`) | elementwise bias add |
| … | tail | tokenization, tensor glue |

Total self-samples: 9,053 (marian finished the same work in ~1/15th the samples).

## Interpretation

Both engines are **compute-bound in the identical operation** — shifted-uint8 × int8 GEMM,
mostly the decoder's full-vocabulary output projection. Neither is bottlenecked on memory
layout, allocation, attention bookkeeping, or the decode loop structure. The entire ~14×
gap is **kernel quality**:

- inference-rs runs a **scalar Rust loop** (`intgemm_affine` / `full_logits`) — numerically
  exact and marian-oracle-validated, but no SIMD and no cache/register blocking, so the
  weight matrix is streamed from memory with one MAC per element.
- marian runs `gemmology`'s **hand-vectorized ARM i8mm dot-product kernel** with
  register/cache blocking — the same math, many lanes per instruction, weights reused
  across the block's rows while they're hot in cache.

The single most important number here is **61.4% in `full_logits`**: the full-vocab output
projection is the biggest cost in the whole engine, more than everything else combined.
(This is also the quantitative reason shortlisting exists — it shrinks exactly this matmul.
We keep it off by default; the lever below applies with or without it.)

## What this changes about the plan

This **sharpens** the batched-inference follow-on (§6 of
[07-batched-inference.md](./07-batched-inference.md)). The takeaways:

1. **The bottleneck is the kernel, not the batching layout.** Milestones 1–2 already proved
   the batched path is numerically correct (batch-invariant, bit-identical to the
   oracle-validated single path). Batching created the *opportunity* for matrix×matrix reuse;
   the scalar kernel doesn't capture it. Optimizing encoder batching further would move <1%.
2. **Target `full_logits` first (61%), then `intgemm_affine` (35%).** These are ~97% of
   self-time; nothing else is worth touching until they are vectorized.
3. **We already vendored the kernel.** `gemmology` is in-tree (commit `a9635de5`) — the exact
   i8mm kernel marian uses. The path is to route inference-rs's matmuls (output projection
   first) through a gemmology-style i8mm kernel instead of the scalar loop, keeping the
   existing marian-oracle parity + batch-invariance tests as the correctness gate (a pure
   perf change must not move a single logit).
4. **Re-run this exact profile to confirm.** `perf.py --samply --blocks` after the kernel
   swap should show the `full_logits` / `intgemm_affine` bars collapse toward marian's
   `gemmology` share, and the block-mode ratio move off 0.07×.

## Update — gemmology kernel landed (result)

The kernel swap is done (commit `34272650`, `gemmology` cargo feature): `ops::intgemm_affine`
— the transformer affines, and the full-vocab output projection under `lean-embed` — now
routes through the vendored gemmology i8mm SIMD kernel via a thin C-ABI shim
(`src/gemmology_shim.cpp`, `src/gemm.rs`). Validated bit-for-bit against the scalar kernel
(`tests/gemm_parity.rs`, max diff `0.00e0` across shapes incl. the non-multiple-of-8 padding
path), so it inherits the marian-oracle validation transitively; the full suite passes under
`--features gemmology` and `--features lean-embed,gemmology`.

**Throughput (block mode, en→fr, 307 blocks / 1000 sentences, single-thread, shortlist off):**

| config | sent/s | ratio vs marian |
|---|---:|---:|
| default build (scalar f32 proj + scalar int8 affine) | — | **0.07×** (original) |
| `lean-embed` (scalar int8) | 15.7 | 0.14× |
| **`lean-embed,gemmology` (i8mm SIMD)** | **54.4** | **0.48×** |

That is **3.5× over the scalar int8 path** and **~7× over the original default-build baseline**,
closing the gap to marian from 0.07× to 0.48× (now within ~2×).

**Re-profiled after the swap** (`perf.py --samply --blocks --features lean-embed,gemmology`):
total self-samples dropped **141,574 → 18,469** (~7.7× less CPU for identical work), and the
hot path is now the intended one:

| % self | function |
|-------:|----------|
| **73.5%** | `gemmology::Shift::Multiply<i8mm<neon64>>` — the SIMD int8 GEMM (same kernel as marian) |
| 6.0% | `engine::project_argmax` |
| 4.9% | iterator/`map` (activation quantize / dequant) |
| 3.1% | `multihead_batched` |
| 2.6% | `weights::affine` (glue), 2.6% `layer_normalization` |

The engine is now GEMM-bound on the *same* kernel marian uses — the scalar bottleneck is gone.

### Next lever — batch the output projection

The remaining ~2× is structural, not kernel quality: the full-vocab output projection runs
**per row** (`project_argmax` → `full_logits`, `m = 1`) for each sentence in the block, so the
large vocab weight is streamed from memory once *per sentence* instead of once *per block*.
marian projects the whole minibatch in one GEMM. Projecting the block's decoder tops together
(`m = batch`) through one gemmology call would amortize the vocab-weight streaming across the
batch — the same matrix×matrix reuse win, now at the output layer. The decoder affines are
already batched (`m = batch`); this is the one remaining `m = 1` GEMM. Correctness gate is
unchanged (batch-invariance + oracle parity).

## Reproducing

```
# throughput ratio (block mode) — pass the perf config features
inference-rs/scripts/perf.py en fr --blocks --features lean-embed,gemmology

# throughput ratio (block mode)
inference-rs/scripts/perf.py en fr --blocks

# profiles of both baselines (block path)
inference-rs/scripts/perf.py en fr --samply --blocks
# → artifacts/perf-blocks-rs-enfr.json.gz, artifacts/perf-blocks-marian-enfr.json.gz
# view: samply load <file>   (or drag onto https://profiler.firefox.com)
```

`artifacts/` is gitignored (throwaway); regenerate with the commands above.
