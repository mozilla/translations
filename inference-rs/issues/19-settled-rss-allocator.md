# Settled RSS is gated by the allocator, not the live heap

**Open, scoped follow-on.** The C1+C2+B pass (commit `5e748e6e`) made the affine path hold each
weight once (packed) instead of twice (raw + packed) and cut allocation churn hard (dhat live
blocks at gmax 128,750 → 599; total churn 29 → 22 GB). Yet **settled ps-RSS barely moved**
(251 → 248 MiB on the en→ru base / Frankenstein benchmark; see
[../09-final-comparison.md](../09-final-comparison.md)). This issue records *why* and what would
actually move it.

## Why the dedup doesn't show up in RSS

Two macOS realities:

1. **libmalloc retains freed pages.** Dropping the raw affine weights (~26 MiB) returns them to the
   allocator, but absent memory pressure macOS does not return those pages to the OS — they're
   cached in the allocator's magazines and reused for the translation working set. So `ps` RSS,
   measured in an unpressured run, doesn't fall. dhat confirms the dedup at the *heap* level (the
   raw affine `Vec`s are freed; live-heap is lower), and it *would* help under memory pressure or
   with an allocator that releases — but the benchmark metric doesn't see it.

2. **Load touches a 2× transient.** `Weights::load` → `Model::load` does `std::fs::read` of the
   whole 42 MiB file, then `Model::from_bytes` does a `to_vec()` per tensor (another 42 MiB). dhat
   t-gmax shows both live at once (~86 MiB) during parsing. Those pages are dirtied and stay
   resident, so even a perfect steady-state dedup can't get below what load already touched.

## What would actually reduce settled RSS

- **mmap the model, pack from the mapping. ✅ DONE (commit `fa75cc0c`, `--mmap` flag, default off).**
  `Model::load_mmapped` maps the file and each tensor's `data` is a `Bytes::Mapped` view; no
  `fs::read` buffer, no per-tensor `to_vec`. **Measured: settled RSS 250 → 229 MiB, peak 251 → 229
  on en→ru base / Frankenstein** — a real ~21 MiB win (the raw copies are gone and the mapped pages
  are clean/file-backed, not dirty anon). Bit-identical (`tests/translate.rs::mmap_matches_owned`).
  Further headroom: `madvise(MADV_DONTNEED/FREE_REUSABLE)` the affine regions after packing so even
  the touched-once weight pages leave RSS, and make it the default once proven on more models.
  Original sketch: replace `fs::read` + per-tensor `to_vec` with an mmap
  and offset views (`ModelItem` = `{offset, len}` into the map, `int8_transposed()` = a slice of
  the mapping). Then: no 42 MiB read buffer, no per-tensor heap copies; affine weights are packed
  directly from the mapped bytes and, once packed, those file-backed pages are **clean and
  reclaimable** by the OS (the raw Wemb stays hot for embedding lookups). This removes the load
  transient *and* makes the raw-weight footprint evictable. Biggest lever; a `model.rs` rewrite
  (mmap crate or `libc::mmap`, lifetime of the mapping in `Model`, `int8_transposed` returns a view
  with the map's lifetime). Pairs naturally with the Rust gemmology port ([18](./18-gemmology-rust-port.md)):
  pack Rust-side straight from the mapping into a Rust-owned packed buffer, and the whole weight
  store is one representation with no double and no C++ blind spot.

- **A page-returning global allocator.** Swapping in mimalloc/jemalloc (with decommit/purge on)
  would let the freed raw-weight pages actually leave RSS after load. Cheap to try (a
  `#[global_allocator]`), and it would immediately expose the ~26 MiB dedup win the current libmalloc
  hides. Worth a quick experiment before committing to the mmap rewrite.

## What the churn pass *did* buy

Not nothing: peak RSS dropped (256 → 249, the projection no longer allocates a fresh full-vocab
buffer per step), throughput rose ~3% (fewer allocs), and the allocation profile is far cleaner
(599 vs 128,750 live blocks at gmax), which matters for CPU and for any allocator that *does* return
pages. The dedup is also the correct architecture ("retain the packed representation") and the
prerequisite for the mmap fix above. The remaining Rust churn (~22 GB) is the affine/ops result
`Vec`s returned through the hand-written forward; eliminating it needs engine-owned scratch threaded
through the layer ops (a larger refactor), and would further shrink the allocator working set.

## Update — jemalloc confirms the hypothesis and is the real lever (commit `9a35c04e`)

Added an opt-in `jemalloc` feature (`tikv-jemallocator` global allocator) and measured settled RSS
(en→ru base / Frankenstein, 20 ms RSS sampling):

| allocator | settled MiB | peak MiB | words/s |
|---|--:|--:|--:|
| libmalloc (default) | 248.7 | 248.7 | 1283 |
| jemalloc (default decay) | **145.9** | 165.3 | 1271 |
| jemalloc `dirty/muzzy_decay_ms:0` | **124.7** | 146.4 | 1221 |

**−103 MiB at ~0% throughput cost** (default decay), or **−124 MiB at ~5%** (aggressive purge).
124.7 MiB ≈ 86 (dhat t-gmax live Rust heap) + 41 (gemmology prepared-B) — jemalloc reaches the
**live-memory floor**. This directly confirms the hypothesis above: settled RSS was gated by
libmalloc's page retention, not by our live footprint. A page-returning allocator, not churn
reduction, is the settled-RSS lever. Candidate to make default (or ship with a tuned `MALLOC_CONF`);
kept opt-in for now pending a cross-platform check. Note the scratch pool ([21](./21-activation-scratch-pool.md))
does **not** stack with this — jemalloc already hits the floor, so the pool adds nothing to settled RSS.
