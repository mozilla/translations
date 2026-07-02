# Wemb Dequantization Memory Expansion

**Out of scope** — recording it because it's worth noting, not scheduling it.

`Wemb` ships in the model as int8 (~12 MB for `[32000, 384]`). At load, `weights.rs`
dequantizes it into an f32 matrix (~49 MB) for the embedding lookup and the full-vocab
float projection, and *also* keeps the raw int8 copy (~12 MB) for the shortlist int8
projection. So the embedding table is resident as roughly **~61 MB across two copies** for
something that is ~12 MB on disk — about a 4× expansion.

Nothing is wrong with the numerics; this is purely a footprint observation. It is the most
notable single allocation in the engine and a natural first thing to look at whenever the
memory pass ([11-dhat-report.md](./11-dhat-report.md)) happens.

Possible directions if it ever becomes worth it (not now): keep only the int8 table and
dequantize embedding rows on demand, or share one representation between the lookup and the
projection rather than holding both.

## Update: partly addressed by the dhat pass

The [11-dhat-report.md](./11-dhat-report.md) memory pass profiled this and found the peak
was actually **four** copies for shared-vocab models (f32 + int8, each duplicated into a
separate source slot). Two were pure redundancy and are now removed (see
[../06-memory-approach.md](../06-memory-approach.md)):

- shared vocab no longer clones the embedding for the source (`src_wemb: Option`, `None` =
  reuse the target table);
- the raw int8 is no longer copied out of the model at load — it's read back on demand for
  the int8 projection.

Result: shared-vocab peak **154.5 MB → 89.4 MB**, split-vocab **~175 MB → 150.8 MB**,
output byte-identical. The remaining f32↔int8 double (one f32 lookup table + the model's
one int8 copy) is the original ~4× expansion noted above and is **left as-is** — collapsing
it means dequantizing rows on demand (a hot-path change), which isn't worth it for a
one-time load cost.

Residual, still open (small): split-vocab keeps the model's `encoder_Wemb` int8 (~12 MB)
that is unused once the source f32 is extracted — that single tensor could be dropped after
load to trim CJK peak.
