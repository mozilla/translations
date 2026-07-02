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
