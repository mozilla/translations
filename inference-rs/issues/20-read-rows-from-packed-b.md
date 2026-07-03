# Read embedding rows out of gemmology's packed `Wemb` (drop the last weight double-copy)

**Open, unscheduled follow-on.** Closes the one genuine weight duplication left in the fast
config: the raw int8 `Wemb` kept in `Model` solely so embedding lookups can read it, held
*alongside* the packed projection copy gemmology already owns. Continues
[16-wemb-dequant-memory.md](./16-wemb-dequant-memory.md) (which removed the f32 table and the
redundant int8 copies but deliberately left this last raw↔packed double) and complements
[19-settled-rss-allocator.md](./19-settled-rss-allocator.md) / `--mmap`. See the memory
breakdown in [../09-final-comparison.md](../09-final-comparison.md) ("How the memory inflates").

## The duplication

In the fast config (`lean-embed,gemmology`), `Wemb [32000×512]` int8 is resident **twice**:

1. **packed** in gemmology's register-blocked layout (`proj_pb`, 15.6 MiB), for the full-vocab
   output projection — built lazily on first projection; and
2. **raw** int8 in `Model`, because embedding *lookups* (`weights.rs::dequant_row`) read
   `raw[id*dim .. (id+1)*dim]` directly.

Affines are already held once (`prepare_affines` packs them and frees the raw bytes); `Wemb`
is the sole exception, ~15.6 MiB of pure duplication, because two code paths want two layouts.

## Why we can collapse it: the transposed pack is a lossless reblocking

The shim packs via gemmology's `PrepareBQuantizedTransposed` (`gemmology.h:1201`). Its whole body:

```cpp
for (size_t r = 0; r < rows; r += 8)      // rows = n_pad (= vocab), step 8
  for (size_t c = 0; c < cols; c += 16)   // cols = k   (= dim),  step 16
    for (size_t ri = 0; ri < 8; ++ri)
      *output_it++ = *(const batch8*)(input + (r+ri)*cols + c);   // copy 16 int8
```

Crucially it does **no arithmetic** — no re-quantization (input is already int8), no
`interleave`, no `Transpose16InLane` (unlike the *float* `PrepareB` path, lines 1258–1271). It
is a pure 16-wide **reblocking copy** of the bytes: a bijection whose inverse is closed-form.

For logical row `id` (vocab id) and channel `ch` (`0..dim`), with register width 16 and column
stride 8:

```
reg         = ( (id / 8) * (k / 16) + (ch / 16) ) * 8 + (id % 8)
byte_offset = reg * 16 + (ch % 16)
```

Gathering a full row `Wemb[id, :]` (dim=512 → 32 chunks) is 32 contiguous 16-byte copies with
that index — ~512 bytes/token, negligible. Verified against the loop; it reproduces
`raw[id*dim + ch]` exactly. Our dims cooperate: `dim=512` is a multiple of 16, `vocab=32000` a
multiple of 8, so `n_pad == vocab` — no padding, no edge cases (padded rows, when they exist,
are zeros and are never looked up since ids `< vocab`).

One packed buffer therefore serves **both** the projection and the embedding lookups: it *is*
the transposed `[vocab, dim]` int8 matrix, just reblocked.

## Shape of the change

- **Shim:** add `gemmology_read_row(handle, id, out /*len k*/)` that gathers the row via the
  index above (keep the layout knowledge next to the pack that defines it, behind the C ABI).
- **`weights.rs`:** `dequant_row` reads through `gemmology_read_row` instead of
  `model.get(param).int8_transposed()`.
- **Build `proj_pb` eagerly at load**, not lazily — embedding lookups happen on the first
  encode, before any projection, so the single copy must exist by then.
- Then `prepare_affines` (or the loader) can **free the raw `Wemb`** too (it currently excludes
  `embed_param`). That's the −15.6 MiB.

## Payoff

- **−15.6 MiB settled** (subject to the same libmalloc/`--mmap` caveats as
  [19](./19-settled-rss-allocator.md): with owned load the freed pages may linger; with `--mmap`
  the raw `Wemb` was a clean file-backed view, so freeing its handle lets those pages go).
  Resident weights drop from ~58 MiB to ~42 MiB — **every** weight then held exactly once.
- Removes the last "held twice" line from the memory breakdown.

## Costs / risks

- **Couples us to gemmology's internal layout**, which we otherwise treat as opaque. Mitigate
  in the house style with a **round-trip test**: pack a known matrix → read every row back →
  assert bit-equal to the raw int8, pinning the layout the way `gemm_parity` pins the multiply.
  If gemmology ever changes the pack, that test fails loudly.
- Layout is valid only for the fixed `Arch` (neon64, 16-wide register, 8-col stride); guard with
  a `static_assert` / build-time check.
- Embedding lookup goes from one contiguous slice to 32 indexed 16-byte gathers — negligible,
  but it *is* a per-token hot path, so spot-check throughput (expect flat).

## Generalization (why this is a reusable pattern)

What makes it work is that the pack is a **byte-bijection** (permutation/reblocking), not a
fused/lossy transform. The clean generic interface: a matrix backend optionally exposes
`packed_offset(row, col) -> usize` (or a `read_row`); if present, callers drop the second copy;
if absent (a pack that folds in the quant multiplier or interleaves irreversibly), fall back to
keeping the raw copy. gemmology's transposed path qualifies; its float `PrepareB` path is also a
bijection but needs the interleave/transpose inverted too. Pairs naturally with the Rust
gemmology port ([18-gemmology-rust-port.md](./18-gemmology-rust-port.md)): a Rust-owned packed
buffer with a `read_row` method is one representation, no C++ blind spot, no double.
