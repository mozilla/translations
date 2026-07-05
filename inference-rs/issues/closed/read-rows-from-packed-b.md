# Read embedding rows out of gemmology's packed `Wemb` (drop the last weight double-copy)

**Open, scoped — worth doing.** Closes the one genuine weight duplication left in the fast
config: the raw int8 `Wemb` kept in `Model` solely so embedding lookups can read it, held
*alongside* the packed projection copy gemmology already owns. Continues
[wemb-dequant-memory.md](./wemb-dequant-memory.md) (which removed the f32 table and the redundant
int8 copies but deliberately left this last raw↔packed double). Motivated by on-device memory
sensitivity: ~15.6 MiB is ~27% of resident weights, well worth carrying a little layout coupling.

## The duplication

In the fast config (`lean-embed` + `gemmology`), `Wemb [32000×512]` int8 is resident **twice**:

1. **packed** in gemmology's register-blocked layout (`proj_pb`, 15.6 MiB), for the full-vocab
   output projection — built lazily on first projection; and
2. **raw** int8 in `Model`, because embedding *lookups* (`weights.rs::dequant_row`) read
   `raw[id*dim .. (id+1)*dim]` directly.

Affines are already held once (`prepare_affines` packs them and frees the raw bytes); `Wemb`
is the sole exception, ~15.6 MiB of pure duplication, because two code paths want two layouts.

## Why we can collapse it: the pack is a lossless, arch-generic reblocking

The shim packs via gemmology's `PrepareBQuantizedTransposed` (`gemmology.h:1201`). Its whole body
does **no arithmetic** — no re-quantization (input is already int8), no interleave, no transpose.
It is a pure register-wide **reblocking copy**, a bijection whose inverse is closed-form:

```cpp
const size_t RegisterElems = batch8::size;   // 16 NEON / 32 AVX2 / 64 AVX-512
const size_t kColStride = 8;                 // constant on every arch
for (r = 0; r < rows; r += kColStride)
  for (c = 0; c < cols; c += RegisterElems)
    for (ri = 0; ri < 8; ++ri)
      *out++ = *(const batch8*)(input + (r+ri)*cols + c);
```

Crucially it is **one arch-generic template** — the *only* thing that varies across architectures
is `RegisterElems` (the SIMD register width); `kColStride = 8` is constant. So the row-read inverse
is a **single formula parameterized by register width**, not a per-arch reimplementation. For
logical row `id` (vocab id) and channel `ch` (`0..dim`), with `W = RegisterElems`:

```
reg = ( (id / 8) * (k / W) + (ch / W) ) * 8 + (id % 8)
off = reg * W + (ch % W)
```

Gathering a full row `Wemb[id, :]` is `dim / W` contiguous W-byte copies — ~512 bytes/token,
negligible. Verified against the loop; it reproduces `raw[id*dim + ch]` exactly. One packed buffer
therefore serves **both** the projection and the embedding lookups: it *is* the transposed
`[vocab, dim]` int8 matrix, just reblocked.

## Shape of the change (rides the existing opportunistic-SIMD seam)

`PreparedB` is already split real-vs-stub on `--cfg gemmology_simd` (build.rs emits it only where a
kernel is compiled). This drops onto that seam so it stays opportunistic and per-arch-graceful,
exactly like the kernel itself:

- **Shim:** add `gemmology_read_row(handle, id, out /*len k*/)` that gathers the row via the index
  above (using the shim's `RegisterElems`), keeping layout knowledge next to the pack behind the C ABI.
- **`gemm.rs`:** the real `PreparedB` gains `read_row`; the scalar stub does not need it (no packed
  buffer on those builds).
- **`weights.rs`:** `dequant_row` reads through `PreparedB::read_row` **when a packed buffer exists**,
  else keeps reading `model.get(param).int8_transposed()`. On scalar/unwired builds nothing changes —
  there is no packed buffer and hence no double to begin with (projection also reads raw on demand).
- **Build `proj_pb` eagerly at load** (not lazily): embedding lookups happen on the first encode,
  before any projection, so the single copy must exist by then.
- Then the loader can **free the raw `Wemb`** on builds that have the packed buffer (`prepare_affines`
  currently excludes `embed_param`). That's the −15.6 MiB.

## Payoff

- **−15.6 MiB settled** where a SIMD kernel is built: resident weights drop from ~58 MiB to
  ~42 MiB — every weight then held exactly once. On-device-relevant; well above the ~5 MiB bar.
- Removes the last "held twice" line from the memory breakdown.

## Costs / risks

- **Couples to gemmology's pack layout**, which we otherwise treat as opaque — but it's *one*
  generic formula keyed on `RegisterElems`, not N. Mitigate in the house style with a **round-trip
  test**: pack a known matrix → read every row back → assert bit-equal to the raw int8, pinning the
  layout the way `gemm_parity` pins the multiply. If gemmology ever changes the pack, it fails loudly.
- **Wider registers tighten the gate.** The pack needs `k % RegisterElems == 0` (NEON 16, AVX2 32,
  AVX-512 64). `dim = 512` satisfies all three; a model whose `dim` isn't a multiple of the register
  width falls back to keeping the raw copy on that arch (same opportunistic pattern).
- Embedding lookup goes from one contiguous slice to `dim/W` indexed gathers — negligible, but it
  *is* a per-token hot path, so spot-check throughput stays flat.

Correctness gate: token-identical output + the existing oracle/batch-invariance parity (a pure
memory change must not move a logit). When [x86-gemmology-backend.md](./x86-gemmology-backend.md)
wires x86, `read_row` comes along for free (same formula, wider register), verified by its own
round-trip test.
