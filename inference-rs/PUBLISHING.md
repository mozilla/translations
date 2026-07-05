# Publishing plan — `fxtranslate` (engine) and `fxtranslate-cli`

**Status: NOT published yet.** The engine crate (`fxtranslate`) is structurally
publish-ready; `fxtranslate-cli` and `fxtranslate-oracle` carry `publish = false`
guards. This is the durable plan (issue 14) for when we do publish — a checklist
to react to, with the open decisions for the maintainer at the end.

## The workspace

Three crates under `crates/`:

- **`fxtranslate`** — the engine library. Portable pure-Rust scalar by default
  (builds on any target with no C++ toolchain); the native SIMD config is opt-in
  via `--features fast` (`lean-embed` + `gemmology`, aarch64 + C++17, headers
  vendored under `crates/fxtranslate/vendor/`). It also carries the
  batteries-included **model management** (Remote Settings discovery, verified
  cache, the pluggable `Fetch` client, language display names, and the
  `src→trg`→engine loader) behind the opt-in `download` feature — `net` adds the
  built-in `ureq` HTTP client. Both are off by default so the plain engine stays
  `memmap2`-only. This is the publishable crate.
- **`fxtranslate-cli`** — the batteries-included CLI (model discovery +
  download/cache + translate/REPL). Its binary is named `fxtranslate`; it depends
  on the engine with a versioned path dep, enabling the engine's `net` feature for
  model management, and inherits its fast-by-default config (SIMD where wired,
  scalar fallback elsewhere). `publish = false` until ready.
- **`fxtranslate-oracle`** — the marian-oracle validation harness (tolerance
  comparator, trace-replay bisector, the raw diagnostic binary, and the parity
  tests). Dev-only; `publish = false`, never published.

For `cargo install fxtranslate-cli` to work, both `fxtranslate` and
`fxtranslate-cli` must be on crates.io (the CLI links the engine). The binary is
`fxtranslate`, so users get `cargo install fxtranslate-cli` → an `fxtranslate`
command (the `git-delta` → `delta`, `fd-find` → `fd` pattern).

## Name availability

Re-check immediately before publishing (names get taken): `cargo search
fxtranslate` and `cargo search fxtranslate-cli`. As of the last check (issue 14),
`fxtranslate` was free. The generic `inference-rs` name has been dropped.

## Already resolved by the workspace reorg

- **Vendored C++ is inside the crate.** `gemmology.h` + the xsimd headers (with
  their LICENSE files) live under `crates/fxtranslate/vendor/`; `build.rs`
  resolves them from `CARGO_MANIFEST_DIR/vendor`, so `cargo publish` packages a
  buildable `fast` config. (`GEMMOLOGY_DIR`/`XSIMD_INCLUDE_DIR` still override.)
- **Fast by default, builds everywhere.** `default = ["fast"]` requests the SIMD
  kernel, but `build.rs` compiles it only where one is wired (aarch64 + a C++17
  toolchain) and otherwise falls back to the portable scalar kernel without
  failing the build — so `cargo install` never fails on non-aarch64 targets. A
  `portable` feature forces scalar (no C++). Wiring x86 SIMD is issue 24.
- **Versioned dependency.** `fxtranslate-cli` depends on
  `fxtranslate = { version = "=0.1.0", path = "../fxtranslate", features = ["net"] }`.
- **Publish guards + order are in place.** The engine is publishable; the CLI and
  oracle are `publish = false`. Publish the engine first, then (when ready) flip
  the CLI guard and publish it.
- **Oracle/dev weight is out of the library.** The trace-comparison harness and
  the jemalloc/dhat allocators live in `fxtranslate-oracle`, so the published
  engine and the shippable CLI cannot contain them.

## Remaining before `cargo publish`

1. **Engine metadata.** `crates/fxtranslate/Cargo.toml` has `license` +
   `description`; still needs `repository`, `readme`, `keywords`, `categories`.
   The `repository` value depends on where the crate will live (see open
   decisions).
2. **Engine README.** Add `crates/fxtranslate/README.md` (the repo-root README
   describes the whole project) and point `readme` at it.
3. **Package hygiene.** Add `include`/`exclude` so the tarball ships only source
   — exclude `corpora/`, `artifacts/`, `data/`, `notes/`, `issues/`, `scripts/`,
   and the large test fixtures if not needed. `cargo package -p fxtranslate
   --list` to review, and verify the vendored gemmology/xsimd LICENSE files are
   carried.
4. **MSRV + CI.** Declare `rust-version`; add CI that builds the portable engine
   on Linux/macOS (and ideally a non-aarch64 target to guard portability), runs
   the offline tests, and does `cargo publish --dry-run` for both crates.

## Dependency budget (model management)

"As dependency-free as possible" (issue 14), with the line drawn at what can't
reasonably be hand-rolled. These deps now live in the **engine** crate, all
**optional** and gated behind `download`/`net` — so the default engine adds only
`memmap2` (plus `cc` at build time under `fast`), and only a consumer that opts
into model management (the CLI, via `net`) pulls them:

| dep | feature | why | why not hand-rolled / why not heavier |
|---|---|---|---|
| `ureq` (rustls) | `net` | HTTPS to Remote Settings + the CDN | TLS is not hand-rollable; rustls avoids a system OpenSSL dep. Lighter than `reqwest` (no async/tokio). Gated so BYO-HTTP embedders skip it. |
| `ruzstd` | `download` | decode zstd attachments | pure-Rust, decode-only — no C `zstd`/toolchain. |
| `tinyjson` | `download` | parse the RS records JSON | no `serde`/proc-macro; the response shape is simple and flat. |
| `sha2` | `download` | verify `decompressedHash` | audited; a hand-rolled hash isn't worth the risk. |
| `dirs` | `download` | platform-native cache directory | native cache location per OS, not an XDG path everywhere. |

Deliberately avoided: `reqwest`/`tokio`, `serde`/`serde_json`, C-backed `zstd`,
`clap` (arg parsing is hand-rolled, and stays in `fxtranslate-cli`).

## Steps to publish

Use `scripts/publish.py` — it automates the sequence below: bump the workspace
version (lockstep), validate packaging, publish the publishable crates in
dependency order (engine first), then create + push the `fxtranslate-vX.Y.Z` tag
only after every crate is up (crates.io first, atomic tag last — see the script's
header for the why). It reads which crates to publish from the manifests, so it
publishes only `fxtranslate` today and picks up `fxtranslate-cli` automatically
once its guard is removed (item below).

```bash
# 0. Resolve the remaining items above first.
# 1. Preview — read-only: prints the plan, bumps nothing, publishes nothing.
inference-rs/scripts/publish.py patch --dry-run
# 2. Release for real: bump + test + publish + tag + push.
inference-rs/scripts/publish.py patch     # or minor / major / --set X.Y.Z
```

The underlying cargo commands, for reference / manual fallback:

```bash
cargo publish --dry-run -p fxtranslate       # + fxtranslate-cli once unguarded
cargo package  --list    -p fxtranslate      # review the tarball contents
cargo publish            -p fxtranslate       # engine first
cargo publish            -p fxtranslate-cli   # then the CLI (after flipping its guard)
```

## Open decisions for the maintainer

- **Where the engine lives**: keep it in this monorepo, or split `fxtranslate`
  into its own repository (the `fxhash`-style standalone goal)? This sets the
  `repository` metadata and the README's framing.
- **x86 speed**: wire up gemmology's existing x86 SIMD kernels (issue 24) so
  `cargo install` is fast on x86 too, or wait for the pure-Rust kernel (issue 18)?
  Until then x86 installs work but use the scalar fallback.
- **Versioning** *(resolved: lockstep)*: the workspace crates share one version and
  bump together, with `fxtranslate-cli` pinning the engine exactly (`= X.Y.Z`), so a
  CLI release always links the engine it was validated against. `scripts/publish.py`
  enforces this (it refuses to run if the versions have drifted). Revisit only if the
  crates ever need to release on independent cadences.
- **`--offline` mode / model pinning**: every translate currently resolves the
  latest version from Remote Settings before using the cache; an `--offline` flag
  (use whatever's cached) and/or a pinned model version would make the CLI usable
  without network. Worth deciding before a public release sets expectations.
