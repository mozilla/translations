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
  vendored under `crates/fxtranslate/vendor/`). This is the publishable crate.
- **`fxtranslate-cli`** — the batteries-included CLI (model discovery +
  download/cache + translate/REPL). Its binary is named `fxtranslate`; it depends
  on the engine with a versioned path dep and inherits its fast-by-default config
  (SIMD where wired, scalar fallback elsewhere). `publish = false` until ready.
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
  `fxtranslate = { version = "=0.1.0", path = "../fxtranslate" }`.
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

## Dependency budget (fxtranslate-cli)

"As dependency-free as possible" (issue 14), with the line drawn at what can't
reasonably be hand-rolled:

| dep | why | why not hand-rolled / why not heavier |
|---|---|---|
| `ureq` (rustls) | HTTPS to Remote Settings + the CDN | TLS is not hand-rollable; rustls avoids a system OpenSSL dep. Lighter than `reqwest` (no async/tokio). |
| `ruzstd` | decode zstd attachments | pure-Rust, decode-only — no C `zstd`/toolchain. |
| `tinyjson` | parse the RS records JSON | no `serde`/proc-macro; the response shape is simple and flat. |
| `sha2` | verify `decompressedHash` | audited; a hand-rolled hash isn't worth the risk. |
| `dirs` | platform-native cache directory | native cache location per OS, not an XDG path everywhere. |

Deliberately avoided: `reqwest`/`tokio`, `serde`/`serde_json`, C-backed `zstd`,
`clap` (arg parsing is hand-rolled). The engine library adds only `memmap2`
(plus `cc` at build time under `fast`).

## Steps to publish (DO NOT RUN YET)

```bash
# 0. Resolve the remaining items above first.
# 1. Dry-run both, engine first.
cargo publish --dry-run -p fxtranslate
cargo publish --dry-run -p fxtranslate-cli
# 2. Review the tarball contents.
cargo package -p fxtranslate --list
cargo package -p fxtranslate-cli --list
# 3. Publish the engine, then flip fxtranslate-cli's publish guard and publish it.
cargo publish -p fxtranslate
cargo publish -p fxtranslate-cli
```

## Open decisions for the maintainer

- **Where the engine lives**: keep it in this monorepo, or split `fxtranslate`
  into its own repository (the `fxhash`-style standalone goal)? This sets the
  `repository` metadata and the README's framing.
- **x86 speed**: wire up gemmology's existing x86 SIMD kernels (issue 24) so
  `cargo install` is fast on x86 too, or wait for the pure-Rust kernel (issue 18)?
  Until then x86 installs work but use the scalar fallback.
- **Versioning**: start both at `0.1.0`; lockstep the versions or let them move
  independently?
- **`--offline` mode / model pinning**: every translate currently resolves the
  latest version from Remote Settings before using the cache; an `--offline` flag
  (use whatever's cached) and/or a pinned model version would make the CLI usable
  without network. Worth deciding before a public release sets expectations.
