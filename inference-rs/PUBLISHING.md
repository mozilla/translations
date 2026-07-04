# Publishing plan — `fxtranslate` (and the `inference-rs` engine)

**Status: NOT published, and guarded against it** (`fxtranslate/Cargo.toml` has
`publish = false`). This is the durable plan (issue 14) for when we do publish. Nothing here
should be executed yet — it's a checklist to react to, and the open decisions at the end are for
the maintainer.

## What we'd be publishing

Two crates in this workspace:

- **`inference-rs`** — the engine library (+ a `translate`/`encode` binary). The fast default
  config is `lean-embed,gemmology,jemalloc`; it compiles vendored C++ (gemmology/xsimd) and is
  native-aarch64-only unless built `--no-default-features`.
- **`fxtranslate`** — the batteries-included CLI (this doc's subject): model discovery +
  download/cache + translate/REPL. Depends on `inference-rs` (currently a **path** dependency).

For `cargo install fxtranslate` to work, **both** crates must be on crates.io, because the CLI
links the engine. So publishing the CLI implies publishing (or otherwise resolving) the engine.

## Name availability

Checked with `cargo search` on 2026-07-03: **`fxtranslate`** and **`inference-rs`** both return no
results — i.e. both names appear free. (Re-check immediately before publishing; names get taken.)
`inference-rs` is a generic name — consider a less collision-prone engine crate name (e.g.
`fxtranslate-engine`) so the two ship as an obvious pair.

## Blockers to resolve before `cargo publish` (in rough priority)

1. **Vendored C++ lives outside the crate.** `build.rs` compiles `gemmology.h` / xsimd from
   `../inference/marian-fork/src/3rd_party/...` — a path *outside* the package directory. `cargo
   publish` only packages files under the crate root, so the published crate could not build the
   default (gemmology) config. Options:
   - **(a)** Vendor `gemmology.h` + the needed xsimd headers into `inference-rs/vendor/` and point
     `build.rs`/`GEMMOLOGY_DIR` at them (respecting their licenses); include them in the package.
   - **(b)** Publish with a **portable scalar default** (`default = []` or `["lean-embed"]`) and
     make `gemmology` an opt-in feature that requires the vendored headers or `GEMMOLOGY_DIR`.
     Cleaner for a public crate; the C++ SIMD path becomes opt-in. Pairs with
     [issue 18](./issues/18-gemmology-rust-port.md) (a pure-Rust kernel would remove this blocker
     entirely).
2. **Default features are native-aarch64-only.** `gemmology` (build.rs panics off-aarch64) and
   `jemalloc` (tikv-jemallocator, native-only) in `default` mean `cargo install` fails on other
   targets. For a public crate, default should build **everywhere** (scalar, system allocator) and
   gate the fast path behind features and/or `cfg(target_arch = "aarch64")`.
3. **Path dep → versioned dep.** `fxtranslate`'s `inference-rs = { path = ".." }` must become
   `{ version = "=X.Y.Z", path = ".." }` so the published CLI pins a real engine version.
4. **Publish order + `publish = false`.** Publish `inference-rs` first, then `fxtranslate`; flip the
   `publish = false` guard only at the moment of publishing.
5. **Metadata + license.** `inference-rs/Cargo.toml` has no `license`/`repository`/`readme`/
   `keywords`/`categories` — add them; confirm both crates carry `LICENSE` (repo is MPL-2.0).
   Verify the vendored gemmology/xsimd licenses permit redistribution and add attribution.
6. **Package hygiene.** Add `include`/`exclude` so the published tarball ships only source (exclude
   `corpora/`, `artifacts/`, `data/`, the numbered design docs, and the large test fixtures if not
   needed). `cargo package --list` to review.
7. **MSRV + CI.** Declare `rust-version`; add a CI job that builds the portable default on
   Linux/macOS and runs the offline tests, plus a `cargo publish --dry-run` for both crates.

## Dependency budget (fxtranslate)

"As dependency-free as possible" (issue 14), with the line drawn at what can't reasonably be
hand-rolled:

| dep | why | why not hand-rolled / why not heavier |
|---|---|---|
| `ureq` (rustls) | HTTPS to Remote Settings + the CDN | TLS is not hand-rollable; rustls avoids a system OpenSSL dep. Lighter than `reqwest` (no async/tokio). |
| `ruzstd` | decode zstd attachments | pure-Rust, decode-only — no C `zstd`/toolchain. |
| `tinyjson` | parse the RS records JSON | no `serde`/proc-macro; the response shape is simple and flat. |
| `sha2` | verify `decompressedHash` | audited; a hand-rolled hash isn't worth the risk. |
| `dirs` | platform-native cache directory | chosen over hand-rolling XDG so the cache lands in the *native* location per OS (`~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows), not an XDG path everywhere. |

Deliberately avoided: `reqwest`/`tokio` (async weight), `serde`/`serde_json` (proc-macro weight),
C-backed `zstd`, `clap` (arg parsing is hand-rolled).

## Steps to publish (DO NOT RUN YET)

```bash
# 0. Resolve blockers 1–7 above first.
# 1. Dry-run both, engine first.
cargo publish --dry-run -p inference-rs
cargo publish --dry-run -p fxtranslate     # after switching to a versioned engine dep
# 2. Review the tarball contents.
cargo package -p fxtranslate --list
# 3. Publish (flip publish=false first), engine then CLI.
cargo publish -p inference-rs
cargo publish -p fxtranslate
```

## Open decisions for the maintainer

- **Engine crate name**: keep `inference-rs`, or rename to something pair-obvious
  (`fxtranslate-engine`)? Publishing the CLI forces publishing the engine under *some* name.
- **Default features for the public crate**: portable scalar default (builds everywhere, slow) with
  the fast path opt-in, vs. keeping the native fast default and documenting the aarch64 requirement.
  Recommended: portable default + `fast`/`gemmology`/`jemalloc` opt-in features.
- **Vendor the C++** (blocker 1a) vs. wait for the pure-Rust kernel ([issue 18](./issues/18-gemmology-rust-port.md), 1b).
- **Versioning**: start both at `0.1.0`? Lockstep the versions or independent?
- **`--offline` mode / model pinning**: currently every translate hits Remote Settings to resolve
  the latest version before using the cache; a `--offline` flag (use whatever's cached) and/or
  pinning a model version in the cache would make the CLI usable without network. Worth deciding
  before a public release sets expectations.
