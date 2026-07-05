#!/usr/bin/env python3
"""
Release build + size characterization + cheat-proof validation (issue 13).

Builds the release artifacts and reports their size, then — unless
`--skip-validation` — proves the engine build:

  1. the shippable CLI (`fxtranslate`, from fxtranslate-cli) is lean: the
     marian-oracle diagnostics (trace/replay) and the dhat allocator live in the
     separate fxtranslate-oracle crate, so they cannot leak into the product;
  2. the engine runs correctly: a committed corpus piped through the release
     oracle binary matches the debug build exactly (a faithful, non-miscompiled
     build of the oracle-validated engine), and its greedy output is compared to
     the reference `translator-cli` as a tracking metric. The parity number needs
     the C++ oracle built and the pair's model downloaded; otherwise it is
     skipped (use --skip-validation to bypass validation entirely).

Usage:
    inference-rs/scripts/release_build.py                 # build + validate (en-fr)
    inference-rs/scripts/release_build.py --pair en ru
    inference-rs/scripts/release_build.py --bloat         # + cargo bloat breakdown
    inference-rs/scripts/release_build.py --skip-validation
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import translate_common as common

CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CRATE_DIR.parent
MANIFEST = CRATE_DIR / "Cargo.toml"
TARGET = CRATE_DIR / "target"
TRANSLATOR_CLI = REPO_ROOT / "inference/build/src/app/translator-cli"
DEFAULT_CORPUS = CRATE_DIR / "corpora/dev-en.txt"

# The shippable product and the dev/validation binary.
CLI_BIN = TARGET / "release" / "fxtranslate"
ORACLE_BIN = TARGET / "release" / "fxtranslate-oracle"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB ({n:,} B)"


def build(pkg_args: list[str]) -> None:
    cmd = ["cargo", "build", "--release", "--manifest-path", str(MANIFEST)] + pkg_args
    print(f"[build] {' '.join(cmd)}", file=sys.stderr)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"[build] failed: {' '.join(cmd)}")


def size_report(bloat: bool) -> None:
    print("\n== size ==")
    for name, binp in (("fxtranslate (CLI)", CLI_BIN), ("fxtranslate-oracle", ORACLE_BIN)):
        if binp.exists():
            print(f"  {name:20} {mib(binp.stat().st_size)}")

    if bloat:
        if shutil.which("cargo-bloat"):
            print("\n== cargo bloat (fxtranslate-cli, top crates) ==")
            r = sh(
                [
                    "cargo",
                    "bloat",
                    "--release",
                    "--manifest-path",
                    str(MANIFEST),
                    "-p",
                    "fxtranslate-cli",
                    "--crates",
                    "-n",
                    "15",
                ]
            )
            print(r.stdout or r.stderr)
        else:
            print("\n[bloat] cargo-bloat not installed (cargo install cargo-bloat); skipping")


def validate_lean() -> list[str]:
    """The product CLI must carry none of the oracle diagnostics or the dhat
    allocator. These live in the fxtranslate-oracle crate, so a no-args run of the
    CLI (which prints its usage) must never mention trace/replay or the dhat
    banner. A structural invariant now — this guards against it regressing."""
    fails = []
    out = sh([str(CLI_BIN)])  # no args -> usage on stderr
    blob = out.stdout + out.stderr
    if "replay" in blob or "trace <" in blob:
        fails.append("CLI usage lists trace/replay (oracle diagnostics leaked into the product)")
    if "[dhat]" in blob:
        fails.append("dhat allocator banner present in the product CLI")
    return fails


def validate_correct(src: str, trg: str, limit: int) -> list[str]:
    """Cheat-proof correctness of the engine artifact, two parts:

    - GATE: the release oracle binary reproduces the debug build's output exactly.
      The engine is validated against the marian oracle by the test suite; this
      proves the optimized release artifact is a *faithful* build of that engine
      (not broken/miscompiled). Deterministic 100% — the thing checked is the
      build, not the algorithm.
    - REPORT (not a gate): release-vs-`translator-cli` exact-match rate — the same
      oracle tracking metric parity.py reports (greedy output is not 100%
      identical to marian on every sentence, so this is informational)."""
    try:
        _s, _t, _langs, config = common.resolve_config(common.DEFAULT_MODELS_DIR, src, trg)
    except SystemExit as e:
        return [
            f"correctness: model for {src}-{trg} not downloaded ({e}); "
            "download it or pass --skip-validation"
        ]

    mc = common.parse_model_config(config)
    vocabs = mc["vocabs"]
    src_v = str(vocabs[0])
    trg_v = str(vocabs[1] if len(vocabs) > 1 else vocabs[0])
    lines = [l for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()][:limit]
    text = "\n".join(lines) + "\n"

    rel = sh(
        [str(ORACLE_BIN), "translate", str(mc["model"]), src_v, trg_v], input=text
    ).stdout.splitlines()
    dbg = sh(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "fxtranslate-oracle",
            "--features",
            "fast",
            "--manifest-path",
            str(MANIFEST),
            "--",
            "translate",
            str(mc["model"]),
            src_v,
            trg_v,
        ],
        input=text,
    ).stdout.splitlines()

    fails = []
    print(f"\n== correctness (artifact, {src}-{trg}) ==")
    if rel == dbg and len(rel) == len(lines):
        print(f"  GATE ok: release == debug on all {len(lines)} sentences (faithful build)")
    else:
        n = min(len(rel), len(dbg))
        diffs = [i for i in range(n) if rel[i] != dbg[i]]
        fails.append(
            f"release output differs from debug build "
            f"({len(diffs)} lines, or line-count {len(rel)} vs {len(dbg)} vs {len(lines)} in)"
        )

    # Oracle tracking metric (reported, never gates).
    if TRANSLATOR_CLI.exists():
        import tempfile

        kept = [
            l for l in config.read_text().splitlines() if not l.strip().startswith("shortlist:")
        ]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
        tmp.write("\n".join(kept) + "\n")
        tmp.close()
        try:
            ref = sh(
                [str(TRANSLATOR_CLI), "--model-config-paths", tmp.name, "--cpu-threads", "1"],
                input=text,
            ).stdout.splitlines()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        n = min(len(ref), len(rel))
        matches = sum(1 for i in range(n) if ref[i] == rel[i])
        print(f"  oracle parity (tracking, non-gating): {matches}/{n} exact vs translator-cli")
    else:
        print(f"  oracle parity: skipped (translator-cli not built at {TRANSLATOR_CLI})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pair", nargs=2, metavar=("SRC", "TRG"), default=["en", "fr"])
    ap.add_argument("--limit", type=int, default=20, help="validation corpus sentence cap")
    ap.add_argument("--bloat", action="store_true", help="add a cargo bloat crate breakdown")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    build(["-p", "fxtranslate-cli"])  # the shippable product (portable)
    build(["-p", "fxtranslate-oracle", "--features", "fast"])  # the native validation engine
    size_report(args.bloat)

    if args.skip_validation:
        print("\n[validation] skipped (--skip-validation)")
        return

    fails = validate_lean()
    fails += validate_correct(args.pair[0], args.pair[1], args.limit)

    print("\n== validation ==")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("  PASS: product CLI is lean and the release engine is a faithful build")


if __name__ == "__main__":
    main()
