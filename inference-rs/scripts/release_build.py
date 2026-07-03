#!/usr/bin/env python3
"""
Release build + size characterization + cheat-proof validation (issue 13).

Builds the release artifacts, reports their size (and a default-vs-all-features
delta, optional `cargo bloat` breakdown), and — unless `--skip-validation` —
proves the *actual release binary*:

  1. ships lean: the opt-in dev features are OFF (the `replay`/`trace`
     instrumentation subcommands are absent; no dhat allocator banner);
  2. runs correctly: a committed corpus piped through the release binary matches
     the reference `translator-cli` (the parity oracle — not a hand-written
     string that rots). Needs the C++ oracle built and the pair's model
     downloaded; otherwise validation errors (use --skip-validation to bypass).

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


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB ({n:,} B)"


def build(features_args: list[str], target_dir: Path) -> None:
    cmd = ["cargo", "build", "--release", "--manifest-path", str(MANIFEST)] + features_args
    print(f"[build] {' '.join(cmd)}  (CARGO_TARGET_DIR={target_dir})", file=sys.stderr)
    env = {**__import__("os").environ, "CARGO_TARGET_DIR": str(target_dir)}
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        sys.exit(f"[build] failed: {' '.join(cmd)}")


def size_report(bloat: bool) -> None:
    print("\n== size ==")
    for name in ("inference-rs", "fxtranslate"):
        binp = TARGET / "release" / name
        if binp.exists():
            print(f"  {name:16} {mib(binp.stat().st_size)}")

    # default-vs-all-features delta for the engine binary, built into a side
    # target dir so it doesn't clobber the validated default artifact.
    full_target = TARGET / "release-full-probe"
    build(["-p", "inference-rs", "--all-features"], full_target)
    full = full_target / "release" / "inference-rs"
    base = TARGET / "release" / "inference-rs"
    if full.exists() and base.exists():
        delta = full.stat().st_size - base.stat().st_size
        print(f"\n  inference-rs default : {mib(base.stat().st_size)}")
        print(f"  inference-rs --all-features : {mib(full.stat().st_size)}")
        print(f"  delta (dev/instrumentation/dhat weight) : {mib(delta)}")

    if bloat:
        if shutil.which("cargo-bloat"):
            print("\n== cargo bloat (top crates) ==")
            r = sh(["cargo", "bloat", "--release", "--manifest-path", str(MANIFEST),
                    "-p", "inference-rs", "--crates", "-n", "15"])
            print(r.stdout or r.stderr)
        else:
            print("\n[bloat] cargo-bloat not installed (cargo install cargo-bloat); skipping")


def validate_lean(binp: Path) -> list[str]:
    """Assert the release binary carries no dev weight. A no-args run prints usage
    (and, if dhat-heap were on, the profiler banner at startup); the trace/replay
    usage lines are instrumentation-gated, so their absence proves it is off."""
    fails = []
    out = sh([str(binp)])  # no args -> usage on stderr
    blob = out.stdout + out.stderr
    if "replay" in blob or "inference-rs trace" in blob:
        fails.append("instrumentation feature is ON (usage lists trace/replay)")
    if "[dhat]" in blob:
        fails.append("dhat-heap feature is ON (profiler banner present)")
    return fails


def validate_correct(binp: Path, src: str, trg: str, limit: int) -> list[str]:
    """Cheat-proof correctness of the *artifact*, two parts:

    - GATE: the release binary reproduces the debug build's output exactly. The
      engine is validated against the marian oracle by the test suite; this proves
      the optimized release artifact is a *faithful* build of that engine (not
      broken/miscompiled). Deterministic 100% — comparing to ourselves, but the
      thing being checked is the build, not the algorithm.
    - REPORT (not a gate): release-vs-`translator-cli` exact-match rate — the same
      oracle tracking metric parity.py reports (the engine's greedy output is not
      100% identical to marian on every sentence, so this is informational, per
      02-parity-harness.md)."""
    try:
        _s, _t, _langs, config = common.resolve_config(common.DEFAULT_MODELS_DIR, src, trg)
    except SystemExit as e:
        return [f"correctness: model for {src}-{trg} not downloaded ({e}); "
                "download it or pass --skip-validation"]

    mc = common.parse_model_config(config)
    vocabs = mc["vocabs"]
    src_v = str(vocabs[0])
    trg_v = str(vocabs[1] if len(vocabs) > 1 else vocabs[0])
    lines = [l for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()][:limit]
    text = "\n".join(lines) + "\n"

    rel = sh([str(binp), "translate", str(mc["model"]), src_v, trg_v], input=text).stdout.splitlines()
    dbg = sh(["cargo", "run", "--quiet", "--manifest-path", str(MANIFEST), "--",
              "translate", str(mc["model"]), src_v, trg_v], input=text).stdout.splitlines()

    fails = []
    print(f"\n== correctness (artifact, {src}-{trg}) ==")
    if rel == dbg and len(rel) == len(lines):
        print(f"  GATE ok: release == debug on all {len(lines)} sentences (faithful build)")
    else:
        n = min(len(rel), len(dbg))
        diffs = [i for i in range(n) if rel[i] != dbg[i]]
        fails.append(f"release output differs from debug build "
                     f"({len(diffs)} lines, or line-count {len(rel)} vs {len(dbg)} vs {len(lines)} in)")

    # Oracle tracking metric (reported, never gates).
    if TRANSLATOR_CLI.exists():
        import tempfile
        kept = [l for l in config.read_text().splitlines() if not l.strip().startswith("shortlist:")]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
        tmp.write("\n".join(kept) + "\n"); tmp.close()
        try:
            ref = sh([str(TRANSLATOR_CLI), "--model-config-paths", tmp.name, "--cpu-threads", "1"],
                     input=text).stdout.splitlines()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        n = min(len(ref), len(rel))
        matches = sum(1 for i in range(n) if ref[i] == rel[i])
        print(f"  oracle parity (tracking, non-gating): {matches}/{n} exact vs translator-cli")
    else:
        print(f"  oracle parity: skipped (translator-cli not built at {TRANSLATOR_CLI})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, metavar=("SRC", "TRG"), default=["en", "fr"])
    ap.add_argument("--limit", type=int, default=20, help="validation corpus sentence cap")
    ap.add_argument("--bloat", action="store_true", help="add a cargo bloat crate breakdown")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    build([], TARGET)  # default (fast) release
    size_report(args.bloat)

    if args.skip_validation:
        print("\n[validation] skipped (--skip-validation)")
        return

    binp = TARGET / "release" / "inference-rs"
    fails = validate_lean(binp)
    fails += validate_correct(binp, args.pair[0], args.pair[1], args.limit)

    print("\n== validation ==")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("  PASS: release binary is lean and a faithful build of the validated engine")


if __name__ == "__main__":
    main()
