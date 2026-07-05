#!/usr/bin/env python3
"""
Publish the fxtranslate crates to crates.io (issues/publish-script.md).

Bumps the workspace version, validates packaging, publishes the publishable crates
in dependency order, and tags the release. `--dry-run` does everything read-only:
it validates packaging and prints the exact plan without touching files, crates.io,
or git.

Versioning policy — LOCKSTEP. The workspace crates share one version and bump
together, and fxtranslate-cli pins the engine exactly (`fxtranslate = "=X.Y.Z"`), so
a CLI release always links the engine it was built and validated against. One number
for the whole workspace is simpler to reason about than independent drift, and the
CLI is useless without a matching engine anyway. (This settles the open "lockstep vs.
independent" question in PUBLISHING.md.)

Which crates publish is read from the manifests, not hard-coded: any workspace member
without `publish = false` is published. Today that's `fxtranslate` (the engine) and
`fxtranslate-cli`, published in that order (the CLI pins the engine exactly, so the
engine must land on crates.io first); only the dev-only `fxtranslate-oracle` keeps
the guard.

Ordering — crates.io first, tag last. Publishing N crates to crates.io is not atomic
(the engine can land, then the CLI fail), and an upload can't be taken back (only
yanked). The git tag, by contrast, is a single atomic ref. So we publish every crate
first and create + push the tag ONLY once they all succeed — the tag therefore never
points at a half-published release. A re-run after a partial failure skips crates
whose version is already on crates.io and proceeds to the rest, then tags.

Usage:
    inference-rs/scripts/publish.py patch --dry-run     # preview a 0.1.0 -> 0.1.1 release
    inference-rs/scripts/publish.py minor --dry-run
    inference-rs/scripts/publish.py major
    inference-rs/scripts/publish.py --set 1.2.3         # explicit version
    inference-rs/scripts/publish.py patch               # real release (tests, publish, tag, push)
"""

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent  # inference-rs/
ROOT_MANIFEST = WORKSPACE / "Cargo.toml"
TAG_PREFIX = "fxtranslate-v"  # bare `0.1.0` etc. are taken by old repo tags

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def log(msg: str) -> None:
    print(f"[publish] {msg}", file=sys.stderr)


def die(msg: str) -> None:
    sys.exit(f"[publish] error: {msg}")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run capturing stdout+stderr as text (for inspection)."""
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def run(cmd: list[str], **kw) -> None:
    """Run streaming to the terminal; exit on failure."""
    log("$ " + " ".join(cmd))
    if subprocess.run(cmd, **kw).returncode != 0:
        die("command failed: " + " ".join(cmd))


# --- workspace model ---------------------------------------------------------


class Crate:
    def __init__(self, name: str, version: str, publishable: bool, manifest: Path, deps: set[str]):
        self.name = name
        self.version = version
        self.publishable = publishable
        self.manifest = manifest
        self.deps = deps  # intra-workspace dependency crate names

    def __repr__(self) -> str:
        return f"Crate({self.name} {self.version} publishable={self.publishable})"


def load_workspace() -> list[Crate]:
    root = tomllib.loads(ROOT_MANIFEST.read_text())
    members = root.get("workspace", {}).get("members", [])
    if not members:
        die(f"no [workspace].members in {ROOT_MANIFEST}")

    by_dir: dict[str, str] = {}  # member dir -> crate name (to resolve path deps)
    raw: list[tuple[dict, Path]] = []
    for m in members:
        manifest = WORKSPACE / m / "Cargo.toml"
        if not manifest.is_file():
            die(f"workspace member {m} has no Cargo.toml at {manifest}")
        data = tomllib.loads(manifest.read_text())
        by_dir[m] = data["package"]["name"]
        raw.append((data, manifest))

    crates: list[Crate] = []
    for data, manifest in raw:
        pkg = data["package"]
        # `publish = false` (or an empty registry allowlist) means "never publish".
        publish = pkg.get("publish", True)
        publishable = publish is not False and publish != []
        # Intra-workspace deps: a dependency whose table carries a `path`.
        deps: set[str] = set()
        for section in ("dependencies", "build-dependencies"):
            for dname, spec in data.get(section, {}).items():
                if isinstance(spec, dict) and "path" in spec:
                    deps.add(dname)
        crates.append(Crate(pkg["name"], pkg["version"], publishable, manifest, deps))
    return crates


def publish_order(crates: list[Crate]) -> list[Crate]:
    """Publishable crates, dependencies before dependents (topological)."""
    publishable = {c.name: c for c in crates if c.publishable}
    ordered: list[Crate] = []
    seen: set[str] = set()

    def visit(c: Crate, stack: set[str]) -> None:
        if c.name in seen:
            return
        if c.name in stack:
            die(f"dependency cycle through {c.name}")
        stack.add(c.name)
        for dep in sorted(c.deps):
            if dep in publishable:
                visit(publishable[dep], stack)
        stack.discard(c.name)
        seen.add(c.name)
        ordered.append(c)

    for c in sorted(publishable.values(), key=lambda c: c.name):
        visit(c, set())
    return ordered


# --- versioning --------------------------------------------------------------


def current_version(crates: list[Crate]) -> str:
    """The single lockstep version — refuse to proceed if the crates have drifted."""
    versions = {c.version for c in crates}
    if len(versions) != 1:
        detail = ", ".join(f"{c.name}={c.version}" for c in crates)
        die(f"crate versions have drifted ({detail}); lockstep is required — reconcile first")
    return versions.pop()


def bump(version: str, level: str) -> str:
    m = SEMVER.match(version)
    if not m:
        die(f"current version {version!r} is not a plain MAJOR.MINOR.PATCH; use --set")
    major, minor, patch = (int(g) for g in m.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def rewrite_versions(crates: list[Crate], old: str, new: str, apply: bool) -> list[str]:
    """Set every crate's package version to `new`, and every exact intra-workspace
    pin (`version = "=OLD"`) to `=NEW`. Returns human-readable descriptions of each
    edit; only writes when `apply`."""
    changes: list[str] = []
    pkg_re = re.compile(rf'^version = "{re.escape(old)}"$', re.MULTILINE)
    pin_old, pin_new = f'version = "={old}"', f'version = "={new}"'
    for c in crates:
        text = c.manifest.read_text()
        new_text, n_pkg = pkg_re.subn(f'version = "{new}"', text, count=1)
        n_pin = new_text.count(pin_old)
        new_text = new_text.replace(pin_old, pin_new)
        rel = c.manifest.relative_to(WORKSPACE)
        if n_pkg:
            changes.append(f"{rel}: package version {old} -> {new}")
        else:
            die(f'{rel}: could not find `version = "{old}"` to bump')
        if n_pin:
            changes.append(f"{rel}: {n_pin} exact pin(s) ={old} -> ={new}")
        if apply:
            c.manifest.write_text(new_text)
    return changes


# --- README hygiene ----------------------------------------------------------


def readme_report(crates: list[Crate], old: str) -> None:
    """Publishable crates should ship a README (cargo warns otherwise), and stale
    version strings in any README want a look before release. Report both — never
    silently rewrite README prose, since a bare version string is easy to false-match."""
    for c in crates:
        if c.publishable:
            data = tomllib.loads(c.manifest.read_text())
            has_field = "readme" in data.get("package", {})
            has_file = (c.manifest.parent / "README.md").is_file()
            if not (has_field or has_file):
                log(f"WARNING: {c.name} has no README.md / readme = (crates.io recommends one)")

    for readme in sorted(WORKSPACE.glob("**/README.md")):
        if "target" in readme.parts:
            continue
        hits = [
            f"    {readme.relative_to(WORKSPACE)}:{i}: {line.strip()}"
            for i, line in enumerate(readme.read_text().splitlines(), 1)
            if old in line
        ]
        if hits:
            log(f"README mentions the old version {old} — review before publishing:")
            for h in hits:
                print(h, file=sys.stderr)


# --- preflight ---------------------------------------------------------------


def git(*args: str) -> str:
    r = sh(["git", "-C", str(WORKSPACE), *args])
    if r.returncode != 0:
        die(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def preflight(allow_dirty: bool) -> None:
    if (
        not (WORKSPACE / ".git").exists()
        and not sh(["git", "-C", str(WORKSPACE), "rev-parse"]).returncode == 0
    ):
        die("not inside a git repository")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        log(f"WARNING: on branch {branch!r}, not main")
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        die("working tree is dirty; commit/stash first (or pass --allow-dirty)")


# --- publish steps -----------------------------------------------------------


def validate_packaging(order: list[Crate]) -> None:
    """Dry-run packaging for each publishable crate (catches missing files, bad
    metadata). Uses the committed manifests — for a dependent crate this needs its
    workspace dependency already on crates.io, so a not-yet-published dep is reported,
    not treated as fatal."""
    for c in order:
        log(f"cargo package --list -p {c.name}")
        r = sh(["cargo", "package", "--list", "-p", c.name, "--manifest-path", str(ROOT_MANIFEST)])
        if r.returncode != 0:
            log(f"  (package --list reported: {r.stderr.strip().splitlines()[-1:] })")
        else:
            log(f"  {len(r.stdout.splitlines())} files would be packaged")
        log(f"cargo publish --dry-run -p {c.name}")
        # `--allow-dirty`: cleanliness is the real run's concern (preflight); a
        # dry-run must validate packaging regardless of unrelated working changes.
        r = sh(
            [
                "cargo",
                "publish",
                "--dry-run",
                "--allow-dirty",
                "-p",
                c.name,
                "--manifest-path",
                str(ROOT_MANIFEST),
            ]
        )
        if r.returncode != 0:
            tail = "\n".join(r.stderr.strip().splitlines()[-4:])
            log(f"  dry-run did not pass (fine if it needs a not-yet-published dep):\n{tail}")
        else:
            log("  dry-run OK")


def publish_crate(crate: str) -> None:
    """Publish one crate; treat 'already uploaded' as success so a re-run is safe."""
    log(f"cargo publish -p {crate}")
    r = subprocess.run(
        ["cargo", "publish", "-p", crate, "--manifest-path", str(ROOT_MANIFEST)],
        text=True,
        capture_output=True,
    )
    sys.stderr.write(r.stderr)
    if r.returncode == 0:
        return
    if "already uploaded" in r.stderr or "already exists" in r.stderr:
        log(f"  {crate} is already on crates.io at this version; skipping")
        return
    die(
        f"publishing {crate} failed (see above); crates already published stay up — "
        f"fix and re-run to publish the rest, then the tag is created"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "level", nargs="?", choices=["major", "minor", "patch"], help="semver component to bump"
    )
    ap.add_argument(
        "--set",
        dest="set_version",
        metavar="X.Y.Z",
        help="set an explicit version instead of bumping",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + print the plan; touch nothing (no edits, publish, or git)",
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty working tree")
    ap.add_argument(
        "--skip-tests", action="store_true", help="skip `cargo test` before publishing"
    )
    ap.add_argument("--remote", default="origin", help="git remote to push to (default: origin)")
    ap.add_argument("--no-push", action="store_true", help="commit + tag locally but don't push")
    args = ap.parse_args()

    if bool(args.level) == bool(args.set_version):
        die("give exactly one of: a bump level (major|minor|patch) or --set X.Y.Z")

    crates = load_workspace()
    old = current_version(crates)
    new = args.set_version if args.set_version else bump(old, args.level)
    if args.set_version and not SEMVER.match(new):
        die(f"--set {new!r} is not a plain MAJOR.MINOR.PATCH")
    if new == old:
        die(f"new version {new} equals the current version")
    order = publish_order(crates)
    tag = f"{TAG_PREFIX}{new}"

    log(f"workspace at {old} -> {new}")
    log("publishable crates (in order): " + ", ".join(c.name for c in order))
    guarded = [c.name for c in crates if not c.publishable]
    if guarded:
        log("guarded (publish = false): " + ", ".join(guarded))
    log(f"release tag: {tag}")

    # The version edits, previewed for everyone and applied only on a real run.
    edits = rewrite_versions(crates, old, new, apply=False)
    log("version edits:")
    for e in edits:
        print(f"    {e}", file=sys.stderr)

    readme_report(crates, old)

    if args.dry_run:
        log("dry-run: validating current packaging (no files/crates/git touched)")
        validate_packaging(order)
        log(
            f"dry-run complete. A real run would: edit the manifests, commit, publish "
            f"[{', '.join(c.name for c in order)}], then tag {tag} and push to {args.remote}."
        )
        return

    # --- real release ---
    preflight(args.allow_dirty)
    rewrite_versions(crates, old, new, apply=True)
    log(f"bumped manifests to {new}")

    # Build (refreshes Cargo.lock with the new versions) and test.
    run(["cargo", "build", "--manifest-path", str(ROOT_MANIFEST)])
    if not args.skip_tests:
        run(["cargo", "test", "--manifest-path", str(ROOT_MANIFEST)])

    validate_packaging(order)

    # Commit the bump before publishing, so the published crates correspond to a
    # committed state (and the tag we create points at exactly what shipped).
    manifests = [str(c.manifest.relative_to(WORKSPACE)) for c in crates]
    git("add", *manifests, "Cargo.lock")
    git("commit", "-m", f"release: fxtranslate {new}")
    log(f"committed release bump for {new}")

    # crates.io first (not atomic, not reversible) ...
    for c in order:
        publish_crate(c.name)

    # ... tag last (atomic), only now that every crate is up.
    git("tag", "-a", tag, "-m", f"fxtranslate {new}")
    log(f"created tag {tag}")

    if args.no_push:
        log(
            f"--no-push: skipping push. Push manually: git push {args.remote} HEAD && "
            f"git push {args.remote} {tag}"
        )
    else:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        git("push", args.remote, branch)
        git("push", args.remote, tag)
        log(f"pushed {branch} and {tag} to {args.remote}")

    log(f"published fxtranslate {new} 🎉")


if __name__ == "__main__":
    main()
