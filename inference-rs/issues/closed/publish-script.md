# Publish Script

**DONE** — `scripts/publish.py`. Bumps the workspace version (`major|minor|patch` or
`--set X.Y.Z`), validates packaging, publishes the publishable crates in dependency
order, and tags `fxtranslate-vX.Y.Z`. `--dry-run` is read-only (validates + prints the
plan, touches nothing). Decisions made: **lockstep versioning** across the workspace
(the CLI exact-pins the engine), **crates.io first / atomic tag last** (a partial publish
never gets tagged; a re-run skips already-uploaded crates), and the publishable set is
**read from the manifests' `publish` flags** (so the guarded CLI/oracle are excluded until
their guards drop). Housekeeping: `PUBLISHING.md` updated to point at the script and record
the lockstep decision (its former open question). Original ask below.

---

Create a publish python script that has a `--dry-run` that handles the publishing of all of the relevant fxtranslate packages. Do whatever housekeeping that needs to happen for it. Consider how we tag and push up things. Ensure that all of our README.md are up to date for publishing. Consider that the tag is atomic while the deploy to crates can fail. Allow for a `major` `minor` `patch` argument for when publishing. Consider and decide how best to handle versioning between crates in the same codebase, but may drift in numbers, or share the same versioning numbers.
