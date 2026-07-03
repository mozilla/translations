# Task Release Build

I want a task to do a release build. I want it to be orchtestrated from a python script so we can do some kind of analysis ontop of it. I want to characterize the binary size. Basically let's get some validation built ontop and reporting to see how we are doing. You can add a flag `--skip-validation` here.

Consider any other issues collected that would be good to sequence first here.

## What "validation" means (cheat-proof, or `--skip-validation`)

Binary size alone is gameable (strip everything, ship something broken). Pair the size
number with correctness + config assertions on the *actual release artifact*:

- **Runs correctly**: pipe a golden sentence through the built release binary and match the
  parity harness expectation ([02-parity-harness.md](./02-parity-harness.md)) — not a
  hand-written expected string (that's tautological and rots).
- **Ships lean**: assert the opt-in features from
  [09-cfg-feature-instrumentation.md](./09-cfg-feature-instrumentation.md) (`trace`, `dhat`,
  …) are **off** in the release build — otherwise "release" quietly carries the dev weight.
- **Characterize size**: report total + optionally a `cargo bloat` breakdown, and the
  default-vs-full delta from issue 09.

## Open questions

- Is there a size **budget** to assert against (fail over N MB), or just report and track?
- Validation corpus: reuse the parity harness's committed sentences (keeps one source of
  truth for "correct").
- Sequencing: this wants 09 (features) and 02 (parity goldens) to exist first — agrees with
  their placement earlier in the sequence.

## Done (commit pending) — `scripts/release_build.py` + `task inference-rs:release`

A Python-orchestrated release build with size characterization and cheat-proof validation
(`--skip-validation` to bypass):

- **Size**: reports the `inference-rs` (0.93 MiB) and `fxtranslate` (2.85 MiB) release binaries,
  plus the **default-vs-`--all-features` delta** (0.11 MiB — the instrumentation/dhat/dev weight,
  built into a side target dir so it doesn't clobber the validated artifact). `--bloat` adds a
  `cargo bloat` crate breakdown when `cargo-bloat` is installed.
- **Ships lean** (hard gate): a no-args run of the release binary must not list the
  `trace`/`replay` subcommands (instrumentation-gated) nor print the `[dhat]` banner — proving
  those opt-in features are OFF.
- **Runs correctly** (hard gate, non-tautological, non-rotting): the release binary's output must
  equal the **debug build's** output over the committed corpus — i.e. the optimized artifact is a
  *faithful* build of the engine the test suite validates against the marian oracle. The
  `translator-cli` exact-match rate is **reported as a tracking metric** (not a gate), matching
  02-parity-harness.md / parity.py's stance that greedy output is not 100% identical to marian on
  every sentence (observed 15/20 on en-fr dev — a few are sentence-initial casing differences worth
  a separate look, but that's engine behavior, not the artifact).

Answers the open questions: no hard size budget yet (report + track; the delta is the number to
watch); validation reuses the committed parity corpus; sequencing after 09 (features) and 02
(parity) held.
