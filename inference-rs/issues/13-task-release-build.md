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
