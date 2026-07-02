## Audit Wasm Parity

This package aims to be a replacement for the Wasm project. I want to ensure we have parity with it. Do an audit that we have all of the features the Wasm interfaces have wired in. For the HTML support and alignments work, let's assume we can wire in HTML parsing via Firefox. Actually getting parsed HTML out of Firefox is OUT OF SCOPE, but we could build the interface. The HTML alignments Wasm work could be an oracle we could target to get parity.

Let's skip quality estimation.

## Deliverable shape

A **coverage matrix**: each Wasm/bergamot-translator interface capability → {have / partial /
missing / out-of-scope} in inference-rs, with a pointer to where it's wired (or the gap).
This is an audit artifact for review, not a pass/fail test.

## Cheat-proof where it can be

- **HTML alignments** are the one piece with a real oracle: the Wasm build emits alignments,
  so once [07-wasm-native-build.md](./07-wasm-native-build.md) lets us run it natively, diff
  our alignment output against it on a corpus (same discipline as
  [02-parity-harness.md](./02-parity-harness.md)). Getting parsed HTML *out of Firefox* is
  out of scope — build the interface, oracle the alignment math.
- Interface-presence items (does the API exist / is it wired) are a checklist, inherently a
  review artifact, not oracle-testable.

## Open questions

- What's the authoritative interface surface to audit against — the bergamot-translator
  `BlockingService`/`AsyncService` API, or a specific Gecko-facing surface?
- Confirmed out: quality estimation, HTML parsing from Firefox. Anything else explicitly out
  (e.g. async/batching APIs, pivoting through a second model)?
