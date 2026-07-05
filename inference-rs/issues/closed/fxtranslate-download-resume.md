# fxtranslate download: `Range`-based resume of a partial attachment

**DONE** (`fxtranslate-cli: resume interrupted model downloads via HTTP Range`).
`download_retrying` now streams the attachment to a persistent partial file and, on a
retryable mid-stream failure, resumes with `Range: bytes=<have>-` instead of restarting;
`get_to` returns whether the server honored the range (206 append) or sent a full body
(200 restart), and the client truncates any stale tail. This also closed the "30 MB
buffered in memory" gap — the body never fully buffers. Verified by offline tests
(resume issues a tail Range and verifies; server-ignored range restarts cleanly; genuine
hash mismatch doesn't retry).

**Scope choices vs. the original design below:**
- **Resume spans one `download_retrying` call's retry loop** (the flaky-network target:
  drops mid-download, retried in-process). A partial left by a *previous process* is wiped
  on entry rather than resumed, because it can't be validated cheaply — cross-process resume
  would need to persist a validator and is a further small step.
- **`If-Range`/`ETag` was omitted.** The Remote Settings CDN attachments are effectively
  immutable (content-addressed `location` + `decompressedHash`), and the existing SHA-256
  verification already guarantees correctness: a mis-spliced body fails the hash, and
  `Cache::ensure` then wipes the partial and re-fetches once cleanly (self-heal). That
  backstop is strictly stronger than an `If-Range` precondition, so it replaced it.

The original scoping follows, for the record.

---

**Open follow-up**, split out of `closed/fxtranslate-download-resilience.md` (which landed
scopes 1 + 2: timeouts, retry/backoff, and streaming TTY progress). This is that issue's
deferred scope 3 — bigger and riskier, so it was intentionally left out of that pass.

## Current behavior (the gap)

After scopes 1 + 2, `download_retrying` (`crates/fxtranslate-cli/src/fetch.rs`) retries a
failed attachment download by **restarting from scratch**: each attempt streams into a
*fresh* buffer via `Fetch::get_to`, so a connection that drops at 28 / 30 MB throws away
the 28 MB and re-downloads the whole body. Correct and simple, but wasteful on a flaky link.

## Design (resume instead of restart)

On a mid-stream failure, keep the bytes already received and re-request only the tail:

- Stream each attempt into a **persistent partial file** (e.g. `.{name}.partial`) rather than
  a throwaway in-memory buffer, tracking `bytes_received`.
- On retry, send `Range: bytes=<bytes_received>-`. Handle the server's answer:
  - **206 Partial Content** with a matching `Content-Range` → append to the partial file.
  - **200 OK** (server ignored the range) → truncate and restart from scratch (today's behavior).
- Guard against the attachment changing between attempts with an **`If-Range`** using the
  first response's `ETag` (or `Last-Modified`). Note the Remote Settings CDN attachments are
  effectively content-addressed (immutable `location` paths carrying a `decompressedHash`),
  so a mid-download change is unlikely — but `If-Range` makes "server changed it" fail safe
  (falls back to a full 200) rather than silently splicing two different bodies.
- The existing decompressed-SHA-256 verification in `Cache::ensure` remains the backstop: a
  mis-spliced or truncated body fails the hash and re-fetches cleanly, so resume can never
  promote corrupt bytes to a trusted cache file.

## Why it was deferred

- Needs partial-file bookkeeping (a `.partial` that now survives *across* process runs, not
  just within one `ensure`) and cleanup policy for stale partials.
- Needs the `If-Range` / `Content-Range` / `206` vs `200` handshake, which is awkward to test
  offline — `MockFetch` must grow range-awareness (serve a byte suballocation, honor/ignore
  `Range`, echo `Content-Range`, and simulate an `ETag` mismatch → 200).
- Marginal benefit vs. restart for ~30 MB base models on most links; the win grows with larger
  models / worse networks.

## Tests (offline, cheat-proof)

- `MockFetch` gains range support: a `get_to` that honors `Range` returns only the tail and a
  `Content-Range`; one that ignores it returns the full body with 200.
- Resume path: fail after N bytes → retry issues `Range: bytes=N-` → final file is byte-identical
  to a non-resumed download and passes the hash.
- Server-ignored-range path: retry gets a 200 → partial is truncated, full body re-downloaded,
  still verifies.
- `If-Range` mismatch (ETag changed) → falls back to a clean full download.
