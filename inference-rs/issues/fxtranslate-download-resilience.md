# fxtranslate download resilience: timeout, retry, streaming progress

**Open, scoped follow-on.** `fxtranslate`'s model download has no timeout, no retry, and no
progress — a poor fit for pulling ~30 MB base models over a flaky network. This scopes the fix.
Depends on nothing; touches only the `fxtranslate` crate.

## Current behavior (the gap)

The whole network path is one blocking call in `NetworkFetch::get` (`fxtranslate/src/fetch.rs`):

```rust
let resp = ureq::get(url).call()?;              // no Agent, no timeouts
resp.into_reader().read_to_end(&mut buf)?;      // whole body into memory, one opaque read
```

called **once** by `Cache::ensure` (`fxtranslate/src/cache.rs`) with no retry loop. Consequences on
a large attachment:

- **Blocks the main thread.** The CLI is single-threaded and synchronous
  (`main → run → cmd_translate → ensure_model → Cache::ensure → NetworkFetch::get`); `ureq` is a
  blocking client with no runtime/thread. During a download the main thread is parked in a socket
  `read()` syscall. It's an I/O wait (Ctrl-C still exits), but there is no feedback and no way to
  render progress *during* `read_to_end` — that call owns the thread until it returns.
- **No timeout.** `ureq` applies none unless configured, so a server that accepts then stalls
  mid-body parks that syscall effectively forever.
- **No retry.** A transient failure (CDN 5xx/429, connection reset, read timeout, truncated body)
  fails `get` → `ensure` → the whole invocation.
- **No resume, no progress, 30 MB buffered in memory.**

What *is* already safe: `ensure` verifies decompressed SHA-256 against `decompressedHash` and writes
atomically (temp → rename), so a failed/partial download never leaves a corrupt file a later run
would trust — re-running cleanly re-fetches. That's correctness, not resilience; the user still has
to notice and re-run by hand.

## Design

### Timeouts — a reused `Agent` with an *inactivity* read timeout

`NetworkFetch` gains a `ureq::Agent` field (no longer a unit struct; `&NetworkFetch` →
`&NetworkFetch::new()` at the two call sites in `main.rs`):

```rust
ureq::AgentBuilder::new()
    .timeout_connect(Duration::from_secs(10))
    .timeout_read(Duration::from_secs(30)) // per-read INACTIVITY, not total
    .build()
```

Use `timeout_read` (fires only when the connection stalls), **not** an overall `.timeout()` — a
legitimate 30 MB download on a slow link can take minutes and an overall cap would wrongly kill it.

### Retry with backoff — classify what's worth retrying

Wrap the connect+download attempt in a small loop (~4 attempts, sleep 0.5 → 1 → 2 s via
`std::thread::sleep`, log each retry to stderr):

```rust
fn is_retryable(e: &ureq::Error) -> bool {
    matches!(e, ureq::Error::Transport(_))                    // DNS/connect/read-timeout/reset
        || matches!(e, ureq::Error::Status(c, _) if *c == 429 || *c >= 500)
}
```

Do **not** retry other 4xx (a 404 = model absent from Remote Settings; retrying is pointless). On a
mid-stream failure, restart from scratch (truncate the temp file) — no resume yet (see follow-up).

### Streaming + progress — chunked read so the single thread can render

Replace `read_to_end` with a chunk loop, which returns control between reads so the *same* main
thread renders progress (no second thread needed) and memory is bounded:

```rust
let total = resp.header("Content-Length").and_then(|s| s.parse::<u64>().ok()); // compressed size
let mut reader = resp.into_reader();
let mut buf = [0u8; 64 * 1024];
let mut done = 0u64;
loop {
    let n = reader.read(&mut buf)?;
    if n == 0 { break; }
    sink.write_all(&buf[..n])?;      // stream into the cache's temp file
    done += n as u64;
    on_progress(done, total);
}
```

- Small `records` JSON keeps `get() -> Vec<u8>` (no progress). The large attachment download becomes
  a streaming method (e.g. `get_to(url, sink, on_progress)`) used by `Cache::ensure`; `MockFetch`
  implements it trivially (write bytes, call the callback once) so offline tests stay simple.
- **Progress policy in `main.rs`, mechanism in the lib** — same split as `write_list`'s `color: bool`.
  The lib takes `on_progress: &mut dyn FnMut(u64, Option<u64>)`; `main` supplies a renderer that
  draws a `\r`-updated line **only when `std::io::stderr().is_terminal()`**, and a no-op otherwise
  (pipes/CI/tests get no `\r` spam):

  ```text
    model.enes.intgemm.alphas.bin: 12.4 / 31.0 MiB (40%)
  ```
  Fall back to a bytes-only line when `Content-Length` is absent; print a trailing newline on
  completion.

## Scopes (smallest → fullest)

1. **Timeouts + retry** — Agent (connect + inactivity read timeouts) and the retry/backoff loop.
   Fixes hangs and transient failures. Contained to `fetch.rs` + `cache.rs`.
2. **Streaming + TTY progress** — the chunk loop into the temp file, `get_to` + callback, and the
   `main.rs` TTY renderer. Shares the chunk loop with (1), so do them together.
3. **`Range`-based resume** *(follow-up, not this pass)* — resume a partial attachment instead of
   restarting; needs partial-file bookkeeping and an `If-Range`/`Content-Range` handshake.

Recommend landing **1 + 2** together.

## Tests (offline, cheat-proof — extend `tests/`)

- `is_retryable` classification: transport + 429/5xx retry; 404/other 4xx don't.
- Retry loop: a `MockFetch` that fails N times then succeeds → `ensure` succeeds with the expected
  attempt count; a permanent failure → errors after the cap with nothing left in the cache.
- Progress: `get_to` into a buffer invokes `on_progress` with monotonically increasing `done` that
  ends at the total; last value == byte length.
- Keep it network-free: extend `MockFetch` (tests/common) to script per-attempt outcomes and to feed
  bytes in chunks so the progress callback fires more than once.

No new runtime deps (`std::time`, `std::thread::sleep`, `IsTerminal` all in std; `ureq` already in).
