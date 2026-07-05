//! A small [`Fetch`] abstraction (URL → bytes) so the Remote Settings client and
//! the cache can be driven by the real network client in the binary or an
//! in-memory fake in tests. The production impl is [`NetworkFetch`]; the test fake
//! (`MockFetch`) lives with the integration tests (`tests/common`), so it never ships.
//!
//! Two shapes: [`Fetch::get`] pulls a small body (the Remote Settings records JSON)
//! fully into memory in one call, and [`Fetch::get_to`] *streams* a large attachment
//! into a seekable sink while reporting progress — the split lets the model download
//! render a progress line, be retried, and *resume* without buffering the whole body
//! in memory. Resilience (connect/read timeouts, retry with backoff, HTTP-`Range`
//! resume) is layered on top of `get_to` by [`download_retrying`].

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;

/// A fetch failure carrying whether it is worth retrying. `get_to` returns this
/// (rather than a bare `String`) so [`download_retrying`] can decide to back off
/// and try again vs. fail fast, without either layer having to re-parse an error
/// string or know about `ureq`'s error types.
#[derive(Debug)]
pub struct FetchError {
    pub message: String,
    pub retryable: bool,
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Whether an HTTP *status* is worth retrying: 429 (rate limited) and any 5xx
/// (server/CDN trouble) are transient; other 4xx (notably 404 = record absent
/// from Remote Settings) are not — retrying them just wastes time. Transport
/// errors (DNS/connect/reset/read-timeout) are always retryable and don't reach
/// this classifier. Pure and `pub` so the classification is unit-testable without
/// synthesizing a `ureq::Error`.
pub fn status_is_retryable(code: u16) -> bool {
    code == 429 || code >= 500
}

/// How [`download_retrying`] paces its attempts: at most `max_attempts` tries,
/// sleeping `base_delay * 2^(n-1)` after the n-th failure (so the default is
/// 0.5s → 1s → 2s before the 4th and final try).
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::from_millis(500),
        }
    }
}

impl RetryPolicy {
    /// A policy that retries the same number of times but never sleeps — for tests
    /// that exercise the retry loop without spending wall-clock on backoff.
    pub fn no_delay() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::ZERO,
        }
    }

    /// Backoff before the attempt *after* the `attempt`-th failure (1-based).
    fn delay(&self, attempt: u32) -> Duration {
        self.base_delay * 2u32.saturating_pow(attempt.saturating_sub(1))
    }
}

/// A download destination: writable and seekable, so a resumed transfer can position
/// at the byte where it left off (and a server that ignores our `Range` can rewrite
/// from the start). A blanket impl covers `std::fs::File` (production) and
/// `Cursor<Vec<u8>>` (tests).
pub trait Sink: Write + Seek {}
impl<T: Write + Seek + ?Sized> Sink for T {}

/// What a single [`Fetch::get_to`] attempt did, so [`download_retrying`] knows how to
/// continue: `resumed` is true when the server honored our `Range` (HTTP 206, the sink
/// was appended to) and false when it sent the full body (HTTP 200, the sink was
/// rewritten from offset 0). `total` is the full resource size when known.
#[derive(Clone, Copy, Debug)]
pub struct DownloadOutcome {
    pub resumed: bool,
    pub total: Option<u64>,
}

/// Summary of a completed [`download_retrying`], for the caller's own retry decisions.
#[derive(Clone, Copy, Debug)]
pub struct DownloadStats {
    /// How many `get_to` attempts it took (1 = clean first try; >1 = it resumed).
    pub attempts: u32,
}

/// Fetch bytes at a URL.
pub trait Fetch {
    /// Pull the whole body at `url` into memory with a single blocking GET. For
    /// small responses (the Remote Settings records JSON); no progress, no retry.
    fn get(&self, url: &str) -> Result<Vec<u8>, String>;

    /// Stream the body at `url` into `sink`, invoking `on_progress(bytes_so_far,
    /// total)` after each chunk, where `bytes_so_far` is *absolute* (counts any
    /// resumed prefix). When `range_from > 0`, request `Range: bytes=range_from-`
    /// and, if the server honors it (206), seek `sink` to `range_from` and append;
    /// otherwise (200) seek to 0 and rewrite. A *single* attempt — retry/backoff and
    /// the resume bookkeeping are [`download_retrying`]'s job, so a mid-stream failure
    /// just returns a [`FetchError`] with whatever was written left in place.
    fn get_to(
        &self,
        url: &str,
        range_from: u64,
        sink: &mut dyn Sink,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<DownloadOutcome, FetchError>;
}

/// Download `url` into the file at `dest` with retry/backoff, *resuming* across
/// attempts: a mid-stream failure leaves the bytes received so far on disk, and the
/// next attempt asks for only the tail (`Range: bytes=<have>-`). Memory stays bounded
/// — the body streams straight to the file, never fully buffered. On success `dest`
/// holds the complete body; correctness of the assembled bytes is the caller's to
/// verify (a hash), and a caller that finds them bad should delete `dest` and call
/// again for a clean restart.
///
/// A stale partial from a *previous process* can't be validated here, so `dest` is
/// removed first — resume spans only this call's own retry loop. Retryable failures
/// back off per `policy`; a non-retryable failure (e.g. a 404) or an exhausted budget
/// removes the partial and returns the error.
pub fn download_retrying(
    fetch: &dyn Fetch,
    url: &str,
    dest: &Path,
    policy: &RetryPolicy,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<DownloadStats, String> {
    let _ = std::fs::remove_file(dest);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        // Read+write, create if missing, never truncate — `get_to` seeks explicitly.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // keep the resumable partial; `get_to` seeks explicitly
            .open(dest)
            .map_err(|e| format!("opening {}: {e}", dest.display()))?;
        let have = file.metadata().map_err(|e| e.to_string())?.len();
        match fetch.get_to(url, have, &mut file, on_progress) {
            Ok(_) => {
                // Drop any stale tail past what we just wrote — e.g. a server that
                // ignored our `Range` and re-sent a (possibly shorter) full body.
                let end = file.stream_position().map_err(|e| e.to_string())?;
                file.set_len(end).map_err(|e| e.to_string())?;
                return Ok(DownloadStats { attempts: attempt });
            }
            Err(e) if e.retryable && attempt < policy.max_attempts => {
                drop(file); // flush the partial before the next attempt reopens it
                let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                let backoff = policy.delay(attempt);
                eprintln!(
                    "[fetch] {url}: {} — resuming from {have} B (attempt {}/{}) in {:.1}s",
                    e.message,
                    attempt + 1,
                    policy.max_attempts,
                    backoff.as_secs_f64()
                );
                std::thread::sleep(backoff);
            }
            Err(e) => {
                let _ = std::fs::remove_file(dest);
                return Err(e.message);
            }
        }
    }
}

/// The production [`Fetch`]: live HTTP(S) over a reused `ureq::Agent` configured
/// with a connect timeout and a per-read *inactivity* timeout.
pub struct NetworkFetch {
    agent: ureq::Agent,
}

impl NetworkFetch {
    pub fn new() -> NetworkFetch {
        // `timeout_read` is a per-read *inactivity* timeout, not an overall cap: a
        // legitimate multi-minute download on a slow link keeps making progress and
        // survives, while a server that accepts then stalls mid-body trips it.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(30))
            .build();
        NetworkFetch { agent }
    }
}

impl Default for NetworkFetch {
    fn default() -> NetworkFetch {
        NetworkFetch::new()
    }
}

/// Classify a `ureq` call error: transport-level trouble is always transient;
/// status errors defer to [`status_is_retryable`].
fn ureq_retryable(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Transport(_) => true,
        ureq::Error::Status(code, _) => status_is_retryable(*code),
    }
}

/// Total resource size from a `Content-Range: bytes <start>-<end>/<total>` header.
fn content_range_total(header: &str) -> Option<u64> {
    header.rsplit('/').next()?.trim().parse::<u64>().ok()
}

impl Fetch for NetworkFetch {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let resp = self
            .agent
            .get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading {url}: {e}"))?;
        Ok(buf)
    }

    fn get_to(
        &self,
        url: &str,
        range_from: u64,
        sink: &mut dyn Sink,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<DownloadOutcome, FetchError> {
        let mut req = self.agent.get(url);
        if range_from > 0 {
            req = req.set("Range", &format!("bytes={range_from}-"));
        }
        let resp = req.call().map_err(|e| FetchError {
            message: format!("GET {url}: {e}"),
            retryable: ureq_retryable(&e),
        })?;

        // 206 => the server honored our Range (body is the tail from `range_from`);
        // anything else (200) => it sent the whole body, so we rewrite from 0.
        let resumed = resp.status() == 206;
        let total = if resumed {
            resp.header("Content-Range").and_then(content_range_total)
        } else {
            resp.header("Content-Length")
                .and_then(|s| s.parse::<u64>().ok())
        };
        let base = if resumed { range_from } else { 0 };
        sink.seek(SeekFrom::Start(base)).map_err(|e| FetchError {
            message: format!("seek {url}: {e}"),
            retryable: false,
        })?;

        let mut reader = resp.into_reader();
        let mut buf = [0u8; 64 * 1024];
        let mut done = base;
        on_progress(done, total);
        loop {
            // A stalled connection surfaces here as a read timeout/reset; treat any
            // mid-stream read error as transient and let the retry loop resume.
            let n = reader.read(&mut buf).map_err(|e| FetchError {
                message: format!("reading {url}: {e}"),
                retryable: true,
            })?;
            if n == 0 {
                break;
            }
            sink.write_all(&buf[..n]).map_err(|e| FetchError {
                message: format!("writing {url}: {e}"),
                retryable: false, // a local write failure won't fix itself
            })?;
            done += n as u64;
            on_progress(done, total);
        }
        Ok(DownloadOutcome { resumed, total })
    }
}
