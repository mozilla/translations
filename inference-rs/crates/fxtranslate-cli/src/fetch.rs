//! A small [`Fetch`] abstraction (URL → bytes) so the Remote Settings client and
//! the cache can be driven by the real network client in the binary or an
//! in-memory fake in tests. The production impl is [`NetworkFetch`]; the test fake
//! (`MockFetch`) lives with the integration tests (`tests/common`), so it never ships.
//!
//! Two shapes: [`Fetch::get`] pulls a small body (the Remote Settings records JSON)
//! fully into memory in one call, and [`Fetch::get_to`] *streams* a large attachment
//! into a sink while reporting progress — the split lets the model download render a
//! progress line and be retried without buffering the whole body twice. Resilience
//! (connect/read timeouts, retry with backoff) is layered on top of `get_to` by
//! [`download_retrying`].

use std::io::{Read, Write};
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

/// Fetch bytes at a URL.
pub trait Fetch {
    /// Pull the whole body at `url` into memory with a single blocking GET. For
    /// small responses (the Remote Settings records JSON); no progress, no retry.
    fn get(&self, url: &str) -> Result<Vec<u8>, String>;

    /// Stream the body at `url` into `sink` in chunks, invoking
    /// `on_progress(bytes_so_far, total)` after each — `total` is the
    /// `Content-Length` if the server sent one, else `None`. A *single* attempt:
    /// retry/backoff is [`download_retrying`]'s job, so a mid-stream failure here
    /// just returns a [`FetchError`] (the caller discards its buffer and retries
    /// from scratch — no resume yet).
    fn get_to(
        &self,
        url: &str,
        sink: &mut dyn Write,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), FetchError>;
}

/// Download `url` with retry/backoff, returning the full body. Each attempt writes
/// into a *fresh* buffer, so a retry restarts cleanly (the partial bytes of a
/// failed attempt are dropped) and the returned `Vec` is always a complete body.
/// Non-retryable failures (e.g. a 404) return immediately; retryable ones back off
/// per `policy` and log each retry to stderr.
pub fn download_retrying(
    fetch: &dyn Fetch,
    url: &str,
    policy: &RetryPolicy,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<Vec<u8>, String> {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let mut buf = Vec::new();
        match fetch.get_to(url, &mut buf, on_progress) {
            Ok(()) => return Ok(buf),
            Err(e) if e.retryable && attempt < policy.max_attempts => {
                let backoff = policy.delay(attempt);
                eprintln!(
                    "[fetch] {url}: {} — retrying (attempt {}/{}) in {:.1}s",
                    e.message,
                    attempt + 1,
                    policy.max_attempts,
                    backoff.as_secs_f64()
                );
                std::thread::sleep(backoff);
            }
            Err(e) => return Err(e.message),
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
        sink: &mut dyn Write,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), FetchError> {
        let resp = self.agent.get(url).call().map_err(|e| FetchError {
            message: format!("GET {url}: {e}"),
            retryable: ureq_retryable(&e),
        })?;
        // Compressed size, when advertised — drives the progress denominator.
        let total = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok());
        let mut reader = resp.into_reader();
        let mut buf = [0u8; 64 * 1024];
        let mut done = 0u64;
        on_progress(done, total);
        loop {
            // A stalled connection surfaces here as a read timeout/reset; treat any
            // mid-stream read error as transient and let the retry loop restart.
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
        Ok(())
    }
}
