//! Test-only [`Fetch`] fake for the offline packaging tests (never shipped —
//! it lives under `tests/`). [`MockFetch`] serves checked-in bytes by exact URL,
//! counts hits (so a test can assert a cache hit skipped the network), and can be
//! scripted to fail/resume/ignore-`Range` to exercise the download retry loop.
#![allow(dead_code)] // not every field/method is exercised by every test

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::SeekFrom;

use fxtranslate::fetch::{DownloadOutcome, Fetch, FetchError, Sink};

/// A scripted `get_to` failure: fail once `after_bytes` of the (tail) body have been
/// written this attempt (`0` = fail before writing anything), with the given
/// `retryable` flag. A partial write leaves bytes on disk for the next attempt to
/// resume from.
#[derive(Clone, Copy)]
struct Fail {
    after_bytes: usize,
    retryable: bool,
}

#[derive(Default)]
pub struct MockFetch {
    routes: HashMap<String, Vec<u8>>,
    hits: RefCell<Vec<String>>,
    /// The `range_from` each `get_to` attempt was called with, in order — so a test
    /// can assert the second attempt actually resumed (`> 0`).
    ranges: RefCell<Vec<u64>>,
    /// Scripted `get_to` failures, consumed in order across attempts. Once drained,
    /// `get_to` serves the route normally.
    script: RefCell<VecDeque<Fail>>,
    /// Bytes-per-chunk `get_to` feeds the sink (and fires `on_progress`); `0` means
    /// the whole body in one chunk. Small values make progress fire repeatedly.
    chunk_size: usize,
    /// Simulate a server that ignores `Range`: always answer with the full body from
    /// offset 0 (HTTP 200), never a 206 tail.
    ignore_range: bool,
}

impl MockFetch {
    pub fn new() -> MockFetch {
        MockFetch::default()
    }

    pub fn route(mut self, url: &str, body: Vec<u8>) -> MockFetch {
        self.routes.insert(url.to_string(), body);
        self
    }

    /// Script the next `n` `get_to` attempts to fail (before writing anything) with
    /// the given `retryable` flag — to exercise the retry/backoff loop.
    pub fn fail_times(self, n: usize, retryable: bool) -> MockFetch {
        let fail = Fail {
            after_bytes: 0,
            retryable,
        };
        self.script
            .borrow_mut()
            .extend(std::iter::repeat_n(fail, n));
        self
    }

    /// Script the next `get_to` attempt to fail *after* writing `after_bytes` of the
    /// body — so the partial bytes land on disk and a later attempt resumes from them.
    pub fn fail_after(self, after_bytes: usize, retryable: bool) -> MockFetch {
        self.script.borrow_mut().push_back(Fail {
            after_bytes,
            retryable,
        });
        self
    }

    /// Feed `get_to` bodies in `size`-byte chunks (so `on_progress` fires more
    /// than once).
    pub fn chunk_size(mut self, size: usize) -> MockFetch {
        self.chunk_size = size;
        self
    }

    /// Make the mock ignore `Range` requests (always a full HTTP-200 body), to
    /// exercise the client's "server ignored the range → restart" path.
    pub fn ignore_range(mut self) -> MockFetch {
        self.ignore_range = true;
        self
    }

    /// Number of fetches served so far (`get` + each `get_to` attempt), for
    /// cache-hit and retry-count assertions.
    pub fn hit_count(&self) -> usize {
        self.hits.borrow().len()
    }

    /// The `range_from` of each `get_to` attempt, in order.
    pub fn get_to_ranges(&self) -> Vec<u64> {
        self.ranges.borrow().clone()
    }
}

impl Fetch for MockFetch {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        self.hits.borrow_mut().push(url.to_string());
        self.routes
            .get(url)
            .cloned()
            .ok_or_else(|| format!("MockFetch: no route for {url}"))
    }

    fn get_to(
        &self,
        url: &str,
        range_from: u64,
        sink: &mut dyn Sink,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<DownloadOutcome, FetchError> {
        self.hits.borrow_mut().push(url.to_string());
        self.ranges.borrow_mut().push(range_from);
        let body = self.routes.get(url).cloned().ok_or_else(|| FetchError {
            message: format!("MockFetch: no route for {url}"),
            retryable: false,
        })?;

        // Honor the Range (serve the tail from `range_from`, HTTP 206) unless told to
        // ignore it or it's out of bounds — then serve the whole body from 0 (200).
        let resumed = range_from > 0 && !self.ignore_range && (range_from as usize) <= body.len();
        let base = if resumed { range_from as usize } else { 0 };
        sink.seek(SeekFrom::Start(base as u64))
            .map_err(|e| FetchError {
                message: e.to_string(),
                retryable: false,
            })?;

        let total = Some(body.len() as u64);
        let tail = &body[base..];
        let step = if self.chunk_size == 0 {
            tail.len().max(1)
        } else {
            self.chunk_size
        };
        let fail = self.script.borrow_mut().pop_front();
        let mut written = 0usize;
        let mut done = base as u64;
        on_progress(done, total);
        for chunk in tail.chunks(step) {
            // A scripted failure trips once we've written its byte threshold (0 =
            // before the first chunk), leaving `written` bytes on disk to resume from.
            if let Some(f) = fail {
                if written >= f.after_bytes {
                    return Err(FetchError {
                        message: format!("scripted failure for {url} after {written} B"),
                        retryable: f.retryable,
                    });
                }
            }
            sink.write_all(chunk).map_err(|e| FetchError {
                message: e.to_string(),
                retryable: false,
            })?;
            written += chunk.len();
            done += chunk.len() as u64;
            on_progress(done, total);
        }
        Ok(DownloadOutcome { resumed, total })
    }
}
