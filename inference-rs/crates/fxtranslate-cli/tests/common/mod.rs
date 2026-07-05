//! Shared test-only helpers, living with the integration tests (never in the
//! shipped library):
//!
//! - [`MockFetch`] — an in-memory [`Fetch`] that serves checked-in bytes by exact
//!   URL and counts hits, so a test can assert a cache hit skipped the network.
//! - [`MockTranslator`] — an engine-free [`Translator`] whose session upper-cases
//!   each line and tags it with the pair (`[en→es] HELLO`), so a translate
//!   transcript proves pair routing and per-line translation with no model.
//! - [`Recorder`] / [`EchoStdin`] / [`run_transcript`] — drive [`cli::run`]
//!   end-to-end and capture stdout+stderr as one interleaved transcript, exactly
//!   what a user would see.
#![allow(dead_code)] // not every test binary uses every helper

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, Cursor, Read, SeekFrom, Write};
use std::rc::Rc;

use fxtranslate::fetch::{DownloadOutcome, Fetch, FetchError, Sink};
use fxtranslate_cli::cli::{run, Deps, Io};
use fxtranslate_cli::translate::{Session, Translator};

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

/// Engine-free translator: `load` succeeds unless the pair was marked
/// [`unsupported`](MockTranslator::unsupported), returning a session that
/// upper-cases each line and prefixes the pair.
#[derive(Default)]
pub struct MockTranslator {
    unsupported: Vec<(String, String)>,
}

impl MockTranslator {
    pub fn new() -> MockTranslator {
        MockTranslator::default()
    }

    /// Make `load(src, trg, …)` fail, to exercise the error transcript.
    pub fn unsupported(mut self, src: &str, trg: &str) -> MockTranslator {
        self.unsupported.push((src.to_string(), trg.to_string()));
        self
    }
}

impl Translator for MockTranslator {
    fn load(
        &self,
        src: &str,
        trg: &str,
        _cache_dir: Option<&str>,
    ) -> Result<Box<dyn Session>, String> {
        if self.unsupported.iter().any(|(s, t)| s == src && t == trg) {
            return Err(format!("no model for {src}-{trg} in Remote Settings"));
        }
        Ok(Box::new(MockSession {
            src: src.to_string(),
            trg: trg.to_string(),
        }))
    }
}

struct MockSession {
    src: String,
    trg: String,
}

impl Session for MockSession {
    fn translate(&self, text: &str) -> String {
        format!("[{}→{}] {}", self.src, self.trg, text.to_uppercase())
    }
}

/// A cloneable shared buffer. Handing one clone to `io.stdout` and another to
/// `io.stderr` lands every write in a single buffer in true order — a genuine
/// interleaved transcript of what the terminal would show.
#[derive(Clone, Default)]
pub struct Recorder {
    buf: Rc<RefCell<Vec<u8>>>,
}

impl Recorder {
    pub fn new() -> Recorder {
        Recorder::default()
    }

    pub fn text(&self) -> String {
        String::from_utf8(self.buf.borrow().clone()).expect("transcript is UTF-8")
    }
}

impl Write for Recorder {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.borrow_mut().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A stdin fake that mirrors each consumed (typed) line into a [`Recorder`] as a
/// terminal would echo it — so an interactive REPL transcript shows the input
/// line right after its prompt.
struct EchoStdin {
    data: Vec<u8>,
    pos: usize,
    echo: Recorder,
}

impl EchoStdin {
    fn new(data: Vec<u8>, echo: Recorder) -> EchoStdin {
        EchoStdin { data, pos: 0, echo }
    }
}

impl Read for EchoStdin {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = Read::read(&mut &self.data[self.pos..], out)?;
        self.pos += n;
        Ok(n)
    }
}

impl BufRead for EchoStdin {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(&self.data[self.pos..])
    }
    fn consume(&mut self, amt: usize) {
        // `read_line` consumes through the newline: echo exactly those bytes.
        self.echo
            .write_all(&self.data[self.pos..self.pos + amt])
            .ok();
        self.pos += amt;
    }
}

/// How to present the injected environment to [`run_transcript`]. `Default` is
/// the common case: empty stdin, nothing a TTY, no color, no echo.
#[derive(Default)]
pub struct Streams {
    /// Piped / typed stdin.
    pub stdin: String,
    /// stdin is a TTY → `translate` uses the interactive REPL.
    pub stdin_tty: bool,
    /// stdout is a TTY → `list` may color.
    pub stdout_tty: bool,
    /// `NO_COLOR` is set.
    pub no_color: bool,
    /// Mirror typed stdin lines into the transcript (for readable REPL sessions).
    pub echo: bool,
}

/// Drive `cli::run(args)` against `deps` and `s`, returning the combined
/// stdout+stderr transcript (interleaved in write order).
pub fn run_transcript(args: &[&str], deps: &Deps, s: Streams) -> String {
    let rec = Recorder::new();
    let mut out = rec.clone();
    let mut err = rec.clone();
    let argv: Vec<String> = args.iter().map(|a| a.to_string()).collect();

    if s.echo {
        let mut stdin = EchoStdin::new(s.stdin.into_bytes(), rec.clone());
        let mut io = Io {
            stdin: &mut stdin,
            stdout: &mut out,
            stderr: &mut err,
            stdin_is_tty: s.stdin_tty,
            stdout_is_tty: s.stdout_tty,
            no_color: s.no_color,
        };
        let _ = run(&argv, deps, &mut io);
    } else {
        let mut stdin = Cursor::new(s.stdin.into_bytes());
        let mut io = Io {
            stdin: &mut stdin,
            stdout: &mut out,
            stderr: &mut err,
            stdin_is_tty: s.stdin_tty,
            stdout_is_tty: s.stdout_tty,
            no_color: s.no_color,
        };
        let _ = run(&argv, deps, &mut io);
    }

    rec.text()
}

/// Assert a transcript matches `expected` line-for-line. On mismatch, print the
/// actual output as a paste-ready block so a reviewer can audit and drop it in.
pub fn assert_transcript(label: &str, got: &str, expected: &[&str]) {
    let got_lines: Vec<&str> = got.lines().collect();
    if got_lines != expected {
        let paste = got_lines
            .iter()
            .map(|l| format!("            {l:?},"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{label} transcript changed. If intended, replace the expected block with:\n&[\n{paste}\n        ]\n"
        );
    }
}
