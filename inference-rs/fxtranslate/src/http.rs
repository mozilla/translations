//! A tiny HTTP GET abstraction so the Remote Settings client and the cache can be
//! driven by an in-memory fake in tests (offline, deterministic) and by `ureq`
//! (blocking, rustls — no system OpenSSL) in the binary.

use std::io::Read;

/// One blocking HTTP GET returning the full response body.
pub trait Http {
    fn get(&self, url: &str) -> Result<Vec<u8>, String>;
}

/// The real client, used by the binary. `ureq` brings rustls, so HTTPS works with
/// no system TLS dependency.
pub struct UreqHttp;

impl Http for UreqHttp {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let resp = ureq::get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading {url}: {e}"))?;
        Ok(buf)
    }
}

/// An in-memory `Http` for tests: serves checked-in bytes by exact URL and counts
/// hits so a test can assert a cache hit skipped the network.
#[derive(Default)]
pub struct MockHttp {
    routes: std::collections::HashMap<String, Vec<u8>>,
    pub hits: std::cell::RefCell<Vec<String>>,
}

impl MockHttp {
    pub fn new() -> MockHttp {
        MockHttp::default()
    }

    pub fn route(mut self, url: &str, body: Vec<u8>) -> MockHttp {
        self.routes.insert(url.to_string(), body);
        self
    }

    /// Number of GETs served so far (for cache-hit assertions).
    pub fn hit_count(&self) -> usize {
        self.hits.borrow().len()
    }
}

impl Http for MockHttp {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        self.hits.borrow_mut().push(url.to_string());
        self.routes
            .get(url)
            .cloned()
            .ok_or_else(|| format!("MockHttp: no route for {url}"))
    }
}
