//! A one-method [`Fetch`] abstraction (URL → bytes) so the Remote Settings client
//! and the cache can be driven by the real network client in the binary or an
//! in-memory fake in tests. The production impl is [`NetworkFetch`]; the test fake
//! (`MockFetch`) lives with the integration tests (`tests/common`), so it never ships.

use std::io::Read;

/// Fetch the bytes at a URL with a single blocking GET.
pub trait Fetch {
    fn get(&self, url: &str) -> Result<Vec<u8>, String>;
}

/// The production [`Fetch`]: a live network GET over HTTP(S), via `ureq`.
pub struct NetworkFetch;

impl Fetch for NetworkFetch {
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
