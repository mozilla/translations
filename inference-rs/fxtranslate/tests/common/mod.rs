//! Shared test-only helpers. Lives with the integration tests (never in the
//! shipped library): an in-memory [`Fetch`] that serves checked-in bytes by exact
//! URL and counts hits, so a test can assert a cache hit skipped the network.
#![allow(dead_code)] // not every test binary uses every helper

use std::cell::RefCell;
use std::collections::HashMap;

use fxtranslate::fetch::Fetch;

#[derive(Default)]
pub struct MockFetch {
    routes: HashMap<String, Vec<u8>>,
    hits: RefCell<Vec<String>>,
}

impl MockFetch {
    pub fn new() -> MockFetch {
        MockFetch::default()
    }

    pub fn route(mut self, url: &str, body: Vec<u8>) -> MockFetch {
        self.routes.insert(url.to_string(), body);
        self
    }

    /// Number of GETs served so far (for cache-hit assertions).
    pub fn hit_count(&self) -> usize {
        self.hits.borrow().len()
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
}
