//! End-to-end model loading: Remote Settings discovery + verified cache + engine
//! build, in one call. The batteries-included convenience an embedder reaches for
//! when it doesn't want to orchestrate [`remote`](crate::remote),
//! [`cache`](crate::cache), and [`Engine::load`] by hand — pass a [`Fetch`] (the
//! built-in [`NetworkFetch`](crate::fetch::NetworkFetch) under `net`, or your own
//! client) and a [`Cache`], get a ready [`Engine`].

use crate::cache::{ensure_model, Cache, ModelFiles};
use crate::engine::Engine;
use crate::fetch::Fetch;
use crate::remote::fetch_records;

/// Resolve, download (or cache-hit), and hash-verify every file needed to
/// translate `src`→`trg`, without building the engine — for callers that want the
/// on-disk paths (e.g. to `Engine::load_mmapped` themselves, or inspect them).
/// Fetches the Remote Settings records via `fetch`, then defers to
/// [`ensure_model`].
pub fn ensure_files(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<ModelFiles, String> {
    let records = fetch_records(fetch)?;
    ensure_model(fetch, cache, &records, src, trg)
}

/// Discover, download+cache (verified), and build a ready [`Engine`] for
/// `src`→`trg`. The one-call path: `fetch_records` → [`ensure_model`] →
/// [`Engine::load`]. `fetch` supplies HTTP (the built-in `NetworkFetch` under
/// `net`, or an embedder's own [`Fetch`]); `cache` is where verified files land
/// (see [`Cache::locate`] for the platform default).
pub fn load_engine(
    fetch: &dyn Fetch,
    cache: &Cache,
    src: &str,
    trg: &str,
) -> Result<Engine, String> {
    let files = ensure_files(fetch, cache, src, trg)?;
    Engine::load(&files.model, &files.src_vocab, &files.trg_vocab)
}
