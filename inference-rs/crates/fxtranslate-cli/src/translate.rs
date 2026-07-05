//! Translation as a swappable dependency: [`Translator::load`] turns a `src`→`trg`
//! pair into a [`Session`] that translates lines. [`EngineTranslator`] is the
//! real one; tests substitute a fake so the translate path runs with no network
//! and no model.

use crate::cache::{ensure_model, Cache};
use crate::fetch::Fetch;
use crate::remote::fetch_records;
use fxtranslate::engine::Engine;

/// Resolves a `src`→`trg` model into a ready [`Session`]. The two-phase shape
/// (`load` once, then `translate` many lines) matches pipe/REPL usage: the model
/// is downloaded and the engine built a single time.
pub trait Translator {
    /// Prepare to translate `src`→`trg`, optionally overriding the model cache
    /// directory. May hit the network / disk in production.
    fn load(
        &self,
        src: &str,
        trg: &str,
        cache_dir: Option<&str>,
    ) -> Result<Box<dyn Session>, String>;
}

/// A loaded model, ready to translate lines.
pub trait Session {
    fn translate(&self, text: &str) -> String;
}

/// Production translator: Remote Settings discovery + verified cache + the
/// neural engine, all over an injected [`Fetch`].
pub struct EngineTranslator<'a> {
    pub fetch: &'a dyn Fetch,
    /// Render a download progress line to stderr (set by `main` when stderr is a
    /// TTY); threaded onto the [`Cache`] so a first-time model pull shows progress.
    show_progress: bool,
}

impl<'a> EngineTranslator<'a> {
    pub fn new(fetch: &'a dyn Fetch, show_progress: bool) -> EngineTranslator<'a> {
        EngineTranslator {
            fetch,
            show_progress,
        }
    }
}

impl Translator for EngineTranslator<'_> {
    fn load(
        &self,
        src: &str,
        trg: &str,
        cache_dir: Option<&str>,
    ) -> Result<Box<dyn Session>, String> {
        let cache = match cache_dir {
            Some(d) => Cache::with_root(d),
            None => Cache::locate(),
        }
        .with_progress(self.show_progress);
        let records = fetch_records(self.fetch)?;
        let files = ensure_model(self.fetch, &cache, records.as_slice(), src, trg)?;
        let engine = Engine::load(&files.model, &files.src_vocab, &files.trg_vocab)?;
        Ok(Box::new(EngineSession(engine)))
    }
}

/// Newtype so the library (which owns [`Session`]) can implement it for the
/// foreign [`Engine`] without tripping the orphan rule.
struct EngineSession(Engine);

impl Session for EngineSession {
    fn translate(&self, text: &str) -> String {
        self.0.translate(text)
    }
}
