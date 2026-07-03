//! A tiny capacity-keyed scratch pool for `f32` activation buffers.
//!
//! The eager forward produces a fresh `Vec<f32>` per op (affine outputs, FFN
//! hidden, …). This pool hands out reusable buffers instead: [`Pool::take`] pops
//! a buffer of the requested capacity class (or allocates on a cold miss) and
//! returns a [`Buf`] lease whose `Drop` returns it to the pool. Because the
//! activation shapes are stable (`rows·dim`, `rows·ffn` for a handful of `rows`),
//! after the first block the pool is warm and steady-state translation allocates
//! ~nothing here — this is marian's `Gap` free-list, minus coalescing, keyed by
//! [`FxHashMap`] because the key is a small integer hit on every `take`.
//!
//! Chosen over a bump allocator: reuse happens at each buffer's exact RAII
//! lifetime end, so no reset barriers and no escape analysis are needed — see
//! `issues/21-activation-scratch-pool.md`. Note (per the jemalloc experiment in
//! `issues/19-settled-rss-allocator.md`) that pooling reduces *churn*, not settled
//! RSS: a page-returning allocator already reaches the live-memory floor. This is
//! a cleanliness/allocator-CPU change, not a footprint lever.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use rustc_hash::FxHashMap;

/// Free lists of reusable `Vec<f32>` buffers, keyed by capacity.
#[derive(Default)]
pub struct Pool {
    free: RefCell<FxHashMap<usize, Vec<Vec<f32>>>>,
}

impl Pool {
    /// Drop all retained buffers. Called between blocks so the pool's retained
    /// set is bounded by one block's working set, not the whole run's — activation
    /// sizes vary widely (encoder `batch·seq`, decoder `m` shrinks as sentences
    /// retire), so an unbounded free list hoards buffers it rarely reuses.
    pub fn clear(&self) {
        self.free.borrow_mut().clear();
    }

    /// Borrow a buffer of length `len` (contents zeroed). Reuses a pooled buffer
    /// of the same capacity class when one is free, else allocates.
    pub fn take(&self, len: usize) -> Buf<'_> {
        let mut v = self
            .free
            .borrow_mut()
            .get_mut(&len)
            .and_then(Vec::pop)
            .unwrap_or_default();
        v.clear();
        v.resize(len, 0.0);
        Buf {
            v: Some(v),
            pool: self,
        }
    }
}

/// A leased buffer. Derefs to `[f32]`; returns itself to the [`Pool`] on drop.
pub struct Buf<'p> {
    v: Option<Vec<f32>>,
    pool: &'p Pool,
}

impl Buf<'_> {
    /// The backing `Vec`, for ops that resize/write it (e.g. the GEMM shim). It is
    /// already sized to the requested `len`.
    pub fn vec_mut(&mut self) -> &mut Vec<f32> {
        self.v.as_mut().expect("buf live")
    }

    /// Detach from the pool, taking ownership (for values that outlive the pool,
    /// e.g. the encoder context stored in a `BatchedContext`). Not returned on drop.
    pub fn into_owned(mut self) -> Vec<f32> {
        self.v.take().expect("buf live")
    }
}

impl Deref for Buf<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.v.as_ref().expect("buf live")
    }
}

impl DerefMut for Buf<'_> {
    fn deref_mut(&mut self) -> &mut [f32] {
        self.v.as_mut().expect("buf live")
    }
}

impl Drop for Buf<'_> {
    fn drop(&mut self) {
        if let Some(v) = self.v.take() {
            let cap = v.capacity();
            self.pool.free.borrow_mut().entry(cap).or_default().push(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_buffers_of_the_same_size() {
        let pool = Pool::default();
        let ptr = {
            let mut a = pool.take(16);
            a[0] = 1.0;
            a.as_ptr()
        }; // a dropped -> returned
        let b = pool.take(16);
        assert_eq!(b.len(), 16);
        assert_eq!(b[0], 0.0, "reused buffer is re-zeroed");
        assert_eq!(b.as_ptr(), ptr, "same allocation reused");
    }

    #[test]
    fn into_owned_detaches() {
        let pool = Pool::default();
        let owned = {
            let mut a = pool.take(4);
            a[1] = 2.0;
            a.into_owned()
        };
        assert_eq!(owned, vec![0.0, 2.0, 0.0, 0.0]);
        // Pool did not get it back, so a fresh take allocates anew.
        assert!(pool.take(4).as_ptr() != owned.as_ptr());
    }
}
