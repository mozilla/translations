//! Safe Rust wrapper over the vendored gemmology SIMD kernel for the shifted
//! int8 affine (`int8shiftAlphaAll`), compiled from `src/gemmology_shim.cpp`.
//!
//! [`PreparedB`] holds a weight transformed once into gemmology's register-
//! blocked layout; [`PreparedB::matmul`] then runs the SIMD GEMM. It computes the
//! same result as [`crate::ops::intgemm_affine`] — the scalar kernel validated
//! against the marian oracle — so `tests/gemm_parity.rs` pins this path to that
//! scalar reference (and thus, transitively, to the oracle).
//!
//! This module is present whenever the `gemmology` feature is on, but the SIMD
//! kernel is compiled only when `build.rs` could build the shim for the target
//! (which sets `--cfg gemmology_simd`). Otherwise the definitions below are a
//! scalar-fallback stub whose [`PreparedB::new`] always returns `None`, so
//! callers use [`crate::ops::intgemm_affine`]. See build.rs and issues/24.

#[cfg(gemmology_simd)]
mod imp {
    use std::os::raw::c_void;

    extern "C" {
        fn gemmology_prepare_b(b_transposed: *const i8, n: usize, k: usize) -> *mut c_void;
        fn gemmology_free_b(handle: *mut c_void);
        fn gemmology_multiply(
            handle: *mut c_void,
            a: *const u8,
            m: usize,
            unquant: f32,
            bias: *const f32,
            out: *mut f32,
        );
        fn gemmology_prepared_bytes() -> usize;
        fn gemmology_read_row(handle: *const c_void, id: usize, out: *mut i8);
    }

    /// Total retained bytes of prepared-B weight buffers — the persistent C++
    /// allocations gemmology holds, which dhat (Rust-heap only) cannot see. For
    /// memory accounting.
    pub fn prepared_bytes() -> usize {
        // SAFETY: reads an atomic counter in the shim; always valid.
        unsafe { gemmology_prepared_bytes() }
    }

    /// A weight matrix prepared once into gemmology's register-blocked int8 layout.
    ///
    /// Logically a transposed weight `[n, k]` (row-major, `w[col * k + kk]`) — the
    /// same orientation [`crate::ops::intgemm_affine`] consumes. Owns aligned
    /// native memory freed on drop. Holds a raw pointer, so it is neither `Send`
    /// nor `Sync` (used single-threaded behind the engine's `&self`).
    pub struct PreparedB {
        handle: *mut c_void,
        n: usize,
        k: usize,
    }

    impl PreparedB {
        /// Prepare a logical transposed int8 weight `[n, k]`. Returns `None` if
        /// `k` is not a multiple of 16 (caller should use the scalar kernel).
        ///
        /// # Panics
        /// If `b_transposed.len() != n * k`.
        pub fn new(b_transposed: &[i8], n: usize, k: usize) -> Option<PreparedB> {
            assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
            // SAFETY: pointer/len are consistent with (n, k); the shim only reads
            // n*k bytes and copies them into its own buffer.
            let handle = unsafe { gemmology_prepare_b(b_transposed.as_ptr(), n, k) };
            if handle.is_null() {
                None
            } else {
                Some(PreparedB { handle, n, k })
            }
        }

        /// `out[m, n] = unquant * (A[m,k] · W[k,n]) + bias[n]`, where `a` is the
        /// shifted uint8 activation `[m, k]` (row-major) and `bias` is the prepared
        /// bias of length `n`. Returns `[m, n]` row-major.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul(&self, a: &[u8], m: usize, unquant: f32, bias: &[f32]) -> Vec<f32> {
            let mut out = Vec::new();
            self.matmul_into(a, m, unquant, bias, &mut out);
            out
        }

        /// [`matmul`] into a caller-owned buffer (resized to `[m, n]`), reused
        /// across calls to avoid a fresh allocation per GEMM. The shim's own
        /// A/bias/output scratch is likewise persistent, so a steady-state call
        /// allocates nothing.
        ///
        /// # Panics
        /// If `a.len() != m * k` or `bias.len() != n`.
        pub fn matmul_into(
            &self,
            a: &[u8],
            m: usize,
            unquant: f32,
            bias: &[f32],
            out: &mut Vec<f32>,
        ) {
            assert_eq!(a.len(), m * self.k, "A length must be m * k");
            assert_eq!(bias.len(), self.n, "bias length must be n");
            out.clear();
            out.resize(m * self.n, 0.0);
            // SAFETY: buffers match the shim's expected dimensions; `out` is sized
            // m*n and fully written. The shim mutates only its own C++ scratch.
            unsafe {
                gemmology_multiply(
                    self.handle,
                    a.as_ptr(),
                    m,
                    unquant,
                    bias.as_ptr(),
                    out.as_mut_ptr(),
                );
            }
        }

        /// Read logical row `id` of the prepared weight back into `out` (length
        /// `k`), reversing the register-blocked pack. This lets a caller drop the
        /// raw int8 copy and serve row lookups (e.g. embeddings) out of this one
        /// packed buffer.
        ///
        /// # Panics
        /// If `out.len() != k` or `id >= n`.
        pub fn read_row(&self, id: usize, out: &mut [i8]) {
            assert_eq!(out.len(), self.k, "out length must be k");
            assert!(id < self.n, "row id {id} out of range (n = {})", self.n);
            // SAFETY: id < n and out has length k; the shim writes exactly k bytes
            // into out and only reads its own packed buffer.
            unsafe { gemmology_read_row(self.handle, id, out.as_mut_ptr()) };
        }
    }

    impl Drop for PreparedB {
        fn drop(&mut self) {
            // SAFETY: `handle` came from gemmology_prepare_b and is freed exactly once.
            unsafe { gemmology_free_b(self.handle) };
        }
    }
}

/// Scalar-fallback stub: no SIMD kernel was compiled for this target (non-aarch64,
/// `portable`, or no C++ compiler). [`PreparedB::new`] always returns `None`, so
/// every caller keeps the scalar [`crate::ops::intgemm_affine`] path; the compute
/// methods exist only to satisfy those call sites and are never reached.
#[cfg(not(gemmology_simd))]
mod imp {
    /// Always 0 — no SIMD weights are prepared without a kernel.
    pub fn prepared_bytes() -> usize {
        0
    }

    /// Stub mirror of the SIMD [`PreparedB`]; never constructed.
    pub struct PreparedB {
        _never: (),
    }

    impl PreparedB {
        /// Always `None` on a scalar build — the caller uses the scalar kernel.
        pub fn new(b_transposed: &[i8], n: usize, k: usize) -> Option<PreparedB> {
            debug_assert_eq!(b_transposed.len(), n * k, "B length must be n * k");
            None
        }

        pub fn matmul(&self, _a: &[u8], _m: usize, _unquant: f32, _bias: &[f32]) -> Vec<f32> {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }

        pub fn matmul_into(
            &self,
            _a: &[u8],
            _m: usize,
            _unquant: f32,
            _bias: &[f32],
            _out: &mut Vec<f32>,
        ) {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }

        pub fn read_row(&self, _id: usize, _out: &mut [i8]) {
            unreachable!("scalar-fallback PreparedB is never constructed")
        }
    }
}

pub use imp::{prepared_bytes, PreparedB};
