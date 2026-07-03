//! Safe Rust wrapper over the vendored gemmology i8mm kernel (the shifted int8
//! affine `int8shiftAlphaAll`), compiled from `src/gemmology_shim.cpp` under the
//! `gemmology` feature.
//!
//! [`PreparedB`] holds a weight transformed once into gemmology's register-
//! blocked layout; [`PreparedB::matmul`] then runs the SIMD GEMM. It computes
//! the same result as [`crate::ops::intgemm_affine`] — the scalar kernel that is
//! validated against the marian oracle — so `tests/gemm_parity.rs` pins this
//! path to that scalar reference (and thus, transitively, to the oracle).
//!
//! `PreparedB::new` returns `None` when the inner dimension `k` is not a
//! multiple of 16 (gemmology's NEON int8 register width); callers fall back to
//! the scalar kernel in that case.

use std::os::raw::c_void;

extern "C" {
    fn gemmology_prepare_b(b_transposed: *const i8, n: usize, k: usize) -> *mut c_void;
    fn gemmology_free_b(handle: *mut c_void);
    fn gemmology_multiply(
        handle: *const c_void,
        a: *const u8,
        m: usize,
        unquant: f32,
        bias: *const f32,
        out: *mut f32,
    );
    fn gemmology_prepared_bytes() -> usize;
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
/// same orientation [`crate::ops::intgemm_affine`] consumes. Owns aligned native
/// memory freed on drop. Holds a raw pointer, so it is neither `Send` nor `Sync`
/// (used single-threaded behind the engine's `&self`).
pub struct PreparedB {
    handle: *mut c_void,
    n: usize,
    k: usize,
}

impl PreparedB {
    /// Prepare a logical transposed int8 weight `[n, k]`. Returns `None` if `k`
    /// is not a multiple of 16 (caller should use the scalar kernel).
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
        assert_eq!(a.len(), m * self.k, "A length must be m * k");
        assert_eq!(bias.len(), self.n, "bias length must be n");
        let mut out = vec![0.0f32; m * self.n];
        // SAFETY: all buffers match the dimensions the shim expects; `out` is
        // sized m*n and written fully by the shim.
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
        out
    }
}

impl Drop for PreparedB {
    fn drop(&mut self) {
        // SAFETY: `handle` came from gemmology_prepare_b and is freed exactly once.
        unsafe { gemmology_free_b(self.handle) };
    }
}
