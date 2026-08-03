//! Sparse singular value decomposition.
//!
//! Three solvers over [`sprs`] matrices, all returning the same [`SvdRec`]:
//!
//! | module | method | use when |
//! |---|---|---|
//! | [`irlba`] | thick-restarted Lanczos bidiagonalization | **default.** Accurate, memory bounded by the requested rank |
//! | [`randomized`] | randomized range finder, power iteration or block Krylov | very large inputs where an approximation is acceptable |
//! | [`lanczos`] | LAS2 from SVDLIBC | **deprecated, numerically unreliable** — see the module docs |
//!
//! # Quick start
//!
//! ```
//! use single_svdlib::{sprs::TriMatI, SvdMat};
//!
//! // A 4x3 matrix in triplet form, converted to CSR with u32 indices.
//! let mut tri = TriMatI::<f64, u32>::new((4, 3));
//! tri.add_triplet(0, 0, 1.0);
//! tri.add_triplet(1, 1, 2.0);
//! tri.add_triplet(2, 2, 3.0);
//! tri.add_triplet(3, 0, 4.0);
//! let a: SvdMat<f64> = tri.to_csr::<u64>();
//!
//! // Two largest singular triplets.
//! let svd = single_svdlib::svd(&a, 2)?;
//!
//! assert_eq!(svd.s.len(), 2);
//! assert_eq!(svd.u.dim(), (4, 2));   // left vectors are columns
//! assert_eq!(svd.vt.dim(), (2, 3));  // right vectors are rows
//! assert!(svd.s[0] >= svd.s[1]);
//! # Ok::<(), single_svdlib::SvdLibError>(())
//! ```
//!
//! # Index widths
//!
//! [`SvdMat<T>`] defaults to `u32` column indices with `u64` row pointers, which is
//! 12 bytes per non-zero for `f64` data against the 16 that `usize`-everywhere costs
//! (8 against 16 for `f32`). Name the parameters to widen: `SvdMat<f64, u64, u64>`.
//!
//! # Orientation
//!
//! `A ≈ u · diag(s) · vt`, matching `numpy.linalg.svd`: `u` is `m × d` with left
//! vectors as columns, `s` is descending, `vt` is `d × n` with right vectors as rows.
//! 1.x was inconsistent between solvers on this point.

// Numeric kernels index several arrays in step from one loop variable, and
// offset arithmetic is load-bearing; iterator rewrites obscure which array an
// index belongs to.
#![allow(clippy::needless_range_loop)]

pub mod dense;
pub mod error;
pub mod irlba;
pub mod lanczos;
pub mod matrix;
pub mod randomized;
pub mod types;

#[cfg(test)]
mod testing;

pub use error::{Result, SvdLibError};
pub use matrix::{
    MaskedCsMat, SparseMat, SparseMatDense, SvdMat, SvdMatView, DEFAULT_SCRATCH_BUDGET,
};
pub use types::{Algorithm, Detail, Diagnostics, SvdFloat, SvdRec};

/// Re-exported so callers construct matrices without pinning `sprs` themselves.
pub use sprs;

/// The `rank` largest singular triplets.
///
/// Dispatches to [`irlba`], which is accurate and holds a basis bounded by `rank`.
/// Reach past this for a fixed seed ([`irlba::svd_seed`]), PCA
/// ([`irlba::svd_centered`]), or an approximation on a very large input
/// ([`randomized`]).
pub fn svd<T: SvdFloat, M: SparseMat<T>>(a: &M, rank: usize) -> Result<SvdRec<T>> {
    irlba::svd(a, rank)
}

/// The `rank` largest singular triplets, reproducibly.
pub fn svd_seed<T: SvdFloat, M: SparseMat<T>>(a: &M, rank: usize, seed: u64) -> Result<SvdRec<T>> {
    irlba::svd_seed(a, rank, seed)
}

/// PCA: the `rank` largest singular triplets of the implicitly mean-centered matrix.
///
/// The centering is applied as a rank-1 correction inside each product, so the matrix
/// is never densified.
pub fn svd_centered<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    rank: usize,
    seed: Option<u64>,
) -> Result<SvdRec<T>> {
    irlba::svd_centered(a, rank, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{dense_of, gen_lowrank, gen_sparse, reference_singular_values};

    /// Every solver must agree with a dense LAPACK reference on the same matrix, to
    /// each one's own accuracy class. This is the cross-algorithm contract.
    #[test]
    fn all_solvers_agree_with_lapack() {
        let a = gen_lowrank(300, 100, 10, 101);
        let want = reference_singular_values(&dense_of(&a));
        let rank = 10;

        let by_irlba = irlba::svd_seed(&a, rank, 42).unwrap();
        let by_random = randomized::svd_with(
            &a,
            &randomized::RandomizedConfig::new(rank)
                .seed(42)
                .power_iterations(4),
            None,
        )
        .unwrap();
        let by_default = svd_seed(&a, rank, 42).unwrap();

        for i in 0..rank {
            for (name, got, tol) in [
                ("irlba", by_irlba.s[i], 1e-9),
                ("randomized", by_random.s[i], 1e-6),
                ("top-level default", by_default.s[i], 1e-9),
            ] {
                let rel = (got - want[i]).abs() / want[i];
                assert!(
                    rel < tol,
                    "{name} triplet {i}: {got:.12e} vs LAPACK {:.12e} (rel {rel:.3e})",
                    want[i]
                );
            }
        }
    }

    /// The top-level entry point must be IRLBA, as documented.
    #[test]
    fn top_level_dispatches_to_irlba() {
        let a = gen_sparse(120, 60, 0.1, 7);
        let got = svd(&a, 5).unwrap();
        assert_eq!(got.diagnostics.algorithm, Algorithm::Irlba);
    }

    /// `u32`-indexed and `u64`-indexed matrices must give identical answers — the
    /// memory win must not cost accuracy.
    #[test]
    fn index_width_does_not_change_results() {
        use sprs::TriMatI;
        let a32 = gen_sparse(200, 80, 0.08, 13);

        let mut t = TriMatI::<f64, u64>::new((200, 80));
        for (v, (i, j)) in a32.iter() {
            t.add_triplet(i as usize, j as usize, *v);
        }
        let a64: SvdMat<f64, u64, u64> = t.to_csr::<u64>();

        let x = svd_seed(&a32, 10, 42).unwrap();
        let y = svd_seed(&a64, 10, 42).unwrap();
        for (p, q) in x.s.iter().zip(y.s.iter()) {
            approx::assert_relative_eq!(p, q, max_relative = 1e-12);
        }
    }

    /// The documented memory claim, checked against the buffers sprs actually holds.
    #[test]
    fn u32_indices_are_smaller_than_usize_indices() {
        let a = gen_sparse(2000, 500, 0.02, 3);
        let nnz = a.nnz();
        let rows = a.rows();

        assert_eq!(a.indices().len(), nnz);
        assert_eq!(a.data().len(), nnz);

        let ours = (rows + 1) * std::mem::size_of::<u64>()
            + nnz * std::mem::size_of::<u32>()
            + nnz * std::mem::size_of::<f64>();
        let usize_everywhere = (rows + 1) * std::mem::size_of::<usize>()
            + nnz * std::mem::size_of::<usize>()
            + nnz * std::mem::size_of::<f64>();

        let saving = 1.0 - (ours as f64 / usize_everywhere as f64);
        assert!(
            saving > 0.2,
            "expected >20% smaller, got {:.1}%",
            saving * 100.0
        );
    }
}
