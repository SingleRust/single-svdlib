//! Sparse operands and the traits the solvers consume.
//!
//! # Index widths
//!
//! [`SvdMat`] defaults to `u32` column indices with `u64` row pointers. Against
//! `usize`-everywhere that is 12 bytes per non-zero instead of 16 for `f64` data
//! (8 instead of 16 for `f32`), and the separate pointer width keeps matrices with
//! more than `u32::MAX` non-zeros representable. Callers who need wider indices can
//! name them: `SvdMat<f64, u64, u64>`.

pub mod kernels;
pub mod masked;

use crate::types::SvdFloat;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use sprs::{CsMatI, CsMatViewI, SpIndex};

pub use kernels::DEFAULT_SCRATCH_BUDGET;
pub use masked::MaskedCsMat;

/// An owned sparse matrix. Defaults to `u32` indices and `u64` row pointers.
pub type SvdMat<T, I = u32, Iptr = u64> = CsMatI<T, I, Iptr>;
/// A borrowed sparse matrix. Defaults to `u32` indices and `u64` row pointers.
pub type SvdMatView<'a, T, I = u32, Iptr = u64> = CsMatViewI<'a, T, I, Iptr>;

/// The operand interface the Krylov solvers ([`crate::lanczos`], [`crate::irlba`])
/// need: shape and a matrix-vector product.
///
/// In 1.x this trait also carried the four blocked-product methods, which meant every
/// built-in implementation left them as `todo!()` and the randomized solvers panicked
/// on all three stock matrix types. Those methods now live on [`SparseMatDense`],
/// which has working defaults, so the gap cannot reappear.
pub trait SparseMat<T: SvdFloat>: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn nnz(&self) -> usize;

    /// `y = A·x` when `trans` is false, `y = Aᵀ·x` when true.
    ///
    /// `y` is fully overwritten. `x` must have length `cols()` (`rows()` when
    /// transposed) and `y` length `rows()` (`cols()` when transposed).
    fn mul_vec(&self, x: &[T], y: &mut [T], trans: bool);

    /// `‖A‖²_F`. Accumulated in `f64` even for `f32` data — this sums every non-zero,
    /// and an `f32` accumulator would drop the tail or overflow.
    fn squared_frobenius(&self) -> f64;

    /// `‖A − 1·meansᵀ‖²_F`.
    ///
    /// The default is the closed form `‖A‖²_F − rows·Σⱼ meanⱼ²`, which needs no pass over
    /// the matrix but cancels badly when columns are large against their own spread: on
    /// `f32` columns of `offset + O(1)` noise it is 3e-8 wrong at offset 0 and **5e-1**
    /// wrong at offset 1000. Fine for counts data, where most entries are zero. The
    /// built-in types override it with a per-entry sum that never cancels; do the same if
    /// your data carries an offset.
    fn centered_squared_frobenius(&self, means: ArrayView1<T>) -> f64 {
        let shift: f64 = means.iter().map(|&m| m.to_f64() * m.to_f64()).sum();
        (self.squared_frobenius() - self.rows() as f64 * shift).max(0.0)
    }
}

/// `‖A‖²_F`, or the centered version when `means` is supplied. What the solvers store in
/// [`SvdRec::total_squared_norm`](crate::SvdRec).
pub fn total_squared_norm<T: SvdFloat, M: SparseMat<T> + ?Sized>(
    a: &M,
    means: Option<ArrayView1<T>>,
) -> f64 {
    match means {
        Some(m) => a.centered_squared_frobenius(m),
        None => a.squared_frobenius(),
    }
}

/// `stored + Σⱼ (rows − nnzⱼ)·meanⱼ²`, where `stored` is `Σ (aᵢⱼ − meanⱼ)²` over the
/// entries that exist. Every unstored entry contributes `meanⱼ²`.
#[inline]
fn centered_from_parts<T: SvdFloat>(
    stored: f64,
    col_nnz: &[usize],
    means: ArrayView1<T>,
    rows: usize,
) -> f64 {
    let missing: f64 = col_nnz
        .iter()
        .zip(means.iter())
        .map(|(&n, &m)| {
            let m = m.to_f64();
            (rows - n) as f64 * m * m
        })
        .sum();
    (stored + missing).max(0.0)
}

/// Blocked products, needed by the randomized solvers.
///
/// [`mul_dense`](Self::mul_dense) has no default — an implementor must supply it — but
/// [`col_means`](Self::col_means) and
/// [`mul_dense_centered`](Self::mul_dense_centered) do, so mean-centering comes for
/// free once the plain product works.
pub trait SparseMatDense<T: SvdFloat>: SparseMat<T> {
    /// `out = A·rhs` when `trans` is false, `out = Aᵀ·rhs` when true.
    ///
    /// `out` is fully overwritten.
    fn mul_dense(&self, rhs: ArrayView2<T>, out: ArrayViewMut2<T>, trans: bool);

    /// Column means, length `cols()`.
    ///
    /// The default computes `Aᵀ·1 / rows()`, which routes through whichever
    /// [`mul_vec`](SparseMat::mul_vec) direction is cheapest for the storage order.
    fn col_means(&self) -> Array1<T> {
        let m = self.rows();
        let ones = vec![T::one(); m];
        let mut sums = vec![T::zero(); self.cols()];
        self.mul_vec(&ones, &mut sums, true);
        let scale = if m == 0 {
            T::zero()
        } else {
            T::one() / T::from_f64_val(m as f64)
        };
        Array1::from_vec(sums) * scale
    }

    /// The product against `A - 1·meansᵀ`, without ever forming it — centering a sparse
    /// matrix would destroy its sparsity, so it goes in as the rank-1 update it is:
    ///
    /// - `trans == false`: `(A - 1·mᵀ)·D = A·D - 1·(mᵀ·D)`
    /// - `trans == true`:  `(A - 1·mᵀ)ᵀ·D = Aᵀ·D - m·(1ᵀ·D)`
    ///
    /// The correction is one length-`k` vector either way, so it adds
    /// `O(k·(rows + cols))` and allocates nothing else.
    fn mul_dense_centered(
        &self,
        rhs: ArrayView2<T>,
        mut out: ArrayViewMut2<T>,
        trans: bool,
        means: ArrayView1<T>,
    ) {
        assert_eq!(
            means.len(),
            self.cols(),
            "mul_dense_centered: means must have length cols()"
        );
        self.mul_dense(rhs, out.view_mut(), trans);
        apply_centering(rhs, out, trans, means);
    }
}

/// Subtract the rank-1 centering term from an uncentered product. Public so an
/// implementor writing a fused `mul_dense_centered` can reuse it.
pub fn apply_centering<T: SvdFloat>(
    rhs: ArrayView2<T>,
    mut out: ArrayViewMut2<T>,
    trans: bool,
    means: ArrayView1<T>,
) {
    let k = rhs.ncols();
    if k == 0 {
        return;
    }
    if !trans {
        // corr[c] = Σ_j means[j] · rhs[j, c]; subtract from every output row.
        debug_assert_eq!(rhs.nrows(), means.len());
        let mut corr = vec![T::zero(); k];
        for (j, &mj) in means.iter().enumerate() {
            if mj.is_zero() {
                continue;
            }
            for (c, cv) in corr.iter_mut().enumerate() {
                *cv += mj * rhs[[j, c]];
            }
        }
        for mut orow in out.rows_mut() {
            for (o, &c) in orow.iter_mut().zip(corr.iter()) {
                *o -= c;
            }
        }
    } else {
        // colsum[c] = Σ_i rhs[i, c]; subtract means[j] · colsum[c] from out[j, c].
        let mut colsum = vec![T::zero(); k];
        for row in rhs.rows() {
            for (c, cv) in colsum.iter_mut().enumerate() {
                *cv += row[c];
            }
        }
        debug_assert_eq!(out.nrows(), means.len());
        for (j, mut orow) in out.rows_mut().into_iter().enumerate() {
            let mj = means[j];
            if mj.is_zero() {
                continue;
            }
            for (o, &cs) in orow.iter_mut().zip(colsum.iter()) {
                *o -= mj * cs;
            }
        }
    }
}

/// A CSR view of any compressed matrix, plus a flag for whether the view is the
/// transpose. CSC is bit-for-bit the CSR of its own transpose, so `transpose_view()`
/// reaches it for free — every kernel is written once against CSR and the caller's
/// `trans` is XORed with the flag.
#[inline]
fn csr_view<T, I: SpIndex, Iptr: SpIndex>(
    m: &CsMatI<T, I, Iptr>,
) -> (CsMatViewI<'_, T, I, Iptr>, bool) {
    if m.is_csr() {
        (m.view(), false)
    } else {
        (m.transpose_view(), true)
    }
}

impl<T, I, Iptr> SparseMat<T> for CsMatI<T, I, Iptr>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn rows(&self) -> usize {
        CsMatI::rows(self)
    }
    fn cols(&self) -> usize {
        CsMatI::cols(self)
    }
    fn nnz(&self) -> usize {
        CsMatI::nnz(self)
    }

    fn mul_vec(&self, x: &[T], y: &mut [T], trans: bool) {
        let (view, flipped) = csr_view(self);
        if trans ^ flipped {
            kernels::scatter_mul_vec(view, x, y);
        } else {
            kernels::gather_mul_vec(view, x, y);
        }
    }

    fn squared_frobenius(&self) -> f64 {
        // Storage order doesn't matter — every value appears once.
        self.data()
            .par_iter()
            .map(|&v| {
                let x = v.to_f64();
                x * x
            })
            .sum()
    }

    fn centered_squared_frobenius(&self, means: ArrayView1<T>) -> f64 {
        assert_eq!(
            means.len(),
            SparseMat::cols(self),
            "centered_squared_frobenius: means must have length cols()"
        );
        let rows = SparseMat::rows(self);
        let cols = SparseMat::cols(self);

        if self.is_csc() {
            // Outer dimension is the column, so the count comes for free.
            let total: f64 = (0..cols)
                .into_par_iter()
                .map(|j| {
                    let m = means[j].to_f64();
                    let (sum, n) = self.outer_view(j).map_or((0.0, 0), |col| {
                        (
                            col.iter()
                                .map(|(_, &v)| {
                                    let e = v.to_f64() - m;
                                    e * e
                                })
                                .sum::<f64>(),
                            col.nnz(),
                        )
                    });
                    sum + (rows - n) as f64 * m * m
                })
                .sum();
            return total.max(0.0);
        }

        // CSR: count per column first, then sum the entries.
        let mut col_nnz = vec![0usize; cols];
        for j in self.indices() {
            col_nnz[j.index()] += 1;
        }
        let stored: f64 = (0..rows)
            .into_par_iter()
            .map(|i| {
                self.outer_view(i).map_or(0.0, |row| {
                    row.iter()
                        .map(|(j, &v)| {
                            let e = v.to_f64() - means[j].to_f64();
                            e * e
                        })
                        .sum()
                })
            })
            .sum();
        centered_from_parts(stored, &col_nnz, means, rows)
    }
}

impl<T, I, Iptr> SparseMatDense<T> for CsMatI<T, I, Iptr>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn mul_dense(&self, rhs: ArrayView2<T>, out: ArrayViewMut2<T>, trans: bool) {
        let (view, flipped) = csr_view(self);
        if trans ^ flipped {
            kernels::scatter_mul(view, rhs, out, DEFAULT_SCRATCH_BUDGET);
        } else {
            kernels::gather_mul(view, rhs, out);
        }
    }
}

// Blanket forwarding so `&M` and `Arc<M>` work wherever `M` does.
impl<T: SvdFloat, M: SparseMat<T> + ?Sized> SparseMat<T> for &M {
    fn rows(&self) -> usize {
        (**self).rows()
    }
    fn cols(&self) -> usize {
        (**self).cols()
    }
    fn nnz(&self) -> usize {
        (**self).nnz()
    }
    fn mul_vec(&self, x: &[T], y: &mut [T], trans: bool) {
        (**self).mul_vec(x, y, trans)
    }
    fn squared_frobenius(&self) -> f64 {
        (**self).squared_frobenius()
    }
    fn centered_squared_frobenius(&self, means: ArrayView1<T>) -> f64 {
        (**self).centered_squared_frobenius(means)
    }
}

impl<T: SvdFloat, M: SparseMatDense<T> + ?Sized> SparseMatDense<T> for &M {
    fn mul_dense(&self, rhs: ArrayView2<T>, out: ArrayViewMut2<T>, trans: bool) {
        (**self).mul_dense(rhs, out, trans)
    }
    fn col_means(&self) -> Array1<T> {
        (**self).col_means()
    }
    fn mul_dense_centered(
        &self,
        rhs: ArrayView2<T>,
        out: ArrayViewMut2<T>,
        trans: bool,
        means: ArrayView1<T>,
    ) {
        (**self).mul_dense_centered(rhs, out, trans, means)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};
    use sprs::TriMatI;

    fn tiny_csr() -> SvdMat<f64> {
        let mut t = TriMatI::<f64, u32>::new((4, 3));
        t.add_triplet(0, 0, 1.0);
        t.add_triplet(0, 2, 2.0);
        t.add_triplet(1, 1, 3.0);
        t.add_triplet(2, 0, 4.0);
        t.add_triplet(2, 2, 5.0);
        t.add_triplet(3, 1, 6.0);
        t.to_csr::<u64>()
    }

    use crate::testing::dense_of;

    /// CSR and CSC hold the same matrix, so every product must agree — this is what
    /// makes the `transpose_view` dispatch safe.
    #[test]
    fn csr_and_csc_agree() {
        let csr = tiny_csr();
        let csc = csr.to_other_storage();
        assert!(csr.is_csr() && csc.is_csc());
        let d = dense_of(&csr);

        let rhs = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let mut a = Array2::zeros((4, 2));
        let mut b = Array2::zeros((4, 2));
        SparseMatDense::mul_dense(&csr, rhs.view(), a.view_mut(), false);
        SparseMatDense::mul_dense(&csc, rhs.view(), b.view_mut(), false);
        assert_eq!(a, d.dot(&rhs));
        assert_eq!(b, d.dot(&rhs));

        let rhs_t = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let mut at = Array2::zeros((3, 1));
        let mut bt = Array2::zeros((3, 1));
        SparseMatDense::mul_dense(&csr, rhs_t.view(), at.view_mut(), true);
        SparseMatDense::mul_dense(&csc, rhs_t.view(), bt.view_mut(), true);
        assert_eq!(at, d.t().dot(&rhs_t));
        assert_eq!(bt, d.t().dot(&rhs_t));
    }

    #[test]
    fn col_means_match_dense() {
        let a = tiny_csr();
        let d = dense_of(&a);
        let got = SparseMatDense::col_means(&a);
        let want = d.mean_axis(ndarray::Axis(0)).unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-14);
        }
        // The CSC path must agree.
        let got_csc = SparseMatDense::col_means(&a.to_other_storage());
        for (g, w) in got_csc.iter().zip(want.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-14);
        }
    }

    /// The rank-1 correction must equal explicitly forming the dense centered matrix.
    #[test]
    fn centering_matches_explicit_dense_centering() {
        let a = tiny_csr();
        let d = dense_of(&a);
        let means = SparseMatDense::col_means(&a);
        let centered = &d - &means.view().insert_axis(ndarray::Axis(0));

        let rhs = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let mut out = Array2::zeros((4, 2));
        SparseMatDense::mul_dense_centered(&a, rhs.view(), out.view_mut(), false, means.view());
        let want = centered.dot(&rhs);
        for (g, w) in out.iter().zip(want.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12);
        }

        let rhs_t = arr2(&[[1.0, 0.5], [2.0, 1.5], [3.0, 2.5], [4.0, 3.5]]);
        let mut out_t = Array2::zeros((3, 2));
        SparseMatDense::mul_dense_centered(&a, rhs_t.view(), out_t.view_mut(), true, means.view());
        let want_t = centered.t().dot(&rhs_t);
        for (g, w) in out_t.iter().zip(want_t.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12);
        }
    }

    #[test]
    fn squared_frobenius_matches_dense_and_is_storage_agnostic() {
        let a = tiny_csr();
        let d = dense_of(&a);
        let want: f64 = d.iter().map(|&v| v * v).sum();
        approx::assert_relative_eq!(SparseMat::squared_frobenius(&a), want, max_relative = 1e-14);
        // Every stored value appears once whichever way the matrix is compressed.
        approx::assert_relative_eq!(
            SparseMat::squared_frobenius(&a.to_other_storage()),
            want,
            max_relative = 1e-14
        );
    }

    /// Must equal building the centered matrix and summing it, in both storage orders.
    #[test]
    fn centered_squared_frobenius_matches_explicit_centering() {
        let a = tiny_csr();
        let d = dense_of(&a);
        let means = SparseMatDense::col_means(&a);
        let centered = &d - &means.view().insert_axis(ndarray::Axis(0));
        let want: f64 = centered.iter().map(|&v| v * v).sum();

        approx::assert_relative_eq!(
            SparseMat::centered_squared_frobenius(&a, means.view()),
            want,
            max_relative = 1e-12
        );
        // CSC reaches the same answer by iterating columns instead of rows.
        approx::assert_relative_eq!(
            SparseMat::centered_squared_frobenius(&a.to_other_storage(), means.view()),
            want,
            max_relative = 1e-12
        );
    }

    /// Centering can only remove norm, never add it, and never take it below zero.
    #[test]
    fn centering_never_increases_the_norm() {
        let a = tiny_csr();
        let means = SparseMatDense::col_means(&a);
        let raw = SparseMat::squared_frobenius(&a);
        let centered = SparseMat::centered_squared_frobenius(&a, means.view());
        assert!(centered >= 0.0);
        assert!(centered <= raw);
    }

    /// The closed form was 51% wrong at an `f32` offset of 1000. The per-entry form must
    /// stay at machine precision and not degrade as the offset grows.
    #[test]
    fn centered_norm_survives_a_large_column_offset() {
        for offset in [0.0f32, 10.0, 100.0, 1000.0] {
            let (rows, cols) = (200usize, 20usize);
            let mut t = sprs::TriMatI::<f32, u32>::new((rows, cols));
            let mut dense = Array2::<f64>::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    let v = offset + (((i * 7 + j * 3) % 11) as f32 - 5.0) * 0.1;
                    t.add_triplet(i, j, v);
                    dense[[i, j]] = v as f64;
                }
            }
            let a: SvdMat<f32> = t.to_csr::<u64>();
            let means = SparseMatDense::col_means(&a);

            // The reference uses the same f32 means the solver would, so this measures
            // the summation and nothing else.
            let mut want = 0.0f64;
            for i in 0..rows {
                for j in 0..cols {
                    let e = dense[[i, j]] - means[j] as f64;
                    want += e * e;
                }
            }

            let got = SparseMat::centered_squared_frobenius(&a, means.view());
            let rel = (got - want).abs() / want;
            assert!(
                rel < 1e-6,
                "offset {offset}: centered norm {got:.6e} vs {want:.6e} (rel {rel:.3e})"
            );
        }
    }

    #[test]
    fn mul_vec_matches_dense_both_directions() {
        let a = tiny_csr();
        let d = dense_of(&a);
        let mut y = vec![0.0; 4];
        SparseMat::mul_vec(&a, &[1.0, 2.0, 3.0], &mut y, false);
        assert_eq!(y, d.dot(&ndarray::arr1(&[1.0, 2.0, 3.0])).to_vec());

        let mut yt = vec![0.0; 3];
        SparseMat::mul_vec(&a, &[1.0, 2.0, 3.0, 4.0], &mut yt, true);
        assert_eq!(
            yt,
            d.t().dot(&ndarray::arr1(&[1.0, 2.0, 3.0, 4.0])).to_vec()
        );
    }
}
