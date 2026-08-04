//! Parallel sparse × dense kernels.
//!
//! # Why there are two kernels
//!
//! A compressed matrix can only be walked along its outer dimension. For a CSR matrix
//! that is rows, so:
//!
//! - `A · D` writes output row `i` from sparse row `i`. Threads own disjoint output
//!   rows, so this needs **no scratch and no reduction** — [`gather_mul`].
//! - `Aᵀ · D` reads sparse row `i` and scatters into output rows `j` for every column
//!   `j` present in that row. Threads collide, so accumulation is needed —
//!   [`scatter_mul`].
//!
//! `transpose_view()` does not escape this: it relabels a CSR matrix as a CSC view of
//! the transpose, but the traversable dimension is unchanged. What it *does* buy is
//! that a CSC-stored matrix gets the disjoint kernel for `Aᵀ · D` for free, so callers
//! holding CSC pay nothing for the transposed direction.
//!
//! # Scratch budgeting
//!
//! The 1.x code allocated one full `n × k` buffer **per chunk**, with chunk count
//! driven by matrix size — 64 chunks on a 200k-row matrix, so ~922 MiB of scratch for
//! a single 30000 × 60 product. Here the accumulator count is the *thread* count, and
//! if `threads × n × k` still exceeds [`DEFAULT_SCRATCH_BUDGET`] the dense columns are
//! processed in blocks so the bound always holds.

// Numeric kernels index several arrays in step from one loop variable, and
// offset arithmetic is load-bearing; iterator rewrites obscure which array an
// index belongs to.
#![allow(clippy::needless_range_loop)]

use crate::types::SvdFloat;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;
use sprs::{CsMatViewI, SpIndex};

/// Upper bound on transient scratch for scatter-direction products, in bytes.
///
/// Exceeding this trades an extra pass over the sparse indices for a smaller
/// footprint. 64 MiB keeps the accumulators comfortably inside last-level cache
/// pressure on typical machines while still amortising index reads over many columns.
pub const DEFAULT_SCRATCH_BUDGET: usize = 64 << 20;

/// Below this many output elements, threading costs more than it saves.
const SERIAL_ELEMS: usize = 8 << 10;

#[inline]
fn threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// `y += alpha * x`, over contiguous slices so LLVM can vectorise it.
#[inline]
fn axpy<T: SvdFloat>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Split `[0, outer)` into `p` contiguous ranges holding roughly equal non-zeros.
///
/// Row counts are a poor proxy for work when the non-zero distribution is skewed,
/// which it reliably is for count matrices (a few rows carry a large share of the
/// mass). Returns `p + 1` boundaries.
fn nnz_balanced_split<N, I: SpIndex, Iptr: SpIndex>(
    m: &CsMatViewI<N, I, Iptr>,
    p: usize,
) -> Vec<usize> {
    let outer = m.outer_dims();
    let total = m.nnz();
    let mut bounds = Vec::with_capacity(p + 1);
    bounds.push(0);
    if p <= 1 || outer == 0 || total == 0 {
        bounds.push(outer);
        while bounds.len() < p + 1 {
            bounds.push(outer);
        }
        return bounds;
    }
    let mut acc = 0usize;
    let mut next = 1usize;
    for i in 0..outer {
        acc += m.outer_view(i).map_or(0, |v| v.nnz());
        // Advance past every boundary this row crosses, so a single heavy row cannot
        // leave later partitions unassigned.
        while next < p && acc * p >= total * next {
            bounds.push(i + 1);
            next += 1;
        }
    }
    while bounds.len() < p + 1 {
        bounds.push(outer);
    }
    bounds
}

/// `out = lhs · rhs` where `lhs` is CSR. Write-disjoint: no scratch, no reduction.
///
/// `out` is fully overwritten.
pub fn gather_mul<T, I, Iptr>(
    lhs: CsMatViewI<T, I, Iptr>,
    rhs: ArrayView2<T>,
    mut out: ArrayViewMut2<T>,
) where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(lhs.is_csr(), "gather_mul requires CSR storage");
    assert_eq!(lhs.cols(), rhs.nrows(), "gather_mul: lhs.cols != rhs.rows");
    assert_eq!(lhs.rows(), out.nrows(), "gather_mul: lhs.rows != out.rows");
    assert_eq!(rhs.ncols(), out.ncols(), "gather_mul: rhs.cols != out.cols");

    let k = rhs.ncols();
    let m = lhs.rows();
    if m == 0 || k == 0 {
        out.fill(T::zero());
        return;
    }

    let row_op = |i: usize, orow: &mut [T], rhs: &ArrayView2<T>| {
        orow.fill(T::zero());
        let Some(row) = lhs.outer_view(i) else { return };
        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
            let rrow = rhs.row(j.index());
            // rhs is built row-major by this crate; fall back if a caller passes a view.
            match rrow.as_slice() {
                Some(sl) => axpy(v, sl, orow),
                None => {
                    for (o, &r) in orow.iter_mut().zip(rrow.iter()) {
                        *o += v * r;
                    }
                }
            }
        }
    };

    if m * k <= SERIAL_ELEMS {
        for i in 0..m {
            let mut orow = out.row_mut(i);
            match orow.as_slice_mut() {
                Some(sl) => row_op(i, sl, &rhs),
                None => {
                    let mut tmp = vec![T::zero(); k];
                    row_op(i, &mut tmp, &rhs);
                    for (o, t) in orow.iter_mut().zip(tmp) {
                        *o = t;
                    }
                }
            }
        }
        return;
    }

    // 4 chunks per thread lets rayon steal work when rows are unevenly filled.
    let chunk = m.div_ceil(threads() * 4).max(1);
    out.axis_chunks_iter_mut(Axis(0), chunk)
        .into_par_iter()
        .enumerate()
        .for_each(|(ci, mut block)| {
            let base = ci * chunk;
            for (local, mut orow) in block.rows_mut().into_iter().enumerate() {
                let i = base + local;
                match orow.as_slice_mut() {
                    Some(sl) => row_op(i, sl, &rhs),
                    None => {
                        let mut tmp = vec![T::zero(); k];
                        row_op(i, &mut tmp, &rhs);
                        for (o, t) in orow.iter_mut().zip(tmp) {
                            *o = t;
                        }
                    }
                }
            }
        });
}

/// `out = lhsᵀ · rhs` where `lhs` is CSR. Scatter direction: uses one accumulator per
/// thread, blocking over the columns of `rhs` to keep scratch under `budget`.
///
/// `out` is fully overwritten.
pub fn scatter_mul<T, I, Iptr>(
    lhs: CsMatViewI<T, I, Iptr>,
    rhs: ArrayView2<T>,
    mut out: ArrayViewMut2<T>,
    budget: usize,
) where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(lhs.is_csr(), "scatter_mul requires CSR storage");
    assert_eq!(lhs.rows(), rhs.nrows(), "scatter_mul: lhs.rows != rhs.rows");
    assert_eq!(lhs.cols(), out.nrows(), "scatter_mul: lhs.cols != out.rows");
    assert_eq!(
        rhs.ncols(),
        out.ncols(),
        "scatter_mul: rhs.cols != out.cols"
    );

    let (m, n, k) = (lhs.rows(), lhs.cols(), rhs.ncols());
    out.fill(T::zero());
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // Serial path: accumulate straight into `out`, no scratch at all.
    let p = threads();
    if p == 1 || n * k <= SERIAL_ELEMS {
        for i in 0..m {
            let Some(row) = lhs.outer_view(i) else {
                continue;
            };
            let rrow = rhs.row(i);
            let rslice = rrow.as_slice();
            for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                let mut orow = out.row_mut(j.index());
                match (orow.as_slice_mut(), rslice) {
                    (Some(o), Some(r)) => axpy(v, r, o),
                    _ => {
                        for (o, &r) in orow.iter_mut().zip(rrow.iter()) {
                            *o += v * r;
                        }
                    }
                }
            }
        }
        return;
    }

    // Widest column block whose p accumulators fit the budget.
    let bytes_per_col = p.saturating_mul(n).saturating_mul(std::mem::size_of::<T>());
    let kb = budget
        .checked_div(bytes_per_col)
        .map_or(k, |wide| wide.clamp(1, k));

    let bounds = nnz_balanced_split(&lhs, p);

    for cstart in (0..k).step_by(kb) {
        let cend = (cstart + kb).min(k);
        let width = cend - cstart;
        let rhs_blk = rhs.slice(s![.., cstart..cend]);

        let partials: Vec<Array2<T>> = (0..p)
            .into_par_iter()
            .map(|t| {
                let (lo, hi) = (bounds[t], bounds[t + 1]);
                let mut acc = Array2::<T>::zeros((n, width));
                for i in lo..hi {
                    let Some(row) = lhs.outer_view(i) else {
                        continue;
                    };
                    let rrow = rhs_blk.row(i);
                    // rhs_blk is a column slice, so its rows stay contiguous only when
                    // the block spans every column; handle both.
                    let rslice = rrow.as_slice();
                    for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                        let jj = j.index();
                        let mut arow = acc.row_mut(jj);
                        let aslice = arow.as_slice_mut().expect("owned array is contiguous");
                        match rslice {
                            Some(r) => axpy(v, r, aslice),
                            None => {
                                for (a, &r) in aslice.iter_mut().zip(rrow.iter()) {
                                    *a += v * r;
                                }
                            }
                        }
                    }
                }
                acc
            })
            .collect();

        // Reduce, parallel over disjoint output rows.
        let mut out_blk = out.slice_mut(s![.., cstart..cend]);
        let rchunk = n.div_ceil(p * 4).max(1);
        out_blk
            .axis_chunks_iter_mut(Axis(0), rchunk)
            .into_par_iter()
            .enumerate()
            .for_each(|(ci, mut block)| {
                let base = ci * rchunk;
                for (local, mut orow) in block.rows_mut().into_iter().enumerate() {
                    let gi = base + local;
                    for acc in &partials {
                        let arow = acc.row(gi);
                        for (o, &a) in orow.iter_mut().zip(arow.iter()) {
                            *o += a;
                        }
                    }
                }
            });
    }
}

/// `y = lhs · x` where `lhs` is CSR. Write-disjoint.
pub fn gather_mul_vec<T, I, Iptr>(lhs: CsMatViewI<T, I, Iptr>, x: &[T], y: &mut [T])
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(lhs.is_csr(), "gather_mul_vec requires CSR storage");
    assert_eq!(lhs.cols(), x.len(), "gather_mul_vec: lhs.cols != x.len");
    assert_eq!(lhs.rows(), y.len(), "gather_mul_vec: lhs.rows != y.len");

    let dot = |i: usize| -> T {
        let Some(row) = lhs.outer_view(i) else {
            return T::zero();
        };
        let mut sum = T::zero();
        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
            sum += v * x[j.index()];
        }
        sum
    };

    if y.len() <= SERIAL_ELEMS {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = dot(i);
        }
        return;
    }
    let chunk = y.len().div_ceil(threads() * 4).max(1);
    y.par_chunks_mut(chunk).enumerate().for_each(|(ci, blk)| {
        let base = ci * chunk;
        for (local, yi) in blk.iter_mut().enumerate() {
            *yi = dot(base + local);
        }
    });
}

/// `y = lhsᵀ · x` where `lhs` is CSR. Scatter direction, one accumulator per thread.
pub fn scatter_mul_vec<T, I, Iptr>(lhs: CsMatViewI<T, I, Iptr>, x: &[T], y: &mut [T])
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(lhs.is_csr(), "scatter_mul_vec requires CSR storage");
    assert_eq!(lhs.rows(), x.len(), "scatter_mul_vec: lhs.rows != x.len");
    assert_eq!(lhs.cols(), y.len(), "scatter_mul_vec: lhs.cols != y.len");

    let (m, n) = (lhs.rows(), lhs.cols());
    y.fill(T::zero());
    if m == 0 || n == 0 {
        return;
    }

    let p = threads();
    // A single vector's worth of accumulator is p * n scalars; only worth splitting
    // when there is enough work to pay for the reduction.
    if p == 1 || lhs.nnz() <= SERIAL_ELEMS {
        for i in 0..m {
            let Some(row) = lhs.outer_view(i) else {
                continue;
            };
            let xi = x[i];
            if xi.is_zero() {
                continue;
            }
            for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                y[j.index()] += v * xi;
            }
        }
        return;
    }

    let bounds = nnz_balanced_split(&lhs, p);
    let partials: Vec<Vec<T>> = (0..p)
        .into_par_iter()
        .map(|t| {
            let (lo, hi) = (bounds[t], bounds[t + 1]);
            let mut acc = vec![T::zero(); n];
            for i in lo..hi {
                let Some(row) = lhs.outer_view(i) else {
                    continue;
                };
                let xi = x[i];
                if xi.is_zero() {
                    continue;
                }
                for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                    acc[j.index()] += v * xi;
                }
            }
            acc
        })
        .collect();

    let chunk = n.div_ceil(p * 4).max(1);
    y.par_chunks_mut(chunk).enumerate().for_each(|(ci, blk)| {
        let base = ci * chunk;
        for (local, yi) in blk.iter_mut().enumerate() {
            let gi = base + local;
            for acc in &partials {
                *yi += acc[gi];
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};
    use sprs::{CsMatI, TriMatI};

    fn tiny() -> CsMatI<f64, u32, u64> {
        // [ 1 0 2 ]
        // [ 0 3 0 ]
        // [ 4 0 5 ]
        // [ 0 6 0 ]
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

    #[test]
    fn gather_matches_dense() {
        let a = tiny();
        let rhs = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let mut out = Array2::zeros((4, 2));
        gather_mul(a.view(), rhs.view(), out.view_mut());
        let expect = dense_of(&a).dot(&rhs);
        assert_eq!(out, expect);
    }

    #[test]
    fn scatter_matches_dense() {
        let a = tiny();
        let rhs = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let mut out = Array2::zeros((3, 2));
        scatter_mul(a.view(), rhs.view(), out.view_mut(), DEFAULT_SCRATCH_BUDGET);
        let expect = dense_of(&a).t().dot(&rhs);
        assert_eq!(out, expect);
    }

    /// A budget of 0 forces `kb == 1`, exercising the multi-pass path.
    #[test]
    fn scatter_column_blocking_matches_single_pass() {
        let a = tiny();
        let rhs = arr2(&[
            [1.0, 2.0, 9.0],
            [3.0, 4.0, 8.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 6.0],
        ]);
        let mut wide = Array2::zeros((3, 3));
        let mut narrow = Array2::zeros((3, 3));
        scatter_mul(a.view(), rhs.view(), wide.view_mut(), usize::MAX);
        scatter_mul(a.view(), rhs.view(), narrow.view_mut(), 0);
        assert_eq!(wide, narrow);
        assert_eq!(wide, dense_of(&a).t().dot(&rhs));
    }

    #[test]
    fn matvecs_match_dense() {
        let a = tiny();
        let d = dense_of(&a);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 4];
        gather_mul_vec(a.view(), &x, &mut y);
        assert_eq!(y, d.dot(&ndarray::arr1(&x)).to_vec());

        let xt = vec![1.0, 2.0, 3.0, 4.0];
        let mut yt = vec![0.0; 3];
        scatter_mul_vec(a.view(), &xt, &mut yt);
        assert_eq!(yt, d.t().dot(&ndarray::arr1(&xt)).to_vec());
    }

    #[test]
    fn nnz_split_covers_all_rows_and_is_monotone() {
        let a = tiny();
        for p in 1..=8 {
            let b = nnz_balanced_split(&a.view(), p);
            assert_eq!(b.len(), p + 1);
            assert_eq!(b[0], 0);
            assert_eq!(*b.last().unwrap(), a.rows());
            assert!(b.windows(2).all(|w| w[0] <= w[1]), "not monotone: {b:?}");
        }
    }

    /// A matrix whose non-zeros are concentrated in one row: row-count splitting
    /// would put all the work on a single thread.
    #[test]
    fn nnz_split_handles_skew() {
        let mut t = TriMatI::<f64, u32>::new((100, 50));
        for j in 0..50 {
            t.add_triplet(0, j, 1.0);
        }
        for i in 1..100 {
            t.add_triplet(i, 0, 1.0);
        }
        let a: CsMatI<f64, u32, u64> = t.to_csr();
        let b = nnz_balanced_split(&a.view(), 4);
        assert_eq!(b[0], 0);
        assert_eq!(*b.last().unwrap(), 100);
        assert!(b.windows(2).all(|w| w[0] <= w[1]));
        // The heavy first row must be isolated into the first partition.
        assert_eq!(
            b[1], 1,
            "expected the 50-nnz row to form its own partition: {b:?}"
        );
    }
}
