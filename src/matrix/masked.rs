//! A submatrix view over a sparse matrix.
//!
//! Selects rows, columns, or both, and presents the result as a matrix in its own right
//! — the solvers see only the selected entries. Nothing is copied, nothing is
//! reindexed, and the underlying matrix is never modified.
//!
//! # Cost
//!
//! Row selection is free: the matrix is CSR, so a row is an outer index and skipping one
//! means not visiting it. Masking rows makes every product *cheaper* in proportion to
//! what was dropped.
//!
//! Column selection costs one lookup per non-zero visited, against a dense
//! `original -> masked` table of `cols()` entries. That table is the only allocation the
//! view makes beyond the index lists themselves.

use super::{apply_centering, SparseMat, SparseMatDense};
use crate::types::SvdFloat;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;
use sprs::{CsMatI, SpIndex};

/// Sentinel for "this column is not in the mask".
///
/// A sentinel rather than `Option<usize>` halves the lookup table and keeps the inner
/// loop branch-light.
const EXCLUDED: usize = usize::MAX;

/// A view exposing a subset of the rows and/or columns of a CSR matrix.
///
/// Both selections keep ascending original order, so masked index `i` is original index
/// `selected_rows()[i]` (resp. `selected_columns()[i]`).
///
/// ```
/// use single_svdlib::{sprs::TriMatI, MaskedCsMat, SparseMat, SvdMat};
///
/// let mut t = TriMatI::<f64, u32>::new((6, 5));
/// for i in 0..6 { for j in 0..5 { t.add_triplet(i, j, (i * 5 + j) as f64); } }
/// let a: SvdMat<f64> = t.to_csr::<u64>();
///
/// // Cells 0, 2, 4 by genes 1, 3 — a 3x2 matrix, without touching `a`.
/// let view = MaskedCsMat::submatrix(&a, Some(&[0, 2, 4]), Some(&[1, 3]));
/// assert_eq!((view.rows(), view.cols()), (3, 2));
/// ```
pub struct MaskedCsMat<'a, T, I = u32, Iptr = u64>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    matrix: &'a CsMatI<T, I, Iptr>,
    /// Selected original row indices, ascending. `None` means every row.
    rows: Option<Vec<usize>>,
    /// Selected original column indices, ascending. `None` means every column.
    cols: Option<Vec<usize>>,
    /// Original column -> masked column. Empty when no column mask is in force.
    col_to_masked: Vec<usize>,
    nnz: usize,
}

impl<'a, T, I, Iptr> MaskedCsMat<'a, T, I, Iptr>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    /// A view of the given rows and/or columns. `None` keeps that axis whole.
    ///
    /// Index lists may be in any order and may repeat; the view presents each selected
    /// index once, ascending.
    ///
    /// # Panics
    /// If any index is out of bounds, or the matrix is not CSR.
    pub fn submatrix(
        matrix: &'a CsMatI<T, I, Iptr>,
        rows: Option<&[usize]>,
        cols: Option<&[usize]>,
    ) -> Self {
        assert!(matrix.is_csr(), "MaskedCsMat requires a CSR matrix");

        let rows = rows.map(|r| Self::normalise(r, matrix.rows(), "row"));
        let cols = cols.map(|c| Self::normalise(c, matrix.cols(), "column"));

        let col_to_masked = match &cols {
            Some(c) => {
                let mut table = vec![EXCLUDED; matrix.cols()];
                for (masked, &orig) in c.iter().enumerate() {
                    table[orig] = masked;
                }
                table
            }
            None => Vec::new(),
        };

        let mut view = Self {
            matrix,
            rows,
            cols,
            col_to_masked,
            nnz: 0,
        };
        view.nnz = view.count_nnz();
        view
    }

    /// A view of the given columns, every row.
    pub fn with_columns(matrix: &'a CsMatI<T, I, Iptr>, columns: &[usize]) -> Self {
        Self::submatrix(matrix, None, Some(columns))
    }

    /// A view of the given rows, every column.
    pub fn with_rows(matrix: &'a CsMatI<T, I, Iptr>, rows: &[usize]) -> Self {
        Self::submatrix(matrix, Some(rows), None)
    }

    /// A view built from boolean masks, one entry per original row/column.
    ///
    /// # Panics
    /// If a mask's length does not match the corresponding dimension.
    pub fn from_masks(
        matrix: &'a CsMatI<T, I, Iptr>,
        row_mask: Option<&[bool]>,
        col_mask: Option<&[bool]>,
    ) -> Self {
        let to_indices = |mask: &[bool], n: usize, what: &str| {
            assert_eq!(
                mask.len(),
                n,
                "{what} mask has length {} but the matrix has {n} {what}s",
                mask.len()
            );
            mask.iter()
                .enumerate()
                .filter_map(|(i, &keep)| keep.then_some(i))
                .collect::<Vec<_>>()
        };
        let r = row_mask.map(|m| to_indices(m, matrix.rows(), "row"));
        let c = col_mask.map(|m| to_indices(m, matrix.cols(), "column"));
        Self::submatrix(matrix, r.as_deref(), c.as_deref())
    }

    /// Sort, deduplicate and bounds-check a selection.
    fn normalise(indices: &[usize], limit: usize, what: &str) -> Vec<usize> {
        let mut v = indices.to_vec();
        v.sort_unstable();
        v.dedup();
        if let Some(&last) = v.last() {
            assert!(last < limit, "{what} index {last} is out of bounds ({limit})");
        }
        v
    }

    fn count_nnz(&self) -> usize {
        let masked = !self.col_to_masked.is_empty();
        (0..self.rows_len())
            .into_par_iter()
            .map(|i| {
                let orig = self.row_of(i);
                self.matrix.outer_view(orig).map_or(0, |row| {
                    if masked {
                        row.indices()
                            .iter()
                            .filter(|j| self.col_to_masked[j.index()] != EXCLUDED)
                            .count()
                    } else {
                        row.nnz()
                    }
                })
            })
            .sum()
    }

    #[inline]
    fn rows_len(&self) -> usize {
        self.rows.as_ref().map_or(self.matrix.rows(), |r| r.len())
    }

    /// Original row index for masked row `i`.
    #[inline]
    fn row_of(&self, i: usize) -> usize {
        match &self.rows {
            Some(r) => r[i],
            None => i,
        }
    }

    /// Masked column index for original column `j`, or [`EXCLUDED`].
    #[inline]
    fn masked_col(&self, j: usize) -> usize {
        if self.col_to_masked.is_empty() {
            j
        } else {
            self.col_to_masked[j]
        }
    }

    /// Selected original row indices, ascending. Empty slice when every row is kept.
    pub fn selected_rows(&self) -> Option<&[usize]> {
        self.rows.as_deref()
    }

    /// Selected original column indices, ascending. `None` when every column is kept.
    pub fn selected_columns(&self) -> Option<&[usize]> {
        self.cols.as_deref()
    }

    /// Whether the view is the identity, in which case products delegate straight to
    /// the underlying matrix.
    pub fn is_identity(&self) -> bool {
        self.rows.is_none() && self.cols.is_none()
    }

    /// Whether every column is retained.
    pub fn uses_all_columns(&self) -> bool {
        self.cols.is_none()
    }

    /// The matrix being viewed.
    pub fn inner(&self) -> &'a CsMatI<T, I, Iptr> {
        self.matrix
    }

    /// Materialise the view as an owned sparse matrix.
    ///
    /// Still sparse — this is a subset copy, not a densification — and the source is not
    /// modified.
    ///
    /// # When this is worth doing
    ///
    /// A view does not make products cheaper on the masked axis: every product still
    /// walks *all* the source's non-zeros and tests each against the column table. A view
    /// that keeps a small fraction of the columns therefore scans far more than it uses.
    ///
    /// An iterative solver issues hundreds of products, so if the mask is restrictive the
    /// one-off `O(nnz)` extraction is repaid almost immediately — extracting first is
    /// usually much faster. Prefer the view when the mask keeps most columns, when the
    /// copy would not fit alongside the source, or when only a handful of products are
    /// needed.
    ///
    /// # Panics
    /// If the extracted index range would overflow the index type `I`.
    pub fn to_sparse(&self) -> CsMatI<T, I, Iptr> {
        let (m, n) = (self.rows(), self.cols());
        let mut indptr: Vec<Iptr> = Vec::with_capacity(m + 1);
        let mut indices: Vec<I> = Vec::with_capacity(self.nnz);
        let mut data: Vec<T> = Vec::with_capacity(self.nnz);

        indptr.push(Iptr::from_usize(0));
        for i in 0..m {
            if let Some(row) = self.matrix.outer_view(self.row_of(i)) {
                for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                    let c = self.masked_col(j.index());
                    if c != EXCLUDED {
                        // Source indices ascend and the column table is order-preserving,
                        // so the extracted indices ascend too — CSR's invariant holds
                        // without a sort.
                        indices.push(I::from_usize(c));
                        data.push(v);
                    }
                }
            }
            indptr.push(Iptr::from_usize(indices.len()));
        }

        CsMatI::new((m, n), indptr, indices, data)
    }
}

impl<T, I, Iptr> SparseMat<T> for MaskedCsMat<'_, T, I, Iptr>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn rows(&self) -> usize {
        self.rows_len()
    }
    fn cols(&self) -> usize {
        self.cols.as_ref().map_or(self.matrix.cols(), |c| c.len())
    }
    fn nnz(&self) -> usize {
        self.nnz
    }

    fn mul_vec(&self, x: &[T], y: &mut [T], trans: bool) {
        // 1.x delegated to the *unmasked* matrix whenever the matrix was small,
        // regardless of the mask, which fed a masked-width vector to a full-width
        // product. Delegation is only ever valid when the view is the identity.
        if self.is_identity() {
            return SparseMat::mul_vec(self.matrix, x, y, trans);
        }

        let (m, n) = (self.rows(), self.cols());
        if trans {
            assert_eq!(x.len(), m, "mul_vec: x must have length rows()");
            assert_eq!(y.len(), n, "mul_vec: y must have length cols()");
            y.fill(T::zero());

            let p = rayon::current_num_threads().max(1);
            if p == 1 || self.nnz <= (8 << 10) {
                for i in 0..m {
                    let xi = x[i];
                    if xi.is_zero() {
                        continue;
                    }
                    let Some(row) = self.matrix.outer_view(self.row_of(i)) else {
                        continue;
                    };
                    for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                        let c = self.masked_col(j.index());
                        if c != EXCLUDED {
                            y[c] += v * xi;
                        }
                    }
                }
                return;
            }
            // Scatter direction: one accumulator per thread over the masked width.
            let chunk = m.div_ceil(p).max(1);
            let partials: Vec<Vec<T>> = (0..p)
                .into_par_iter()
                .map(|t| {
                    let lo = (t * chunk).min(m);
                    let hi = ((t + 1) * chunk).min(m);
                    let mut acc = vec![T::zero(); n];
                    for i in lo..hi {
                        let xi = x[i];
                        if xi.is_zero() {
                            continue;
                        }
                        let Some(row) = self.matrix.outer_view(self.row_of(i)) else {
                            continue;
                        };
                        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                            let c = self.masked_col(j.index());
                            if c != EXCLUDED {
                                acc[c] += v * xi;
                            }
                        }
                    }
                    acc
                })
                .collect();
            for acc in &partials {
                for (yi, &a) in y.iter_mut().zip(acc.iter()) {
                    *yi += a;
                }
            }
        } else {
            assert_eq!(x.len(), n, "mul_vec: x must have length cols()");
            assert_eq!(y.len(), m, "mul_vec: y must have length rows()");
            // Gather direction: each output entry is one row's dot product, so masked
            // rows are simply never visited.
            let dot = |i: usize| -> T {
                let Some(row) = self.matrix.outer_view(self.row_of(i)) else {
                    return T::zero();
                };
                let mut sum = T::zero();
                for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                    let c = self.masked_col(j.index());
                    if c != EXCLUDED {
                        sum += v * x[c];
                    }
                }
                sum
            };
            let chunk = m.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
            y.par_chunks_mut(chunk).enumerate().for_each(|(ci, blk)| {
                let base = ci * chunk;
                for (local, yi) in blk.iter_mut().enumerate() {
                    *yi = dot(base + local);
                }
            });
        }
    }
}

impl<T, I, Iptr> SparseMatDense<T> for MaskedCsMat<'_, T, I, Iptr>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn mul_dense(&self, rhs: ArrayView2<T>, mut out: ArrayViewMut2<T>, trans: bool) {
        if self.is_identity() {
            return SparseMatDense::mul_dense(self.matrix, rhs, out, trans);
        }

        let (m, n, k) = (self.rows(), self.cols(), rhs.ncols());
        if trans {
            assert_eq!(rhs.nrows(), m, "mul_dense: rhs.rows != rows()");
            assert_eq!(out.nrows(), n, "mul_dense: out.rows != cols()");
        } else {
            assert_eq!(rhs.nrows(), n, "mul_dense: rhs.rows != cols()");
            assert_eq!(out.nrows(), m, "mul_dense: out.rows != rows()");
        }
        assert_eq!(out.ncols(), k, "mul_dense: out.cols != rhs.cols");

        if !trans {
            // Write-disjoint over output rows.
            let chunk = m.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
            out.axis_chunks_iter_mut(Axis(0), chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(ci, mut block)| {
                    let base = ci * chunk;
                    for (local, mut orow) in block.rows_mut().into_iter().enumerate() {
                        orow.fill(T::zero());
                        let Some(row) = self.matrix.outer_view(self.row_of(base + local))
                        else {
                            continue;
                        };
                        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                            let c = self.masked_col(j.index());
                            if c == EXCLUDED {
                                continue;
                            }
                            let rrow = rhs.row(c);
                            for (o, &r) in orow.iter_mut().zip(rrow.iter()) {
                                *o += v * r;
                            }
                        }
                    }
                });
        } else {
            // Scatter direction, one accumulator per thread over the masked width.
            out.fill(T::zero());
            let p = rayon::current_num_threads().max(1);
            let chunk = m.div_ceil(p).max(1);
            let partials: Vec<ndarray::Array2<T>> = (0..p)
                .into_par_iter()
                .map(|t| {
                    let lo = (t * chunk).min(m);
                    let hi = ((t + 1) * chunk).min(m);
                    let mut acc = ndarray::Array2::<T>::zeros((n, k));
                    for i in lo..hi {
                        let Some(row) = self.matrix.outer_view(self.row_of(i)) else {
                            continue;
                        };
                        let rrow = rhs.row(i);
                        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
                            let c = self.masked_col(j.index());
                            if c == EXCLUDED {
                                continue;
                            }
                            let mut arow = acc.row_mut(c);
                            for (a, &r) in arow.iter_mut().zip(rrow.iter()) {
                                *a += v * r;
                            }
                        }
                    }
                    acc
                })
                .collect();
            for acc in &partials {
                for (mut orow, arow) in out.rows_mut().into_iter().zip(acc.rows()) {
                    for (o, &a) in orow.iter_mut().zip(arow.iter()) {
                        *o += a;
                    }
                }
            }
        }
    }

    /// Column means **of the view** — averaged over the selected rows only, so PCA on a
    /// row subset centers on that subset's means rather than the whole matrix's.
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
            "mul_dense_centered: means must have length cols() (the masked width)"
        );
        self.mul_dense(rhs, out.view_mut(), trans);
        apply_centering(rhs, out, trans, means);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{dense_of, gen_sparse};
    use ndarray::{Array2, Axis};
    use sprs::TriMatI;

    fn sample() -> CsMatI<f64, u32, u64> {
        // 3 x 5
        // [1 0 2 0 3]
        // [0 4 0 5 0]
        // [6 0 7 0 8]
        let mut t = TriMatI::<f64, u32>::new((3, 5));
        for &(i, j, v) in &[
            (0, 0, 1.0),
            (0, 2, 2.0),
            (0, 4, 3.0),
            (1, 1, 4.0),
            (1, 3, 5.0),
            (2, 0, 6.0),
            (2, 2, 7.0),
            (2, 4, 8.0),
        ] {
            t.add_triplet(i, j, v);
        }
        t.to_csr::<u64>()
    }

    /// The dense submatrix a view is supposed to emulate.
    fn physical(m: &CsMatI<f64, u32, u64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
        let full = dense_of(m);
        let mut d = Array2::zeros((rows.len(), cols.len()));
        for (ri, &r) in rows.iter().enumerate() {
            for (ci, &c) in cols.iter().enumerate() {
                d[[ri, ci]] = full[[r, c]];
            }
        }
        d
    }

    fn check_against_physical(view: &MaskedCsMat<f64>, want: &Array2<f64>) {
        assert_eq!((view.rows(), view.cols()), want.dim(), "shape");

        let x: Vec<f64> = (0..view.cols()).map(|i| (i % 5) as f64 - 2.0).collect();
        let mut y = vec![0.0; view.rows()];
        view.mul_vec(&x, &mut y, false);
        let expect = want.dot(&ndarray::Array1::from_vec(x.clone()));
        for (g, w) in y.iter().zip(expect.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
        }

        let xt: Vec<f64> = (0..view.rows()).map(|i| (i % 3) as f64 - 1.0).collect();
        let mut yt = vec![0.0; view.cols()];
        view.mul_vec(&xt, &mut yt, true);
        let expect_t = want.t().dot(&ndarray::Array1::from_vec(xt.clone()));
        for (g, w) in yt.iter().zip(expect_t.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
        }

        // Blocked products, both directions.
        let rhs = Array2::from_shape_fn((view.cols(), 2), |(i, j)| (i + 2 * j) as f64 - 1.0);
        let mut out = Array2::zeros((view.rows(), 2));
        view.mul_dense(rhs.view(), out.view_mut(), false);
        let want_out = want.dot(&rhs);
        for (g, w) in out.iter().zip(want_out.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
        }

        let rhs_t = Array2::from_shape_fn((view.rows(), 2), |(i, j)| (2 * i + j) as f64 - 2.0);
        let mut out_t = Array2::zeros((view.cols(), 2));
        view.mul_dense(rhs_t.view(), out_t.view_mut(), true);
        let want_t = want.t().dot(&rhs_t);
        for (g, w) in out_t.iter().zip(want_t.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    #[test]
    fn column_selection_matches_physical_subset() {
        let a = sample();
        let cols = [0usize, 2, 4];
        let all_rows: Vec<usize> = (0..3).collect();
        check_against_physical(
            &MaskedCsMat::with_columns(&a, &cols),
            &physical(&a, &all_rows, &cols),
        );
    }

    #[test]
    fn row_selection_matches_physical_subset() {
        let a = sample();
        let rows = [0usize, 2];
        let all_cols: Vec<usize> = (0..5).collect();
        let view = MaskedCsMat::with_rows(&a, &rows);
        assert_eq!(view.nnz(), 6, "only the two selected rows' non-zeros count");
        check_against_physical(&view, &physical(&a, &rows, &all_cols));
    }

    #[test]
    fn combined_selection_matches_physical_subset() {
        let a = sample();
        let rows = [0usize, 2];
        let cols = [1usize, 2, 4];
        let view = MaskedCsMat::submatrix(&a, Some(&rows), Some(&cols));
        assert_eq!((view.rows(), view.cols()), (2, 3));
        check_against_physical(&view, &physical(&a, &rows, &cols));
    }

    #[test]
    fn selections_are_sorted_and_deduplicated() {
        let a = sample();
        let view = MaskedCsMat::submatrix(&a, Some(&[2, 0, 2]), Some(&[4, 0, 4, 0]));
        assert_eq!(view.selected_rows().unwrap(), &[0, 2]);
        assert_eq!(view.selected_columns().unwrap(), &[0, 4]);
        check_against_physical(&view, &physical(&a, &[0, 2], &[0, 4]));
    }

    #[test]
    fn boolean_masks_agree_with_index_lists() {
        let a = sample();
        let by_mask = MaskedCsMat::from_masks(
            &a,
            Some(&[true, false, true]),
            Some(&[false, true, false, true, false]),
        );
        let by_index = MaskedCsMat::submatrix(&a, Some(&[0, 2]), Some(&[1, 3]));
        assert_eq!(by_mask.selected_rows(), by_index.selected_rows());
        assert_eq!(by_mask.selected_columns(), by_index.selected_columns());
        assert_eq!(by_mask.nnz(), by_index.nnz());
    }

    /// Regression for the 1.x fast path: a *small* masked matrix used to delegate to the
    /// unmasked product and panic on the length assert.
    #[test]
    fn small_masked_matrix_does_not_delegate() {
        let a = sample();
        let cols = [0usize, 2, 4];
        let view = MaskedCsMat::with_columns(&a, &cols);
        let want = physical(&a, &[0, 1, 2], &cols);
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        view.mul_vec(&x, &mut y, false);
        assert_eq!(y, want.dot(&ndarray::arr1(&x)).to_vec());
    }

    #[test]
    fn identity_view_matches_unmasked() {
        let a = sample();
        let view = MaskedCsMat::submatrix(&a, None, None);
        assert!(view.is_identity());
        assert_eq!(view.nnz(), a.nnz());
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut ym = vec![0.0; 3];
        let mut yu = vec![0.0; 3];
        view.mul_vec(&x, &mut ym, false);
        SparseMat::mul_vec(&a, &x, &mut yu, false);
        assert_eq!(ym, yu);
    }

    /// Means must be taken over the *selected* rows, so PCA on a row subset centers on
    /// that subset.
    #[test]
    fn col_means_respect_the_row_selection() {
        let a = sample();
        let rows = [0usize, 2];
        let cols = [0usize, 2, 4];
        let view = MaskedCsMat::submatrix(&a, Some(&rows), Some(&cols));
        let want = physical(&a, &rows, &cols);

        let got = view.col_means();
        let expect = want.mean_axis(Axis(0)).unwrap();
        for (g, w) in got.iter().zip(expect.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12);
        }

        // And centering must match explicitly centering that submatrix.
        let centered = &want - &expect.view().insert_axis(Axis(0));
        let rhs = ndarray::arr2(&[[1.0], [2.0], [3.0]]);
        let mut out = Array2::zeros((2, 1));
        view.mul_dense_centered(rhs.view(), out.view_mut(), false, got.view());
        let want_out = centered.dot(&rhs);
        for (g, w) in out.iter().zip(want_out.iter()) {
            approx::assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    /// The extracted submatrix must be indistinguishable from the view.
    #[test]
    fn to_sparse_matches_the_view() {
        let a = gen_sparse(120, 40, 0.1, 23);
        let rows: Vec<usize> = (0..120).filter(|r| r % 4 != 0).collect();
        let cols: Vec<usize> = (0..40).filter(|c| c % 3 == 0).collect();
        let view = MaskedCsMat::submatrix(&a, Some(&rows), Some(&cols));
        let extracted = view.to_sparse();

        assert_eq!(extracted.rows(), view.rows());
        assert_eq!(extracted.cols(), view.cols());
        assert_eq!(extracted.nnz(), view.nnz());
        assert!(extracted.is_csr());
        // Extraction must preserve CSR's ascending-index invariant.
        for i in 0..extracted.rows() {
            if let Some(row) = extracted.outer_view(i) {
                let idx = row.indices();
                assert!(idx.windows(2).all(|w| w[0] < w[1]), "row {i} indices not sorted");
            }
        }

        assert_eq!(dense_of(&extracted), physical(&a, &rows, &cols));

        // And every product agrees.
        let x: Vec<f64> = (0..view.cols()).map(|i| (i % 7) as f64 - 3.0).collect();
        let (mut yv, mut ye) = (vec![0.0; view.rows()], vec![0.0; view.rows()]);
        view.mul_vec(&x, &mut yv, false);
        SparseMat::mul_vec(&extracted, &x, &mut ye, false);
        assert_eq!(yv, ye);
    }

    /// A decomposition must not care which representation it was handed.
    #[test]
    fn pca_agrees_between_view_and_extraction() {
        let a = gen_sparse(300, 50, 0.12, 29);
        let cols: Vec<usize> = (0..50).filter(|c| c % 2 == 0).collect();
        let view = MaskedCsMat::with_columns(&a, &cols);
        let extracted = view.to_sparse();

        let by_view = crate::irlba::svd_centered(&view, 6, Some(42)).unwrap();
        let by_copy = crate::irlba::svd_centered(&extracted, 6, Some(42)).unwrap();
        for (x, y) in by_view.s.iter().zip(by_copy.s.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-9);
        }
    }

    #[test]
    fn empty_selections() {
        let a = sample();
        let no_cols = MaskedCsMat::submatrix(&a, None, Some(&[]));
        assert_eq!(no_cols.cols(), 0);
        assert_eq!(no_cols.nnz(), 0);

        let no_rows = MaskedCsMat::submatrix(&a, Some(&[]), None);
        assert_eq!(no_rows.rows(), 0);
        assert_eq!(no_rows.nnz(), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn rejects_out_of_range_rows() {
        let a = sample();
        let _ = MaskedCsMat::with_rows(&a, &[0, 99]);
    }

    /// Masking rows must not change the answer relative to physically extracting them.
    #[test]
    fn larger_random_submatrix() {
        let a = gen_sparse(200, 60, 0.08, 17);
        let rows: Vec<usize> = (0..200).filter(|r| r % 3 == 0).collect();
        let cols: Vec<usize> = (0..60).filter(|c| c % 2 == 1).collect();
        let view = MaskedCsMat::submatrix(&a, Some(&rows), Some(&cols));
        check_against_physical(&view, &physical(&a, &rows, &cols));
    }
}
