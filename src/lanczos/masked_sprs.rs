use crate::utils::determine_chunk_size;
use crate::SMat;
use nalgebra::{DMatrix, DVector};
use num_traits::{Float, FromPrimitive, Zero};
use rayon::prelude::*;
use sprs::{CsMatI, SpIndex};
use std::fmt::Debug;
use std::ops::{AddAssign, SubAssign};

/// A view of a `CsMatI` that exposes only a selected subset of columns,
/// without copying the underlying data.
///
/// The masked (virtual) column indices run `0..ncols()` where `ncols()` is
/// the number of selected columns.
pub struct MaskedCsMatI<'a, T, I, Iptr>
where
    T: Float,
    I: SpIndex,
    Iptr: SpIndex,
{
    matrix: &'a CsMatI<T, I, Iptr>,
    /// masked_col → original_col
    masked_to_original: Vec<usize>,
    /// original_col → masked_col (None = excluded)
    original_to_masked: Vec<Option<usize>>,
}

/// Convenience alias using `usize` index types.
pub type MaskedCsMat<'a, T> = MaskedCsMatI<'a, T, usize, usize>;

impl<'a, T, I, Iptr> MaskedCsMatI<'a, T, I, Iptr>
where
    T: Float,
    I: SpIndex,
    Iptr: SpIndex,
{
    /// Build a masked view from a boolean column mask.
    pub fn new(matrix: &'a CsMatI<T, I, Iptr>, column_mask: &[bool]) -> Self {
        assert_eq!(
            column_mask.len(),
            matrix.cols(),
            "column_mask length must equal matrix column count"
        );
        let mut masked_to_original = Vec::new();
        let mut original_to_masked = vec![None; column_mask.len()];
        for (i, &included) in column_mask.iter().enumerate() {
            if included {
                original_to_masked[i] = Some(masked_to_original.len());
                masked_to_original.push(i);
            }
        }
        Self {
            matrix,
            masked_to_original,
            original_to_masked,
        }
    }

    /// Build a masked view from an explicit list of column indices to include.
    pub fn with_columns(matrix: &'a CsMatI<T, I, Iptr>, columns: &[usize]) -> Self {
        let mut mask = vec![false; matrix.cols()];
        for &col in columns {
            assert!(col < matrix.cols(), "column index {col} out of bounds");
            mask[col] = true;
        }
        Self::new(matrix, &mask)
    }
}

impl<T, I, Iptr> SMat<T> for MaskedCsMatI<'_, T, I, Iptr>
where
    T: Float + Zero + AddAssign + SubAssign + Copy + Sync + Send + FromPrimitive + Debug + 'static,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn nrows(&self) -> usize {
        self.matrix.rows()
    }

    fn ncols(&self) -> usize {
        self.masked_to_original.len()
    }

    fn nnz(&self) -> usize {
        self.matrix
            .iter()
            .filter(|(_, (_, j))| self.original_to_masked[j.index()].is_some())
            .count()
    }

    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = self.matrix.rows();
        let masked_ncols = self.masked_to_original.len();
        let (x_len, y_len) = if transposed {
            (nrows, masked_ncols)
        } else {
            (masked_ncols, nrows)
        };
        assert_eq!(
            x.len(),
            x_len,
            "svd_opa: x length mismatch: x={}, expected={}",
            x.len(),
            x_len
        );
        assert_eq!(
            y.len(),
            y_len,
            "svd_opa: y length mismatch: y={}, expected={}",
            y.len(),
            y_len
        );
        y.fill(T::zero());

        if self.matrix.is_csr() {
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            if !transposed {
                // y[i] = sum_{j in mask} A[i,j] * x[masked_j]  — gather per row, parallel
                let results: Vec<(usize, T)> = (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        let sum =
                            (indptr[i].index()..indptr[i + 1].index()).fold(T::zero(), |acc, k| {
                                let j = indices[k].index();
                                match self.original_to_masked[j] {
                                    Some(mj) => acc + data[k] * x[mj],
                                    None => acc,
                                }
                            });
                        (i, sum)
                    })
                    .collect();
                for (i, v) in results {
                    y[i] = v;
                }
            } else {
                // y[masked_j] += A[i,j] * x[i]  — scatter, parallel chunks
                let chunk_size = determine_chunk_size(nrows);
                let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let row_start = chunk_idx * chunk_size;
                        let row_end = (row_start + chunk_size).min(nrows);
                        let mut local = vec![T::zero(); masked_ncols];
                        for i in row_start..row_end {
                            let xi = x[i];
                            for k in indptr[i].index()..indptr[i + 1].index() {
                                let j = indices[k].index();
                                if let Some(mj) = self.original_to_masked[j] {
                                    local[mj] += data[k] * xi;
                                }
                            }
                        }
                        local
                    })
                    .collect();
                for local in partials {
                    for (mj, &v) in local.iter().enumerate() {
                        if !v.is_zero() {
                            y[mj] += v;
                        }
                    }
                }
            }
        } else {
            // CSC: outer dimension = column
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();
            let ncols_orig = self.matrix.cols();

            if !transposed {
                // scatter: for each included original col j, add A[:,j]*x[mj] into y
                let chunk_size = determine_chunk_size(ncols_orig);
                let partials: Vec<Vec<T>> = (0..ncols_orig.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let col_start = chunk_idx * chunk_size;
                        let col_end = (col_start + chunk_size).min(ncols_orig);
                        let mut local = vec![T::zero(); nrows];
                        for j in col_start..col_end {
                            if let Some(mj) = self.original_to_masked[j] {
                                let xmj = x[mj];
                                for k in indptr[j].index()..indptr[j + 1].index() {
                                    local[indices[k].index()] += data[k] * xmj;
                                }
                            }
                        }
                        local
                    })
                    .collect();
                for local in partials {
                    for (i, &v) in local.iter().enumerate() {
                        if !v.is_zero() {
                            y[i] += v;
                        }
                    }
                }
            } else {
                // gather per masked col — parallel
                let results: Vec<(usize, T)> = self
                    .masked_to_original
                    .par_iter()
                    .enumerate()
                    .map(|(mj, &j)| {
                        let sum = (indptr[j].index()..indptr[j + 1].index())
                            .fold(T::zero(), |acc, k| acc + data[k] * x[indices[k].index()]);
                        (mj, sum)
                    })
                    .collect();
                for (mj, v) in results {
                    y[mj] = v;
                }
            }
        }
    }

    fn compute_column_means(&self) -> Vec<T> {
        let nrows = self.matrix.rows();
        let masked_ncols = self.masked_to_original.len();
        let recip = T::from(nrows).unwrap().recip();

        if self.matrix.is_csr() {
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            let mut sums = vec![T::zero(); masked_ncols];
            for i in 0..nrows {
                for k in indptr[i].index()..indptr[i + 1].index() {
                    let j = indices[k].index();
                    if let Some(mj) = self.original_to_masked[j] {
                        sums[mj] += data[k];
                    }
                }
            }
            sums.iter_mut().for_each(|v| *v = *v * recip);
            sums
        } else {
            // CSC: one col per task — parallel
            self.masked_to_original
                .par_iter()
                .map(|&j| {
                    let indptr_view = self.matrix.indptr();
                    let indptr = indptr_view.raw_storage();
                    let data = self.matrix.data();
                    let sum = (indptr[j].index()..indptr[j + 1].index())
                        .fold(T::zero(), |acc, k| acc + data[k]);
                    sum * recip
                })
                .collect()
        }
    }

    fn multiply_with_dense(
        &self,
        dense: &DMatrix<T>,
        result: &mut DMatrix<T>,
        transpose_self: bool,
    ) {
        let nrows = self.matrix.rows();
        let masked_ncols = self.masked_to_original.len();
        let dense_cols = dense.ncols();

        if self.matrix.is_csr() {
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            if !transpose_self {
                // result = A_masked @ dense, shape (nrows, dense_cols)
                let row_results: Vec<(usize, Vec<T>)> = (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = vec![T::zero(); dense_cols];
                        for k in indptr[i].index()..indptr[i + 1].index() {
                            let j = indices[k].index();
                            if let Some(mj) = self.original_to_masked[j] {
                                let v = data[k];
                                for c in 0..dense_cols {
                                    row[c] += v * dense[(mj, c)];
                                }
                            }
                        }
                        (i, row)
                    })
                    .collect();
                for (i, row) in row_results {
                    for c in 0..dense_cols {
                        result[(i, c)] = row[c];
                    }
                }
            } else {
                // result = A_masked^T @ dense, shape (masked_ncols, dense_cols) — scatter
                let chunk_size = determine_chunk_size(nrows);
                let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let row_start = chunk_idx * chunk_size;
                        let row_end = (row_start + chunk_size).min(nrows);
                        let mut local = vec![T::zero(); masked_ncols * dense_cols];
                        for i in row_start..row_end {
                            for k in indptr[i].index()..indptr[i + 1].index() {
                                let j = indices[k].index();
                                if let Some(mj) = self.original_to_masked[j] {
                                    let v = data[k];
                                    for c in 0..dense_cols {
                                        local[mj * dense_cols + c] += v * dense[(i, c)];
                                    }
                                }
                            }
                        }
                        local
                    })
                    .collect();
                result.fill(T::zero());
                for local in partials {
                    for mj in 0..masked_ncols {
                        for c in 0..dense_cols {
                            result[(mj, c)] += local[mj * dense_cols + c];
                        }
                    }
                }
            }
        } else {
            // CSC path
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            if !transpose_self {
                // scatter over masked cols
                let chunk_size = determine_chunk_size(masked_ncols);
                let partials: Vec<Vec<T>> = (0..masked_ncols.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(masked_ncols);
                        let mut local = vec![T::zero(); nrows * dense_cols];
                        for mj in start..end {
                            let j = self.masked_to_original[mj];
                            for k in indptr[j].index()..indptr[j + 1].index() {
                                let i = indices[k].index();
                                let v = data[k];
                                for c in 0..dense_cols {
                                    local[i * dense_cols + c] += v * dense[(mj, c)];
                                }
                            }
                        }
                        local
                    })
                    .collect();
                result.fill(T::zero());
                for local in partials {
                    for i in 0..nrows {
                        for c in 0..dense_cols {
                            result[(i, c)] += local[i * dense_cols + c];
                        }
                    }
                }
            } else {
                // gather per masked col — parallel
                let col_results: Vec<(usize, Vec<T>)> = (0..masked_ncols)
                    .into_par_iter()
                    .map(|mj| {
                        let j = self.masked_to_original[mj];
                        let mut row = vec![T::zero(); dense_cols];
                        for k in indptr[j].index()..indptr[j + 1].index() {
                            let i = indices[k].index();
                            let v = data[k];
                            for c in 0..dense_cols {
                                row[c] += v * dense[(i, c)];
                            }
                        }
                        (mj, row)
                    })
                    .collect();
                for (mj, row) in col_results {
                    for c in 0..dense_cols {
                        result[(mj, c)] = row[c];
                    }
                }
            }
        }
    }

    fn multiply_with_dense_centered(
        &self,
        dense: &DMatrix<T>,
        result: &mut DMatrix<T>,
        transpose_self: bool,
        means: &DVector<T>,
    ) {
        let dense_cols = dense.ncols();
        if !transpose_self {
            // result = (A_masked - 1·means^T) @ dense
            // correction[c] = sum_mj means[mj] * dense[mj,c]
            let masked_ncols = self.masked_to_original.len();
            let correction: Vec<T> = (0..dense_cols)
                .map(|c| {
                    (0..masked_ncols).fold(T::zero(), |acc, mj| acc + means[mj] * dense[(mj, c)])
                })
                .collect();
            self.multiply_with_dense(dense, result, false);
            let nrows = self.matrix.rows();
            for i in 0..nrows {
                for c in 0..dense_cols {
                    result[(i, c)] -= correction[c];
                }
            }
        } else {
            // result = (A_masked^T - means·1^T) @ dense
            // result[mj,c] -= means[mj] * col_sums_dense[c]
            let nrows = self.matrix.rows();
            let masked_ncols = self.masked_to_original.len();
            let col_sums: Vec<T> = (0..dense_cols)
                .map(|c| (0..nrows).fold(T::zero(), |acc, i| acc + dense[(i, c)]))
                .collect();
            self.multiply_with_dense(dense, result, true);
            for mj in 0..masked_ncols {
                let m = means[mj];
                for c in 0..dense_cols {
                    result[(mj, c)] -= m * col_sums[c];
                }
            }
        }
    }

    fn multiply_transposed_by_dense(&self, q: &DMatrix<T>, result: &mut DMatrix<T>) {
        // result = Q^T @ A_masked, shape (q.ncols, masked_ncols)
        let nrows = self.matrix.rows();
        let masked_ncols = self.masked_to_original.len();
        let q_cols = q.ncols();

        if self.matrix.is_csr() {
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            let chunk_size = determine_chunk_size(nrows);
            let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk_idx| {
                    let row_start = chunk_idx * chunk_size;
                    let row_end = (row_start + chunk_size).min(nrows);
                    let mut local = vec![T::zero(); q_cols * masked_ncols];
                    for i in row_start..row_end {
                        for k in indptr[i].index()..indptr[i + 1].index() {
                            let j = indices[k].index();
                            if let Some(mj) = self.original_to_masked[j] {
                                let v = data[k];
                                for c in 0..q_cols {
                                    local[c * masked_ncols + mj] += q[(i, c)] * v;
                                }
                            }
                        }
                    }
                    local
                })
                .collect();
            result.fill(T::zero());
            for local in partials {
                for c in 0..q_cols {
                    for mj in 0..masked_ncols {
                        result[(c, mj)] += local[c * masked_ncols + mj];
                    }
                }
            }
        } else {
            // CSC: gather per masked col — parallel
            let indptr_view = self.matrix.indptr();
            let indptr = indptr_view.raw_storage();
            let indices = self.matrix.indices();
            let data = self.matrix.data();

            let col_results: Vec<(usize, Vec<T>)> = (0..masked_ncols)
                .into_par_iter()
                .map(|mj| {
                    let j = self.masked_to_original[mj];
                    let mut col = vec![T::zero(); q_cols];
                    for k in indptr[j].index()..indptr[j + 1].index() {
                        let i = indices[k].index();
                        let v = data[k];
                        for c in 0..q_cols {
                            col[c] += q[(i, c)] * v;
                        }
                    }
                    (mj, col)
                })
                .collect();
            result.fill(T::zero());
            for (mj, col) in col_results {
                for c in 0..q_cols {
                    result[(c, mj)] = col[c];
                }
            }
        }
    }

    fn multiply_transposed_by_dense_centered(
        &self,
        q: &DMatrix<T>,
        result: &mut DMatrix<T>,
        means: &DVector<T>,
    ) {
        // result = Q^T @ (A_masked - 1·means^T)
        //        = Q^T @ A_masked - (sum_i Q[i,:])^T · means^T
        let q_rows = q.nrows();
        let q_cols = q.ncols();
        let masked_ncols = self.masked_to_original.len();
        let q_col_sums: Vec<T> = (0..q_cols)
            .map(|c| (0..q_rows).fold(T::zero(), |acc, i| acc + q[(i, c)]))
            .collect();
        self.multiply_transposed_by_dense(q, result);
        for c in 0..q_cols {
            let qs = q_col_sums[c];
            for mj in 0..masked_ncols {
                result[(c, mj)] -= qs * means[mj];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use sprs::TriMat;

    /// Full test matrix A (3×4):
    ///   1  2  0  0
    ///   0  0  3  4
    ///   0  5  0  6
    ///
    /// Mask: columns [1, 3]  →  masked matrix B (3×2):
    ///   2  0
    ///   0  4
    ///   5  6
    fn full_csr() -> sprs::CsMat<f64> {
        let mut tri: TriMat<f64> = TriMat::new((3, 4));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 2, 3.0);
        tri.add_triplet(1, 3, 4.0);
        tri.add_triplet(2, 1, 5.0);
        tri.add_triplet(2, 3, 6.0);
        tri.to_csr()
    }

    fn full_csc() -> sprs::CsMat<f64> {
        full_csr().to_csc()
    }

    const MASK: &[bool] = &[false, true, false, true]; // columns 1 and 3

    fn assert_slice_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-10, "index {i}: actual={a}, expected={e}");
        }
    }

    fn assert_mat_close(m: &DMatrix<f64>, expected: &[(usize, usize, f64)]) {
        for &(i, j, e) in expected {
            let a = m[(i, j)];
            assert!(
                (a - e).abs() < 1e-10,
                "m[{i},{j}]: actual={a}, expected={e}"
            );
        }
    }

    // --- basic properties ---

    #[test]
    fn test_dimensions_and_nnz() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        assert_eq!(masked.nrows(), 3);
        assert_eq!(masked.ncols(), 2);
        // B has non-zeros at (0,0)=2, (1,1)=4, (2,0)=5, (2,1)=6  → 4
        assert_eq!(masked.nnz(), 4);
    }

    // --- svd_opa ---
    // B forward: x=[1,2] → y[0]=2*1=2, y[1]=4*2=8, y[2]=5*1+6*2=17
    // B transposed: x=[1,2,3] → y[0]=2*1+5*3=17, y[1]=4*2+6*3=26

    #[test]
    fn test_csr_svd_opa_forward() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 3];
        masked.svd_opa(&x, &mut y, false);
        assert_slice_close(&y, &[2.0, 8.0, 17.0]);
    }

    #[test]
    fn test_csr_svd_opa_transposed() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2];
        masked.svd_opa(&x, &mut y, true);
        assert_slice_close(&y, &[17.0, 26.0]);
    }

    #[test]
    fn test_csc_svd_opa_forward() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 3];
        masked.svd_opa(&x, &mut y, false);
        assert_slice_close(&y, &[2.0, 8.0, 17.0]);
    }

    #[test]
    fn test_csc_svd_opa_transposed() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2];
        masked.svd_opa(&x, &mut y, true);
        assert_slice_close(&y, &[17.0, 26.0]);
    }

    // --- compute_column_means ---
    // B col 0 (orig col 1): (2+0+5)/3 = 7/3
    // B col 1 (orig col 3): (0+4+6)/3 = 10/3

    #[test]
    fn test_column_means_csr() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        assert_slice_close(&masked.compute_column_means(), &[7.0 / 3.0, 10.0 / 3.0]);
    }

    #[test]
    fn test_column_means_csc() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        assert_slice_close(&masked.compute_column_means(), &[7.0 / 3.0, 10.0 / 3.0]);
    }

    // --- multiply_with_dense ---
    // D (2×2, row-major) = [[1,2],[3,4]]
    // B@D  row0=[2,4], row1=[12,16], row2=[5*1+6*3, 5*2+6*4]=[23,34]
    // B^T@D (3×2 D=[[1,2],[3,4],[5,6]]):
    //   row0 (orig col1): 2*[1,2]+0*[3,4]+5*[5,6] = [27,34]
    //   row1 (orig col3): 0*[1,2]+4*[3,4]+6*[5,6] = [42,52]

    #[test]
    fn test_multiply_dense_forward_csr() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let dense = DMatrix::from_row_slice(2, 2, &[1.0f64, 2.0, 3.0, 4.0]);
        let mut result = DMatrix::zeros(3, 2);
        masked.multiply_with_dense(&dense, &mut result, false);
        assert_mat_close(
            &result,
            &[
                (0, 0, 2.0),
                (0, 1, 4.0),
                (1, 0, 12.0),
                (1, 1, 16.0),
                (2, 0, 23.0),
                (2, 1, 34.0),
            ],
        );
    }

    #[test]
    fn test_multiply_dense_transposed_csr() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_with_dense(&dense, &mut result, true);
        assert_mat_close(
            &result,
            &[(0, 0, 27.0), (0, 1, 34.0), (1, 0, 42.0), (1, 1, 52.0)],
        );
    }

    #[test]
    fn test_multiply_dense_forward_csc() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        let dense = DMatrix::from_row_slice(2, 2, &[1.0f64, 2.0, 3.0, 4.0]);
        let mut result = DMatrix::zeros(3, 2);
        masked.multiply_with_dense(&dense, &mut result, false);
        assert_mat_close(
            &result,
            &[
                (0, 0, 2.0),
                (0, 1, 4.0),
                (1, 0, 12.0),
                (1, 1, 16.0),
                (2, 0, 23.0),
                (2, 1, 34.0),
            ],
        );
    }

    #[test]
    fn test_multiply_dense_transposed_csc() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_with_dense(&dense, &mut result, true);
        assert_mat_close(
            &result,
            &[(0, 0, 27.0), (0, 1, 34.0), (1, 0, 42.0), (1, 1, 52.0)],
        );
    }

    // --- multiply_with_dense_centered ---
    // means = [0.5, 1.0] for masked cols [1,3]
    // D (2×2) = [[1,2],[3,4]]
    // correction[c] = sum_mj means[mj]*D[mj,c]
    //   correction[0] = 0.5*1 + 1.0*3 = 3.5
    //   correction[1] = 0.5*2 + 1.0*4 = 5.0
    // result = B@D - correction: row0=[2-3.5,4-5]=[-1.5,-1], row1=[12-3.5,16-5]=[8.5,11], row2=[23-3.5,34-5]=[19.5,29]

    #[test]
    fn test_centered_forward() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let means = DVector::from_vec(vec![0.5f64, 1.0]);
        let dense = DMatrix::from_row_slice(2, 2, &[1.0f64, 2.0, 3.0, 4.0]);
        let mut result = DMatrix::zeros(3, 2);
        masked.multiply_with_dense_centered(&dense, &mut result, false, &means);
        assert_mat_close(
            &result,
            &[
                (0, 0, -1.5),
                (0, 1, -1.0),
                (1, 0, 8.5),
                (1, 1, 11.0),
                (2, 0, 19.5),
                (2, 1, 29.0),
            ],
        );
    }

    // D (3×2) = [[1,2],[3,4],[5,6]], col_sums=[9,12]
    // result[mj,c] = (B^T@D)[mj,c] - means[mj]*col_sums[c]
    //   mj=0: [27,34] - 0.5*[9,12] = [22.5,28]
    //   mj=1: [42,52] - 1.0*[9,12] = [33,40]

    #[test]
    fn test_centered_transposed() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let means = DVector::from_vec(vec![0.5f64, 1.0]);
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_with_dense_centered(&dense, &mut result, true, &means);
        assert_mat_close(
            &result,
            &[(0, 0, 22.5), (0, 1, 28.0), (1, 0, 33.0), (1, 1, 40.0)],
        );
    }

    // --- multiply_transposed_by_dense ---
    // Q (3×2, row-major) = [[1,2],[3,4],[5,6]]
    // result = Q^T @ B, shape (2, 2)
    //   [0,0] = Q[:,0]·B[:,0] = [1,3,5]·[2,0,5] = 27
    //   [0,1] = Q[:,0]·B[:,1] = [1,3,5]·[0,4,6] = 42
    //   [1,0] = Q[:,1]·B[:,0] = [2,4,6]·[2,0,5] = 34
    //   [1,1] = Q[:,1]·B[:,1] = [2,4,6]·[0,4,6] = 52

    #[test]
    fn test_transposed_by_dense_csr() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_transposed_by_dense(&q, &mut result);
        assert_mat_close(
            &result,
            &[(0, 0, 27.0), (0, 1, 42.0), (1, 0, 34.0), (1, 1, 52.0)],
        );
    }

    #[test]
    fn test_transposed_by_dense_csc() {
        let csc = full_csc();
        let masked = MaskedCsMatI::new(&csc, MASK);
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_transposed_by_dense(&q, &mut result);
        assert_mat_close(
            &result,
            &[(0, 0, 27.0), (0, 1, 42.0), (1, 0, 34.0), (1, 1, 52.0)],
        );
    }

    // --- multiply_transposed_by_dense_centered ---
    // q_col_sums = [9, 12]
    // result[c,mj] = (Q^T@B)[c,mj] - q_col_sums[c]*means[mj]
    //   [0,0] = 27 - 9*0.5 = 22.5
    //   [0,1] = 42 - 9*1.0 = 33
    //   [1,0] = 34 - 12*0.5 = 28
    //   [1,1] = 52 - 12*1.0 = 40

    #[test]
    fn test_transposed_centered() {
        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);
        let means = DVector::from_vec(vec![0.5f64, 1.0]);
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 2);
        masked.multiply_transposed_by_dense_centered(&q, &mut result, &means);
        assert_mat_close(
            &result,
            &[(0, 0, 22.5), (0, 1, 33.0), (1, 0, 28.0), (1, 1, 40.0)],
        );
    }

    // --- end-to-end SVD comparison ---
    // MaskedCsMatI SVD on B must match SVD on a physical sprs matrix of B

    #[test]
    fn test_svd_matches_physical_subset() {
        // Physical matrix B (3×2):
        //   2  0
        //   0  4
        //   5  6
        let mut tri_b: TriMat<f64> = TriMat::new((3, 2));
        tri_b.add_triplet(0, 0, 2.0);
        tri_b.add_triplet(1, 1, 4.0);
        tri_b.add_triplet(2, 0, 5.0);
        tri_b.add_triplet(2, 1, 6.0);
        let physical: sprs::CsMat<f64> = tri_b.to_csr();

        let csr = full_csr();
        let masked = MaskedCsMatI::new(&csr, MASK);

        let svd_masked = crate::lanczos::svd_dim_seed(&masked, 0, 42).unwrap();
        let svd_physical = crate::lanczos::svd_dim_seed(&physical, 0, 42).unwrap();

        assert_eq!(svd_masked.d, svd_physical.d);
        for i in 0..svd_masked.d {
            assert!(
                (svd_masked.s[i] - svd_physical.s[i]).abs() < 1e-10,
                "singular value {i} differs: masked={}, physical={}",
                svd_masked.s[i],
                svd_physical.s[i]
            );
        }
    }
}
