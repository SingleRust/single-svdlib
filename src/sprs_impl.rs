use crate::utils::determine_chunk_size;
use crate::SMat;
use nalgebra::{DMatrix, DVector};
use num_traits::{Float, FromPrimitive, Zero};
use rayon::prelude::*;
use sprs::{CsMatI, SpIndex};
use std::fmt::Debug;
use std::ops::{AddAssign, SubAssign};

impl<T, I, Iptr> SMat<T> for CsMatI<T, I, Iptr>
where
    T: Float + Zero + AddAssign + SubAssign + Copy + Sync + Send + FromPrimitive + Debug + 'static,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn nrows(&self) -> usize {
        self.rows()
    }

    fn ncols(&self) -> usize {
        self.cols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = self.rows();
        let ncols = self.cols();
        let (x_len, y_len) = if transposed {
            (nrows, ncols)
        } else {
            (ncols, nrows)
        };
        assert_eq!(
            x.len(),
            x_len,
            "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}",
            x.len(),
            x_len
        );
        assert_eq!(
            y.len(),
            y_len,
            "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}",
            y.len(),
            y_len
        );

        y.fill(T::zero());

        let indptr_view = self.indptr();
        let indptr = indptr_view.raw_storage();
        let indices = self.indices();
        let data = self.data();

        if self.is_csr() {
            if !transposed {
                // y[i] = sum_j A[i,j] * x[j]  — gather per row, parallel
                let results: Vec<(usize, T)> = (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        let sum = (indptr[i].index()..indptr[i + 1].index())
                            .fold(T::zero(), |acc, k| acc + data[k] * x[indices[k].index()]);
                        (i, sum)
                    })
                    .collect();
                for (i, v) in results {
                    y[i] = v;
                }
            } else {
                // y[j] += sum_i A[i,j] * x[i]  — scatter, parallel chunks + reduce
                let chunk_size = determine_chunk_size(nrows);
                let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let row_start = chunk_idx * chunk_size;
                        let row_end = (row_start + chunk_size).min(nrows);
                        let mut local = vec![T::zero(); ncols];
                        for i in row_start..row_end {
                            let xi = x[i];
                            for k in indptr[i].index()..indptr[i + 1].index() {
                                local[indices[k].index()] += data[k] * xi;
                            }
                        }
                        local
                    })
                    .collect();
                for local in partials {
                    for (j, &v) in local.iter().enumerate() {
                        if !v.is_zero() {
                            y[j] += v;
                        }
                    }
                }
            }
        } else {
            // CSC: outer = col, inner = row
            if !transposed {
                // y[i] += sum_j A[i,j] * x[j]  — scatter, parallel chunks + reduce
                let chunk_size = determine_chunk_size(ncols);
                let partials: Vec<Vec<T>> = (0..ncols.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let col_start = chunk_idx * chunk_size;
                        let col_end = (col_start + chunk_size).min(ncols);
                        let mut local = vec![T::zero(); nrows];
                        for j in col_start..col_end {
                            let xj = x[j];
                            for k in indptr[j].index()..indptr[j + 1].index() {
                                local[indices[k].index()] += data[k] * xj;
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
                // y[j] = sum_i A[i,j] * x[i]  — gather per col, parallel
                let results: Vec<(usize, T)> = (0..ncols)
                    .into_par_iter()
                    .map(|j| {
                        let sum = (indptr[j].index()..indptr[j + 1].index())
                            .fold(T::zero(), |acc, k| acc + data[k] * x[indices[k].index()]);
                        (j, sum)
                    })
                    .collect();
                for (j, v) in results {
                    y[j] = v;
                }
            }
        }
    }

    fn compute_column_means(&self) -> Vec<T> {
        let nrows = self.rows();
        let ncols = self.cols();
        let recip = T::from(nrows).unwrap().recip();
        let indptr_view = self.indptr();
        let indptr = indptr_view.raw_storage();
        let indices = self.indices();
        let data = self.data();

        if self.is_csr() {
            let mut col_sums = vec![T::zero(); ncols];
            for i in 0..nrows {
                for k in indptr[i].index()..indptr[i + 1].index() {
                    col_sums[indices[k].index()] += data[k];
                }
            }
            col_sums.iter_mut().for_each(|v| *v = *v * recip);
            col_sums
        } else {
            // CSC: each outer slice is a column — parallel over cols
            (0..ncols)
                .into_par_iter()
                .map(|j| {
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
        let nrows = self.rows();
        let ncols = self.cols();
        let dense_cols = dense.ncols();
        let indptr_view = self.indptr();
        let indptr = indptr_view.raw_storage();
        let indices = self.indices();
        let data = self.data();

        if self.is_csr() {
            if !transpose_self {
                // result = A @ dense, shape (nrows, dense_cols) — gather per row
                let row_results: Vec<(usize, Vec<T>)> = (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = vec![T::zero(); dense_cols];
                        for k in indptr[i].index()..indptr[i + 1].index() {
                            let j = indices[k].index();
                            let v = data[k];
                            for c in 0..dense_cols {
                                row[c] += v * dense[(j, c)];
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
                // result = A^T @ dense, shape (ncols, dense_cols) — scatter with reduction
                let chunk_size = determine_chunk_size(nrows);
                let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let row_start = chunk_idx * chunk_size;
                        let row_end = (row_start + chunk_size).min(nrows);
                        let mut local = vec![T::zero(); ncols * dense_cols];
                        for i in row_start..row_end {
                            for k in indptr[i].index()..indptr[i + 1].index() {
                                let j = indices[k].index();
                                let v = data[k];
                                for c in 0..dense_cols {
                                    local[j * dense_cols + c] += v * dense[(i, c)];
                                }
                            }
                        }
                        local
                    })
                    .collect();
                result.fill(T::zero());
                for local in partials {
                    for j in 0..ncols {
                        for c in 0..dense_cols {
                            result[(j, c)] += local[j * dense_cols + c];
                        }
                    }
                }
            }
        } else {
            // CSC: outer = col
            if !transpose_self {
                // result = A @ dense, shape (nrows, dense_cols) — scatter with reduction
                let chunk_size = determine_chunk_size(ncols);
                let partials: Vec<Vec<T>> = (0..ncols.div_ceil(chunk_size))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let col_start = chunk_idx * chunk_size;
                        let col_end = (col_start + chunk_size).min(ncols);
                        let mut local = vec![T::zero(); nrows * dense_cols];
                        for j in col_start..col_end {
                            for k in indptr[j].index()..indptr[j + 1].index() {
                                let i = indices[k].index();
                                let v = data[k];
                                for c in 0..dense_cols {
                                    local[i * dense_cols + c] += v * dense[(j, c)];
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
                // result = A^T @ dense, shape (ncols, dense_cols) — gather per col
                let col_results: Vec<(usize, Vec<T>)> = (0..ncols)
                    .into_par_iter()
                    .map(|j| {
                        let mut row = vec![T::zero(); dense_cols];
                        for k in indptr[j].index()..indptr[j + 1].index() {
                            let i = indices[k].index();
                            let v = data[k];
                            for c in 0..dense_cols {
                                row[c] += v * dense[(i, c)];
                            }
                        }
                        (j, row)
                    })
                    .collect();
                for (j, row) in col_results {
                    for c in 0..dense_cols {
                        result[(j, c)] = row[c];
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
            // result = (A - 1·means^T) @ dense
            // correction[c] = sum_j means[j] * dense[j,c]   (dot product, not product of sums)
            let ncols = self.cols();
            let correction: Vec<T> = (0..dense_cols)
                .map(|c| (0..ncols).fold(T::zero(), |acc, j| acc + means[j] * dense[(j, c)]))
                .collect();
            self.multiply_with_dense(dense, result, false);
            let nrows = self.rows();
            for i in 0..nrows {
                for c in 0..dense_cols {
                    result[(i, c)] -= correction[c];
                }
            }
        } else {
            // result = (A^T - means·1^T) @ dense
            // result[j,c] -= means[j] * col_sums_dense[c]
            let nrows = self.rows();
            let ncols = self.cols();
            let col_sums: Vec<T> = (0..dense_cols)
                .map(|c| (0..nrows).fold(T::zero(), |acc, i| acc + dense[(i, c)]))
                .collect();
            self.multiply_with_dense(dense, result, true);
            for j in 0..ncols {
                let mj = means[j];
                for c in 0..dense_cols {
                    result[(j, c)] -= mj * col_sums[c];
                }
            }
        }
    }

    fn multiply_transposed_by_dense(&self, q: &DMatrix<T>, result: &mut DMatrix<T>) {
        // result = Q^T @ A, shape (q.ncols, A.ncols)
        let nrows = self.rows();
        let ncols = self.cols();
        let q_cols = q.ncols();
        let indptr_view = self.indptr();
        let indptr = indptr_view.raw_storage();
        let indices = self.indices();
        let data = self.data();

        if self.is_csr() {
            // Scatter: parallel row chunks, flat partial buffers
            let chunk_size = determine_chunk_size(nrows);
            let partials: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk_idx| {
                    let row_start = chunk_idx * chunk_size;
                    let row_end = (row_start + chunk_size).min(nrows);
                    let mut local = vec![T::zero(); q_cols * ncols];
                    for i in row_start..row_end {
                        for k in indptr[i].index()..indptr[i + 1].index() {
                            let j = indices[k].index();
                            let v = data[k];
                            for c in 0..q_cols {
                                local[c * ncols + j] += q[(i, c)] * v;
                            }
                        }
                    }
                    local
                })
                .collect();
            result.fill(T::zero());
            for local in partials {
                for c in 0..q_cols {
                    for j in 0..ncols {
                        result[(c, j)] += local[c * ncols + j];
                    }
                }
            }
        } else {
            // CSC: gather per col — parallel
            let col_results: Vec<(usize, Vec<T>)> = (0..ncols)
                .into_par_iter()
                .map(|j| {
                    let mut col = vec![T::zero(); q_cols];
                    for k in indptr[j].index()..indptr[j + 1].index() {
                        let i = indices[k].index();
                        let v = data[k];
                        for c in 0..q_cols {
                            col[c] += q[(i, c)] * v;
                        }
                    }
                    (j, col)
                })
                .collect();
            result.fill(T::zero());
            for (j, col) in col_results {
                for c in 0..q_cols {
                    result[(c, j)] = col[c];
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
        // result = Q^T @ (A - 1·means^T)
        //        = Q^T @ A - (sum_i Q[i,:])^T @ means^T
        let q_rows = q.nrows();
        let q_cols = q.ncols();
        let ncols = self.cols();
        let q_col_sums: Vec<T> = (0..q_cols)
            .map(|c| (0..q_rows).fold(T::zero(), |acc, i| acc + q[(i, c)]))
            .collect();
        self.multiply_transposed_by_dense(q, result);
        for c in 0..q_cols {
            let qs = q_col_sums[c];
            for j in 0..ncols {
                result[(c, j)] -= qs * means[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use sprs::TriMat;

    /// Test matrix A (3×4):
    ///   1  2  0  0
    ///   0  0  3  4
    ///   0  5  0  6
    fn test_csr() -> sprs::CsMat<f64> {
        let mut tri: TriMat<f64> = TriMat::new((3, 4));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 2, 3.0);
        tri.add_triplet(1, 3, 4.0);
        tri.add_triplet(2, 1, 5.0);
        tri.add_triplet(2, 3, 6.0);
        tri.to_csr()
    }

    fn test_csc() -> sprs::CsMat<f64> {
        test_csr().to_csc()
    }

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

    // --- svd_opa ---

    #[test]
    fn test_csr_svd_opa_forward() {
        let csr = test_csr();
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 3];
        csr.svd_opa(&x, &mut y, false);
        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 3*3 + 4*4 = 25
        // y[2] = 5*2 + 6*4 = 34
        assert_slice_close(&y, &[5.0, 25.0, 34.0]);
    }

    #[test]
    fn test_csr_svd_opa_transposed() {
        let csr = test_csr();
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 4];
        csr.svd_opa(&x, &mut y, true);
        // y[0] = 1*1 = 1
        // y[1] = 2*1 + 5*3 = 17
        // y[2] = 3*2 = 6
        // y[3] = 4*2 + 6*3 = 26
        assert_slice_close(&y, &[1.0, 17.0, 6.0, 26.0]);
    }

    #[test]
    fn test_csc_svd_opa_forward() {
        let csc = test_csc();
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 3];
        csc.svd_opa(&x, &mut y, false);
        assert_slice_close(&y, &[5.0, 25.0, 34.0]);
    }

    #[test]
    fn test_csc_svd_opa_transposed() {
        let csc = test_csc();
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 4];
        csc.svd_opa(&x, &mut y, true);
        assert_slice_close(&y, &[1.0, 17.0, 6.0, 26.0]);
    }

    // --- compute_column_means ---

    #[test]
    fn test_column_means_csr() {
        let csr = test_csr();
        let means = csr.compute_column_means();
        // col 0: 1/3, col 1: 7/3, col 2: 3/3=1, col 3: 10/3
        assert_slice_close(&means, &[1.0 / 3.0, 7.0 / 3.0, 1.0, 10.0 / 3.0]);
    }

    #[test]
    fn test_column_means_csc() {
        let csc = test_csc();
        let means = csc.compute_column_means();
        assert_slice_close(&means, &[1.0 / 3.0, 7.0 / 3.0, 1.0, 10.0 / 3.0]);
    }

    // --- multiply_with_dense ---

    #[test]
    fn test_multiply_dense_forward_csr() {
        let csr = test_csr();
        // D: 4×2  (row-major: [1 2; 3 4; 5 6; 7 8])
        let dense = DMatrix::from_row_slice(4, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut result = DMatrix::zeros(3, 2);
        csr.multiply_with_dense(&dense, &mut result, false);
        // row 0: 1*[1,2] + 2*[3,4] = [7,10]
        // row 1: 3*[5,6] + 4*[7,8] = [43,50]
        // row 2: 5*[3,4] + 6*[7,8] = [57,68]
        assert_mat_close(
            &result,
            &[
                (0, 0, 7.0),
                (0, 1, 10.0),
                (1, 0, 43.0),
                (1, 1, 50.0),
                (2, 0, 57.0),
                (2, 1, 68.0),
            ],
        );
    }

    #[test]
    fn test_multiply_dense_transposed_csr() {
        let csr = test_csr();
        // D: 3×2  (row-major: [1 2; 3 4; 5 6])
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(4, 2);
        csr.multiply_with_dense(&dense, &mut result, true);
        // result[j,c] = sum_i A[i,j] * D[i,c]
        // j=0: A[0,0]*[1,2] = [1,2]
        // j=1: A[0,1]*[1,2] + A[2,1]*[5,6] = 2*[1,2]+5*[5,6] = [27,34]
        // j=2: A[1,2]*[3,4] = 3*[3,4] = [9,12]
        // j=3: A[1,3]*[3,4] + A[2,3]*[5,6] = 4*[3,4]+6*[5,6] = [42,52]
        assert_mat_close(
            &result,
            &[
                (0, 0, 1.0),
                (0, 1, 2.0),
                (1, 0, 27.0),
                (1, 1, 34.0),
                (2, 0, 9.0),
                (2, 1, 12.0),
                (3, 0, 42.0),
                (3, 1, 52.0),
            ],
        );
    }

    #[test]
    fn test_multiply_dense_forward_csc() {
        let csc = test_csc();
        let dense = DMatrix::from_row_slice(4, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut result = DMatrix::zeros(3, 2);
        csc.multiply_with_dense(&dense, &mut result, false);
        assert_mat_close(
            &result,
            &[
                (0, 0, 7.0),
                (0, 1, 10.0),
                (1, 0, 43.0),
                (1, 1, 50.0),
                (2, 0, 57.0),
                (2, 1, 68.0),
            ],
        );
    }

    #[test]
    fn test_multiply_dense_transposed_csc() {
        let csc = test_csc();
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(4, 2);
        csc.multiply_with_dense(&dense, &mut result, true);
        assert_mat_close(
            &result,
            &[
                (0, 0, 1.0),
                (0, 1, 2.0),
                (1, 0, 27.0),
                (1, 1, 34.0),
                (2, 0, 9.0),
                (2, 1, 12.0),
                (3, 0, 42.0),
                (3, 1, 52.0),
            ],
        );
    }

    // --- multiply_with_dense_centered ---
    // Uses non-constant means to distinguish the correct dot-product formula
    // from the product-of-sums bug in MaskedCSRMatrix.

    #[test]
    fn test_centered_forward() {
        let csr = test_csr();
        let means = DVector::from_vec(vec![0.5f64, 1.0, 1.5, 2.0]);
        let dense = DMatrix::from_row_slice(4, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut result = DMatrix::zeros(3, 2);
        csr.multiply_with_dense_centered(&dense, &mut result, false, &means);
        // correction[0] = 0.5*1 + 1.0*3 + 1.5*5 + 2.0*7 = 25
        // correction[1] = 0.5*2 + 1.0*4 + 1.5*6 + 2.0*8 = 30
        // result[i] = (A@D)[i] - correction
        assert_mat_close(
            &result,
            &[
                (0, 0, -18.0),
                (0, 1, -20.0),
                (1, 0, 18.0),
                (1, 1, 20.0),
                (2, 0, 32.0),
                (2, 1, 38.0),
            ],
        );
    }

    #[test]
    fn test_centered_transposed() {
        let csr = test_csr();
        let means = DVector::from_vec(vec![0.5f64, 1.0, 1.5, 2.0]);
        let dense = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(4, 2);
        csr.multiply_with_dense_centered(&dense, &mut result, true, &means);
        // col_sums_dense = [9, 12]
        // result[j,c] = (A^T@D)[j,c] - means[j]*col_sums[c]
        assert_mat_close(
            &result,
            &[
                (0, 0, -3.5),
                (0, 1, -4.0),
                (1, 0, 18.0),
                (1, 1, 22.0),
                (2, 0, -4.5),
                (2, 1, -6.0),
                (3, 0, 24.0),
                (3, 1, 28.0),
            ],
        );
    }

    // --- multiply_transposed_by_dense ---

    #[test]
    fn test_transposed_by_dense_csr() {
        let csr = test_csr();
        // Q: 3×2  (row-major: [1 2; 3 4; 5 6])
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 4);
        csr.multiply_transposed_by_dense(&q, &mut result);
        // result[c,j] = sum_i Q[i,c] * A[i,j], shape (2,4)
        // [0,0]=1*1=1         [0,1]=1*2+5*5=27  [0,2]=3*3=9   [0,3]=3*4+5*6=42
        // [1,0]=2*1=2         [1,1]=2*2+6*5=34  [1,2]=4*3=12  [1,3]=4*4+6*6=52
        assert_mat_close(
            &result,
            &[
                (0, 0, 1.0),
                (0, 1, 27.0),
                (0, 2, 9.0),
                (0, 3, 42.0),
                (1, 0, 2.0),
                (1, 1, 34.0),
                (1, 2, 12.0),
                (1, 3, 52.0),
            ],
        );
    }

    #[test]
    fn test_transposed_by_dense_csc() {
        let csc = test_csc();
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 4);
        csc.multiply_transposed_by_dense(&q, &mut result);
        assert_mat_close(
            &result,
            &[
                (0, 0, 1.0),
                (0, 1, 27.0),
                (0, 2, 9.0),
                (0, 3, 42.0),
                (1, 0, 2.0),
                (1, 1, 34.0),
                (1, 2, 12.0),
                (1, 3, 52.0),
            ],
        );
    }

    // --- multiply_transposed_by_dense_centered ---

    #[test]
    fn test_transposed_centered() {
        let csr = test_csr();
        let means = DVector::from_vec(vec![0.5f64, 1.0, 1.5, 2.0]);
        let q = DMatrix::from_row_slice(3, 2, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut result = DMatrix::zeros(2, 4);
        csr.multiply_transposed_by_dense_centered(&q, &mut result, &means);
        // q_col_sums = [9, 12]
        // result[c,j] = (Q^T@A)[c,j] - q_col_sums[c] * means[j]
        assert_mat_close(
            &result,
            &[
                (0, 0, -3.5),
                (0, 1, 18.0),
                (0, 2, -4.5),
                (0, 3, 24.0),
                (1, 0, -4.0),
                (1, 1, 22.0),
                (1, 2, -6.0),
                (1, 3, 28.0),
            ],
        );
    }
}
