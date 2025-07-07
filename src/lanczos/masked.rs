use crate::{determine_chunk_size, SMat, SvdFloat};
use nalgebra_sparse::na::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use num_traits::Float;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelSliceMut,
};
use std::fmt::Debug;
use std::ops::AddAssign;

pub struct MaskedCSRMatrix<'a, T: Float> {
    matrix: &'a CsrMatrix<T>,
    column_mask: Vec<bool>,
    masked_to_original: Vec<usize>,
    original_to_masked: Vec<Option<usize>>,
}

impl<'a, T: Float> MaskedCSRMatrix<'a, T> {
    pub fn new(matrix: &'a CsrMatrix<T>, column_mask: Vec<bool>) -> Self {
        assert_eq!(
            column_mask.len(),
            matrix.ncols(),
            "Column mask must have the same length as the number of columns in the matrix"
        );

        let mut masked_to_original = Vec::new();
        let mut original_to_masked = vec![None; column_mask.len()];
        let mut masked_index = 0;

        for (i, &is_included) in column_mask.iter().enumerate() {
            if is_included {
                masked_to_original.push(i);
                original_to_masked[i] = Some(masked_index);
                masked_index += 1;
            }
        }

        Self {
            matrix,
            column_mask,
            masked_to_original,
            original_to_masked,
        }
    }

    pub fn with_columns(matrix: &'a CsrMatrix<T>, columns: &[usize]) -> Self {
        let mut mask = vec![false; matrix.ncols()];
        for &col in columns {
            assert!(col < matrix.ncols(), "Column index out of bounds");
            mask[col] = true;
        }
        Self::new(matrix, mask)
    }

    pub fn uses_all_columns(&self) -> bool {
        self.masked_to_original.len() == self.matrix.ncols() && self.column_mask.iter().all(|&x| x)
    }

    pub fn ensure_identical_results_mode(&self) -> bool {
        // For very small matrices where precision is critical
        let is_small_matrix = self.matrix.nrows() <= 5 && self.matrix.ncols() <= 5;
        is_small_matrix && self.uses_all_columns()
    }
}

impl<
        T: Float
            + AddAssign
            + Sync
            + Send
            + std::ops::MulAssign
            + Debug
            + 'static
            + std::iter::Sum
            + std::ops::SubAssign
            + num_traits::FromPrimitive,
    > SMat<T> for MaskedCSRMatrix<'_, T>
{
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.masked_to_original.len()
    }

    fn nnz(&self) -> usize {
        let (major_offsets, minor_indices, _) = self.matrix.csr_data();
        let mut count = 0;

        for i in 0..self.matrix.nrows() {
            for j in major_offsets[i]..major_offsets[i + 1] {
                let col = minor_indices[j];
                if self.column_mask[col] {
                    count += 1;
                }
            }
        }
        count
    }

    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = if transposed {
            self.ncols()
        } else {
            self.nrows()
        };
        let ncols = if transposed {
            self.nrows()
        } else {
            self.ncols()
        };

        assert_eq!(
            x.len(),
            ncols,
            "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}",
            x.len(),
            ncols
        );
        assert_eq!(
            y.len(),
            nrows,
            "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}",
            y.len(),
            nrows
        );

        let (major_offsets, minor_indices, values) = self.matrix.csr_data();

        if self.uses_all_columns() || (self.matrix.nrows() < 1000 && self.matrix.ncols() < 1000) {
            // Fast path for unmasked matrices or small matrices
            if !transposed {
                // A * x calculation
                self.matrix.svd_opa(x, y, false);
            } else {
                // A^T * x calculation
                self.matrix.svd_opa(x, y, true);
            }
            return;
        }

        y.fill(T::zero());

        if !transposed {
            // A * x calculation
            let valid_indices: Vec<Option<usize>> = (0..self.matrix.ncols())
                .map(|col| self.original_to_masked[col])
                .collect();

            // Parallelization parameters
            let rows = self.matrix.nrows();
            let chunk_size = std::cmp::max(16, rows / (rayon::current_num_threads() * 2));

            // Process in parallel chunks
            y.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, y_chunk)| {
                    let start_row = chunk_idx * chunk_size;
                    let end_row = (start_row + y_chunk.len()).min(rows);

                    for i in start_row..end_row {
                        let row_idx = i - start_row;
                        let mut sum = T::zero();

                        // Process row in blocks of 16 elements for better vectorization
                        let row_start = major_offsets[i];
                        let row_end = major_offsets[i + 1];

                        // Unroll the loop by 4 for better instruction-level parallelism
                        let mut j = row_start;
                        while j + 4 <= row_end {
                            for offset in 0..4 {
                                let idx = j + offset;
                                let col = minor_indices[idx];
                                if let Some(masked_col) = valid_indices[col] {
                                    sum += values[idx] * x[masked_col];
                                }
                            }
                            j += 4;
                        }

                        // Handle remaining elements
                        while j < row_end {
                            let col = minor_indices[j];
                            if let Some(masked_col) = valid_indices[col] {
                                sum += values[j] * x[masked_col];
                            }
                            j += 1;
                        }

                        y_chunk[row_idx] = sum;
                    }
                });
        } else {
            // A^T * x calculation
            let nrows = self.matrix.nrows();
            let chunk_size = crate::utils::determine_chunk_size(nrows);

            // Create thread-local partial results and combine at the end
            let results: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(nrows);
                    let mut local_y = vec![T::zero(); y.len()];

                    // Process a chunk of rows
                    for i in start..end {
                        let row_val = x[i];
                        if row_val.is_zero() {
                            continue; // Skip zero values for performance
                        }

                        for j in major_offsets[i]..major_offsets[i + 1] {
                            let col = minor_indices[j];
                            if let Some(masked_col) = self.original_to_masked[col] {
                                local_y[masked_col] += values[j] * row_val;
                            }
                        }
                    }
                    local_y
                })
                .collect();

            // Combine results efficiently
            for local_y in results {
                // Only update non-zero elements to reduce memory traffic
                for (idx, &val) in local_y.iter().enumerate() {
                    if !val.is_zero() {
                        y[idx] += val;
                    }
                }
            }
        }
    }

    fn compute_column_means(&self) -> Vec<T> {
        let rows = self.nrows();
        let masked_cols = self.ncols();
        let row_count_recip = T::one() / T::from(rows).unwrap();

        let mut col_sums = vec![T::zero(); masked_cols];
        let (row_offsets, col_indices, values) = self.matrix.csr_data();

        for i in 0..rows {
            for j in row_offsets[i]..row_offsets[i + 1] {
                let original_col = col_indices[j];
                if let Some(masked_col) = self.original_to_masked[original_col] {
                    col_sums[masked_col] += values[j];
                }
            }
        }

        // Convert to means
        for j in 0..masked_cols {
            col_sums[j] *= row_count_recip;
        }

        col_sums
    }

    fn multiply_with_dense(
        &self,
        dense: &DMatrix<T>,
        result: &mut DMatrix<T>,
        transpose_self: bool,
    ) {
        let m_rows = if transpose_self {
            self.ncols()
        } else {
            self.nrows()
        };
        let m_cols = if transpose_self {
            self.nrows()
        } else {
            self.ncols()
        };

        assert_eq!(
            dense.nrows(),
            m_cols,
            "Dense matrix has incompatible row count"
        );
        assert_eq!(
            result.nrows(),
            m_rows,
            "Result matrix has incompatible row count"
        );
        assert_eq!(
            result.ncols(),
            dense.ncols(),
            "Result matrix has incompatible column count"
        );

        let (major_offsets, minor_indices, values) = self.matrix.csr_data();

        if !transpose_self {
            let rows = self.matrix.nrows();
            let dense_cols = dense.ncols();

            // Pre-filter valid column mappings to avoid repeated lookups
            let valid_cols: Vec<Option<usize>> = (0..self.matrix.ncols())
                .map(|col| self.original_to_masked.get(col).copied().flatten())
                .collect();

            // Compute results in parallel, then apply to result matrix
            let row_results: Vec<(usize, Vec<T>)> = (0..rows)
                .into_par_iter()
                .map(|row| {
                    let mut row_result = vec![T::zero(); dense_cols];

                    // Process sparse row with blocked inner loop for better vectorization
                    let row_start = major_offsets[row];
                    let row_end = major_offsets[row + 1];

                    // Unroll the sparse elements loop by 4 for better ILP
                    let mut j = row_start;
                    while j + 4 <= row_end {
                        // Process 4 sparse elements at once
                        for offset in 0..4 {
                            let idx = j + offset;
                            let col = minor_indices[idx];
                            if let Some(masked_col) = valid_cols[col] {
                                let val = values[idx];

                                // Vectorized dense column update
                                for c in 0..dense_cols {
                                    row_result[c] += val * dense[(masked_col, c)];
                                }
                            }
                        }
                        j += 4;
                    }

                    // Handle remaining elements
                    while j < row_end {
                        let col = minor_indices[j];
                        if let Some(masked_col) = valid_cols[col] {
                            let val = values[j];

                            for c in 0..dense_cols {
                                row_result[c] += val * dense[(masked_col, c)];
                            }
                        }
                        j += 1;
                    }

                    (row, row_result)
                })
                .collect();

            // Apply results to output matrix
            for (row, row_values) in row_results {
                for c in 0..dense_cols {
                    result[(row, c)] = row_values[c];
                }
            }
        } else {
            let nrows = self.matrix.nrows();
            let ncols = self.ncols();
            let dense_cols = dense.ncols();

            // Clear result matrix once at the beginning
            result.fill(T::zero());

            // Pre-filter valid column mappings
            let valid_cols: Vec<Option<usize>> = (0..self.matrix.ncols())
                .map(|col| self.original_to_masked.get(col).copied().flatten())
                .collect();

            let chunk_size = determine_chunk_size(nrows);

            // Use atomic-free approach with proper synchronization
            let partial_results: Vec<Vec<T>> = (0..nrows.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(nrows);

                    // Use flat vector for better cache performance
                    let mut local_result = vec![T::zero(); ncols * dense_cols];

                    // Process chunk with better memory access patterns
                    for i in start..end {
                        let dense_row = unsafe {
                            std::slice::from_raw_parts(
                                dense.as_ptr().add(i * dense_cols),
                                dense_cols,
                            )
                        };

                        // Block processing for better cache usage
                        let row_start = major_offsets[i];
                        let row_end = major_offsets[i + 1];

                        // Process sparse elements in blocks of 8 for better vectorization
                        let mut j = row_start;
                        while j + 8 <= row_end {
                            for offset in 0..8 {
                                let idx = j + offset;
                                let col = minor_indices[idx];
                                if let Some(masked_col) = valid_cols[col] {
                                    let val = values[idx];
                                    let base_offset = masked_col * dense_cols;

                                    // Vectorized update with manual loop unrolling
                                    let mut c = 0;
                                    while c + 4 <= dense_cols {
                                        local_result[base_offset + c] += val * dense_row[c];
                                        local_result[base_offset + c + 1] += val * dense_row[c + 1];
                                        local_result[base_offset + c + 2] += val * dense_row[c + 2];
                                        local_result[base_offset + c + 3] += val * dense_row[c + 3];
                                        c += 4;
                                    }

                                    // Handle remaining columns
                                    while c < dense_cols {
                                        local_result[base_offset + c] += val * dense_row[c];
                                        c += 1;
                                    }
                                }
                            }
                            j += 8;
                        }

                        // Handle remaining sparse elements
                        while j < row_end {
                            let col = minor_indices[j];
                            if let Some(masked_col) = valid_cols[col] {
                                let val = values[j];
                                let base_offset = masked_col * dense_cols;

                                for c in 0..dense_cols {
                                    local_result[base_offset + c] += val * dense_row[c];
                                }
                            }
                            j += 1;
                        }
                    }

                    local_result
                })
                .collect();

            // Efficient reduction with blocked memory access
            const BLOCK_SIZE: usize = 32;
            for local_result in partial_results {
                // Process in blocks for better cache performance
                for r_block in (0..ncols).step_by(BLOCK_SIZE) {
                    let r_end = (r_block + BLOCK_SIZE).min(ncols);

                    for c_block in (0..dense_cols).step_by(BLOCK_SIZE) {
                        let c_end = (c_block + BLOCK_SIZE).min(dense_cols);

                        // Update result block
                        for r in r_block..r_end {
                            for c in c_block..c_end {
                                let val = local_result[r * dense_cols + c];
                                if !val.is_zero() {
                                    result[(r, c)] += val;
                                }
                            }
                        }
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
        let (major_offsets, minor_indices, values) = self.matrix.csr_data();

        // Pre-compute column sums for the dense matrix - do this once
        let dense_cols = dense.ncols();
        let dense_rows = dense.nrows();

        // Pre-compute all column sums to avoid redundant calculations
        let col_sums: Vec<T> = (0..dense_cols)
            .into_par_iter()
            .map(|c| (0..dense_rows).map(|i| dense[(i, c)]).sum())
            .collect();

        if !transpose_self {
            let rows = self.matrix.nrows();

            // Pre-compute mean adjustments for each column
            let mean_adjustments: Vec<T> = col_sums
                .iter()
                .map(|&col_sum| {
                    means
                        .iter()
                        .enumerate()
                        .filter_map(|(original_idx, &mean_val)| {
                            self.original_to_masked
                                .get(original_idx)
                                .map(|_| mean_val * col_sum)
                        })
                        .sum()
                })
                .collect();

            let row_updates: Vec<(usize, Vec<T>)> = (0..rows)
                .into_par_iter()
                .map(|row| {
                    let mut row_result = vec![T::zero(); dense_cols];

                    for j in major_offsets[row]..major_offsets[row + 1] {
                        let col = minor_indices[j];
                        if let Some(masked_col) = self.original_to_masked[col] {
                            let val = values[j];

                            for c in 0..dense_cols {
                                row_result[c] += val * dense[(masked_col, c)];
                            }
                        }
                    }

                    for c in 0..dense_cols {
                        row_result[c] -= mean_adjustments[c];
                    }

                    (row, row_result)
                })
                .collect();

            for (row, row_values) in row_updates {
                for c in 0..dense_cols {
                    result[(row, c)] = row_values[c];
                }
            }
        } else {
            let nrows = self.matrix.nrows();
            let ncols = self.ncols();

            // Clear the result matrix first
            for i in 0..result.nrows() {
                for j in 0..result.ncols() {
                    result[(i, j)] = T::zero();
                }
            }

            // Choose optimal chunk size
            let chunk_size = determine_chunk_size(nrows);

            // Compute partial results in parallel
            let partial_results: Vec<DMatrix<T>> = (0..nrows.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, nrows);

                    let mut local_result = DMatrix::<T>::zeros(ncols, dense_cols);

                    for i in start..end {
                        for j in major_offsets[i]..major_offsets[i + 1] {
                            let col = minor_indices[j];
                            if let Some(masked_col) = self.original_to_masked[col] {
                                let sparse_val = values[j];

                                for c in 0..dense_cols {
                                    local_result[(masked_col, c)] += sparse_val * dense[(i, c)];
                                }
                            }
                        }
                    }

                    // Apply mean adjustment for this chunk
                    let chunk_fraction =
                        T::from_f64((end - start) as f64 / dense_rows as f64).unwrap();

                    for masked_col in 0..ncols {
                        if masked_col < means.len() {
                            let mean = means[masked_col];
                            for c in 0..dense_cols {
                                local_result[(masked_col, c)] -=
                                    mean * col_sums[c] * chunk_fraction;
                            }
                        }
                    }

                    local_result
                })
                .collect();

            for local_result in partial_results {
                const BLOCK_SIZE: usize = 32;

                for r_block in 0..ncols.div_ceil(BLOCK_SIZE) {
                    let r_start = r_block * BLOCK_SIZE;
                    let r_end = std::cmp::min(r_start + BLOCK_SIZE, ncols);

                    for c_block in 0..dense_cols.div_ceil(BLOCK_SIZE) {
                        let c_start = c_block * BLOCK_SIZE;
                        let c_end = std::cmp::min(c_start + BLOCK_SIZE, dense_cols);

                        for r in r_start..r_end {
                            for c in c_start..c_end {
                                result[(r, c)] += local_result[(r, c)];
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SMat;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_masked_matrix() {
        // Create a test matrix
        let mut coo = CooMatrix::<f64>::new(3, 5);
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(0, 4, 3.0);
        coo.push(1, 1, 4.0);
        coo.push(1, 3, 5.0);
        coo.push(2, 0, 6.0);
        coo.push(2, 2, 7.0);
        coo.push(2, 4, 8.0);

        let csr = CsrMatrix::from(&coo);

        // Create a masked matrix with columns 0, 2, 4
        let columns = vec![0, 2, 4];
        let masked = MaskedCSRMatrix::with_columns(&csr, &columns);

        // Check dimensions
        assert_eq!(masked.nrows(), 3);
        assert_eq!(masked.ncols(), 3);
        assert_eq!(masked.nnz(), 6); // Only entries in the selected columns

        // Test SVD on the masked matrix
        let svd_result = crate::lanczos::svd(&masked);
        assert!(svd_result.is_ok());
    }

    #[test]
    fn test_masked_vs_physical_subset() {
        // Create a fixed seed for reproducible tests
        let mut rng = StdRng::seed_from_u64(42);

        // Generate a random matrix (5x8)
        let nrows = 14;
        let ncols = 10;
        let nnz = 40; // Number of non-zero elements

        let mut coo = CooMatrix::<f64>::new(nrows, ncols);

        // Fill with random non-zero values
        for _ in 0..nnz {
            let row = rng.gen_range(0..nrows);
            let col = rng.gen_range(0..ncols);
            let val = rng.gen_range(0.1..10.0);

            // Note: CooMatrix will overwrite if the position already has a value
            coo.push(row, col, val);
        }

        // Convert to CSR which is what our masked implementation uses
        let csr = CsrMatrix::from(&coo);

        // Select a subset of columns (e.g., columns 1, 3, 5, 7)
        let selected_columns = vec![1, 3, 5, 7];

        // Create the masked matrix view
        let masked_matrix = MaskedCSRMatrix::with_columns(&csr, &selected_columns);

        // Create a physical copy with just those columns
        let mut physical_subset = CooMatrix::<f64>::new(nrows, selected_columns.len());

        // Map original column indices to new column indices
        let col_map: std::collections::HashMap<usize, usize> = selected_columns
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        // Copy the values for the selected columns
        for (row, col, val) in coo.triplet_iter() {
            if let Some(&new_col) = col_map.get(&col) {
                physical_subset.push(row, new_col, *val);
            }
        }

        // Convert to CSR for SVD
        let physical_csr = CsrMatrix::from(&physical_subset);

        // Compare dimensions and nnz
        assert_eq!(masked_matrix.nrows(), physical_csr.nrows());
        assert_eq!(masked_matrix.ncols(), physical_csr.ncols());
        assert_eq!(masked_matrix.nnz(), physical_csr.nnz());

        // Perform SVD on both
        let svd_masked = crate::lanczos::svd(&masked_matrix).unwrap();
        let svd_physical = crate::lanczos::svd(&physical_csr).unwrap();

        // Compare SVD results - they should be very close but not exactly the same
        // due to potential differences in numerical computation

        // Check dimension (rank)
        assert_eq!(svd_masked.d, svd_physical.d);

        // Basic tolerance for floating point comparisons
        let epsilon = 1e-10;

        // Check singular values (may be in different order, so we sort them)
        let mut masked_s = svd_masked.s.to_vec();
        let mut physical_s = svd_physical.s.to_vec();
        masked_s.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort in descending order
        physical_s.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for (m, p) in masked_s.iter().zip(physical_s.iter()) {
            assert!(
                (m - p).abs() < epsilon,
                "Singular values differ: {} vs {}",
                m,
                p
            );
        }

        // Note: Comparing singular vectors is more complex due to potential sign flips
        // and different ordering, so we'll skip that level of detailed comparison
    }
}
