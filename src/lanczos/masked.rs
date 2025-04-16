use crate::utils::determine_chunk_size;
use crate::{SMat, SvdFloat};
use nalgebra_sparse::CsrMatrix;
use num_traits::Float;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelIterator, ParallelBridge};
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

impl<'a, T: Float + AddAssign + Sync + Send> SMat<T> for MaskedCSRMatrix<'a, T> {
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
        // TODO  parallelize me please
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

        y.fill(T::zero());

        let high_precision_mode = self.ensure_identical_results_mode();

        if !transposed {
            if high_precision_mode && self.uses_all_columns() {
                // For small matrices using all columns, mimic the exact behavior of
                // the original implementation to ensure identical results
                for i in 0..self.matrix.nrows() {
                    let mut sum = T::zero();
                    for j in major_offsets[i]..major_offsets[i + 1] {
                        let col = minor_indices[j];
                        // For all-columns mode, we know all columns are included
                        let masked_col = self.original_to_masked[col].unwrap();
                        sum = sum + (values[j] * x[masked_col]);
                    }
                    y[i] = sum;
                }
            } else {
                let chunk_size = determine_chunk_size(self.matrix.nrows());
                y.chunks_mut(chunk_size).enumerate().par_bridge().for_each(
                    |(chunk_idx, y_chunk)| {
                        let start_row = chunk_idx * chunk_size;
                        let end_row = (start_row + y_chunk.len()).min(self.matrix.nrows());

                        for i in start_row..end_row {
                            let row_idx = i - start_row;
                            let mut sum = T::zero();

                            for j in major_offsets[i]..major_offsets[i + 1] {
                                let col = minor_indices[j];
                                if let Some(masked_col) = self.original_to_masked[col] {
                                    sum += values[j] * x[masked_col];
                                };
                            }
                            y_chunk[row_idx] = sum;
                        }
                    },
                );
            }
        } else {
            // For the transposed case (A^T * x)
            if high_precision_mode && self.uses_all_columns() {
                // Clear the output vector first
                for yval in y.iter_mut() {
                    *yval = T::zero();
                }

                // Follow exact same order of operations as original implementation
                for i in 0..self.matrix.nrows() {
                    let row_val = x[i];
                    for j in major_offsets[i]..major_offsets[i + 1] {
                        let col = minor_indices[j];
                        let masked_col = self.original_to_masked[col].unwrap();
                        y[masked_col] = y[masked_col] + (values[j] * row_val);
                    }
                }
            } else {
                let nrows = self.matrix.nrows();
                let chunk_size = determine_chunk_size(nrows);
                let num_chunks = (nrows + chunk_size - 1) / chunk_size;
                let results: Vec<Vec<T>> = (0..chunk_size)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(nrows);

                        let mut local_y = vec![T::zero(); y.len()];
                        for i in start..end {
                            let row_val = x[i];
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

                y.fill(T::zero());

                for local_y in results {
                    for (idx, val) in local_y.iter().enumerate() {
                        if !val.is_zero() {
                            y[idx] += *val;
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
    use crate::{SMat};
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
