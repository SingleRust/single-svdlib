pub mod legacy;
pub mod error;
mod new;
mod masked;
pub(crate) mod utils;

pub mod randomized;

pub use new::*;
pub use masked::*;

#[cfg(test)]
mod simple_comparison_tests {
    use super::*;
    use legacy;
    use nalgebra_sparse::coo::CooMatrix;
    use nalgebra_sparse::CsrMatrix;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn create_sparse_matrix(rows: usize, cols: usize, density: f64) -> nalgebra_sparse::coo::CooMatrix<f64> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use std::collections::HashSet;

        let mut coo = nalgebra_sparse::coo::CooMatrix::new(rows, cols);

        let mut rng = StdRng::seed_from_u64(42);

        let nnz = (rows as f64 * cols as f64 * density).round() as usize;

        let nnz = nnz.max(1);

        let mut positions = HashSet::new();

        while positions.len() < nnz {
            let i = rng.gen_range(0..rows);
            let j = rng.gen_range(0..cols);

            if positions.insert((i, j)) {
                let val = loop {
                    let v: f64 = rng.gen_range(-10.0..10.0);
                    if v.abs() > 1e-10 { // Ensure it's not too close to zero
                        break v;
                    }
                };

                coo.push(i, j, val);
            }
        }

        // Verify the density is as expected
        let actual_density = coo.nnz() as f64 / (rows as f64 * cols as f64);
        println!("Created sparse matrix: {} x {}", rows, cols);
        println!("  - Requested density: {:.6}", density);
        println!("  - Actual density: {:.6}", actual_density);
        println!("  - Sparsity: {:.4}%", (1.0 - actual_density) * 100.0);
        println!("  - Non-zeros: {}", coo.nnz());

        coo
    }
    //#[test]
    fn simple_matrix_comparison() {
        // Create a small, predefined test matrix
        let mut test_matrix = CooMatrix::<f64>::new(3, 3);
        test_matrix.push(0, 0, 1.0);
        test_matrix.push(0, 1, 16.0);
        test_matrix.push(0, 2, 49.0);
        test_matrix.push(1, 0, 4.0);
        test_matrix.push(1, 1, 25.0);
        test_matrix.push(1, 2, 64.0);
        test_matrix.push(2, 0, 9.0);
        test_matrix.push(2, 1, 36.0);
        test_matrix.push(2, 2, 81.0);

        // Run both implementations with the same seed for deterministic behavior
        let seed = 42;
        let current_result = svd_dim_seed(&test_matrix, 0, seed).unwrap();
        let legacy_result = legacy::svd_dim_seed(&test_matrix, 0, seed).unwrap();

        // Compare dimensions
        assert_eq!(current_result.d, legacy_result.d);

        // Compare singular values
        let epsilon = 1.0e-12;
        for i in 0..current_result.d {
            let diff = (current_result.s[i] - legacy_result.s[i]).abs();
            assert!(
                diff < epsilon,
                "Singular value {} differs by {}: current = {}, legacy = {}",
                i, diff, current_result.s[i], legacy_result.s[i]
            );
        }

        // Compare reconstructed matrices
        let current_reconstructed = current_result.recompose();
        let legacy_reconstructed = legacy_result.recompose();

        for i in 0..3 {
            for j in 0..3 {
                let diff = (current_reconstructed[[i, j]] - legacy_reconstructed[[i, j]]).abs();
                assert!(
                    diff < epsilon,
                    "Reconstructed matrix element [{},{}] differs by {}: current = {}, legacy = {}",
                    i, j, diff, current_reconstructed[[i, j]], legacy_reconstructed[[i, j]]
                );
            }
        }
    }

    #[test]
    fn random_matrix_comparison() {
        let seed = 12345;
        let (nrows, ncols) = (50, 30);
        let mut rng = StdRng::seed_from_u64(seed);

        // Create random sparse matrix
        let mut coo = CooMatrix::<f64>::new(nrows, ncols);
        // Insert some random non-zero elements
        for _ in 0..(nrows * ncols / 5) {  // ~20% density
            let i = rng.gen_range(0..nrows);
            let j = rng.gen_range(0..ncols);
            let value = rng.gen_range(-10.0..10.0);
            coo.push(i, j, value);
        }

        let csr = CsrMatrix::from(&coo);

        // Calculate SVD using original method
        let legacy_svd = svd_dim_seed(&csr, 0, seed as u32).unwrap();

        // Calculate SVD using our masked method (using all columns)
        let mask = vec![true; ncols];
        let masked_matrix = MaskedCSRMatrix::new(&csr, mask);
        let current_svd = svd_dim_seed(&masked_matrix, 0, seed as u32).unwrap();

        // Compare with relative tolerance
        let rel_tol = 1e-3;  // 0.1% relative tolerance

        assert_eq!(legacy_svd.d, current_svd.d, "Ranks differ");

        for i in 0..legacy_svd.d {
            let legacy_val = legacy_svd.s[i];
            let current_val = current_svd.s[i];
            let abs_diff = (legacy_val - current_val).abs();
            let rel_diff = abs_diff / legacy_val.max(current_val);

            assert!(
                rel_diff <= rel_tol,
                "Singular value {} differs too much: relative diff = {}, current = {}, legacy = {}",
                i, rel_diff, current_val, legacy_val
            );
        }
    }



    #[test]
    fn test_real_sparse_matrix() {
        // Create a matrix with similar sparsity to your real one (99.02%)
        let test_matrix = create_sparse_matrix(100, 100, 0.0098); // 0.98% non-zeros
        
        // Should no longer fail with convergence error
        let result = svd_dim_seed(&test_matrix, 50, 42);
        assert!(result.is_ok(), "{}", format!("SVD failed on 99.02% sparse matrix, {:?}", result.err().unwrap()));
    }

    #[test]
    fn test_real_sparse_matrix_very_big() {
        // Create a matrix with similar sparsity to your real one (99.02%)
        let test_matrix = create_sparse_matrix(10000, 1000, 0.01); // 0.98% non-zeros

        // Should no longer fail with convergence error
        let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let result = thread_pool.install(|| {
            svd_dim_seed(&test_matrix, 50, 42)
        });
        assert!(result.is_ok(), "{}", format!("SVD failed on 99% sparse matrix, {:?}", result.err().unwrap()));
    }

    #[test]
    fn test_real_sparse_matrix_very_very_big() {
        // Create a matrix with similar sparsity to your real one (99.02%)
        let test_matrix = create_sparse_matrix(100000, 2500, 0.01); // 0.98% non-zeros

        // Should no longer fail with convergence error

        let result = svd(&test_matrix);
        assert!(result.is_ok(), "{}", format!("SVD failed on 99% sparse matrix, {:?}", result.err().unwrap()));
    }

    //#[test]
    fn f32_precision_test() {
        let seed = 12345;
        let (nrows, ncols) = (40, 20);
        let mut rng = StdRng::seed_from_u64(seed);

        // Create random sparse matrix with f32 values
        let mut coo_f32 = CooMatrix::<f32>::new(nrows, ncols);
        // And the same matrix with f64 values
        let mut coo_f64 = CooMatrix::<f64>::new(nrows, ncols);

        // Insert the same random non-zero elements in both matrices
        for _ in 0..(nrows * ncols / 4) {  // ~25% density
            let i = rng.gen_range(0..nrows);
            let j = rng.gen_range(0..ncols);
            let value = rng.gen_range(-10.0..10.0);
            coo_f32.push(i, j, value as f32);
            coo_f64.push(i, j, value);
        }

        let csr_f32 = CsrMatrix::from(&coo_f32);
        let csr_f64 = CsrMatrix::from(&coo_f64);

        // Calculate SVD for both types
        let svd_f32 = svd_dim_seed(&csr_f32, 0, seed as u32).unwrap();
        let svd_f64 = svd_dim_seed(&csr_f64, 0, seed as u32).unwrap();

        // Adaptive tolerance - increases for smaller singular values
        // and values further down the list (which are more affected by accumulated errors)
        fn calculate_tolerance(index: usize, magnitude: f64) -> f64 {
            // Base tolerance is 0.2%
            let base_tol = 0.002;

            // Scale up for smaller values (more affected by precision)
            let magnitude_factor = if magnitude < 1.0 {
                2.0  // Double tolerance for very small values
            } else if magnitude < 10.0 {
                1.5  // 1.5x tolerance for moderately small values
            } else {
                1.0  // Normal tolerance for larger values
            };

            // Scale up for later indices (more affected by accumulated errors)
            let index_factor = 1.0 + (index as f64 * 0.001);  // Add 0.1% per index

            base_tol * magnitude_factor * index_factor
        }

        println!("f32 rank: {}, f64 rank: {}", svd_f32.d, svd_f64.d);

        // Compare singular values up to the minimum rank
        let min_rank = svd_f32.d.min(svd_f64.d);
        for i in 0..min_rank {
            let f32_val = svd_f32.s[i];
            let f64_val = svd_f64.s[i];
            let abs_diff = (f64_val - f32_val as f64).abs();
            let rel_diff = abs_diff / f64_val.max(f32_val as f64);

            // Calculate appropriate tolerance for this value
            let tol = calculate_tolerance(i, f64_val);

            println!("Singular value {}: f32 = {}, f64 = {}, rel_diff = {:.6}%, tol = {:.6}%",
                     i, f32_val, f64_val, rel_diff * 100.0, tol * 100.0);

            assert!(
                rel_diff <= tol,
                "Singular value {} differs too much: f64 = {}, f32 = {}. Relative diff: {} > tolerance: {}",
                i, f64_val, f32_val, rel_diff, tol
            );
        }

        // Optional: Also check that overall behavior is reasonable
        // For example, check that both implementations find similar condition numbers
        if min_rank >= 2 {
            let condition_f32 = svd_f32.s[0] / svd_f32.s[min_rank - 1];
            let condition_f64 = svd_f64.s[0] / svd_f64.s[min_rank - 1];
            let condition_rel_diff = ((condition_f64 - condition_f32 as f64) / condition_f64).abs();

            println!("Condition number: f32 = {}, f64 = {}, rel_diff = {:.6}%",
                     condition_f32, condition_f64, condition_rel_diff * 100.0);

            // Condition number can vary more, use 5% tolerance
            assert!(condition_rel_diff <= 0.05,
                    "Condition numbers differ too much: f32 = {}, f64 = {}",
                    condition_f32, condition_f64);
        }
    }
}