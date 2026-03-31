pub mod legacy;
pub mod error;
pub(crate) mod utils;
pub mod sprs_impl;

pub mod randomized;

pub mod lanczos;

pub use utils::*;


#[cfg(test)]
mod simple_comparison_tests {
    use super::*;
    use legacy;
    use nalgebra_sparse::coo::CooMatrix;
    use nalgebra_sparse::CsrMatrix;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use rayon::ThreadPoolBuilder;
    use sprs::TriMat;

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
        let current_result = lanczos::svd_dim_seed(&test_matrix, 0, seed).unwrap();
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
        let legacy_svd = lanczos::svd_dim_seed(&csr, 0, seed as u32).unwrap();

        // Calculate SVD using our masked method (using all columns)
        let mask = vec![true; ncols];
        let masked_matrix = lanczos::masked::MaskedCSRMatrix::new(&csr, mask);
        let current_svd = lanczos::svd_dim_seed(&masked_matrix, 0, seed as u32).unwrap();

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
        let result = lanczos::svd_dim_seed(&test_matrix, 50, 42);
        assert!(result.is_ok(), "{}", format!("SVD failed on 99.02% sparse matrix, {:?}", result.err().unwrap()));
    }

    #[test]
    fn test_random_svd_computation() {
        let csr = make_sprs_matrix(1000, 250, 0.01);

        let result = randomized::randomized_svd(
            &csr,
            50,
            10,
            3,
            randomized::PowerIterationNormalizer::QR,
            false,
            Some(42),
            false
        );

        assert!(
            result.is_ok(),
            "Randomized SVD failed on 99% sparse matrix: {:?}",
            result.err().unwrap()
        );

        if let Ok(svd_result) = result {
            assert_eq!(svd_result.d, 50, "Expected rank of 50");

            for i in 0..svd_result.s.len() {
                assert!(svd_result.s[i] > 0.0, "Singular values should be positive");
                if i > 0 {
                    assert!(
                        svd_result.s[i-1] >= svd_result.s[i],
                        "Singular values should be in descending order"
                    );
                }
            }

            // u is (nrows × rank), vt is (rank × ncols)
            assert_eq!(svd_result.u.nrows(), 1000, "U should have 1000 rows");
            assert_eq!(svd_result.u.ncols(), 50, "U should have 50 columns");
            assert_eq!(svd_result.vt.nrows(), 50, "Vt should have 50 rows");
            assert_eq!(svd_result.vt.ncols(), 250, "Vt should have 250 columns");
        }
    }

    fn make_sprs_matrix(nrows: usize, ncols: usize, density: f64) -> sprs::CsMat<f64> {
        let mut tri: TriMat<f64> = TriMat::new((nrows, ncols));
        let mut rng = StdRng::seed_from_u64(42);
        let nnz = ((nrows as f64 * ncols as f64 * density).round() as usize).max(1);
        let mut positions = std::collections::HashSet::new();
        while positions.len() < nnz {
            let i = rng.gen_range(0..nrows);
            let j = rng.gen_range(0..ncols);
            if positions.insert((i, j)) {
                let v: f64 = rng.gen_range(-10.0..10.0);
                tri.add_triplet(i, j, v);
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_randomized_svd_very_large_sparse_matrix() {
        let csr = make_sprs_matrix(100000, 2500, 0.01);
        let threadpool = ThreadPoolBuilder::new().num_threads(10).build().unwrap();
        let result = threadpool.install(|| {
            randomized::randomized_svd(
                &csr,
                50,
                10,
                7,
                randomized::PowerIterationNormalizer::QR,
                false,
                Some(42),
                false,
            )
        });
        assert!(
            result.is_ok(),
            "Randomized SVD failed on 99% sparse matrix: {:?}",
            result.err().unwrap()
        );
    }

    #[test]
    fn test_randomized_svd_small_sparse_matrix() {
        let csr = make_sprs_matrix(1000, 250, 0.01);
        let threadpool = ThreadPoolBuilder::new().num_threads(10).build().unwrap();
        let result = threadpool.install(|| {
            randomized::randomized_svd(
                &csr,
                50,
                10,
                2,
                randomized::PowerIterationNormalizer::QR,
                false,
                Some(42),
                false,
            )
        });
        assert!(
            result.is_ok(),
            "Randomized SVD failed on 99% sparse matrix: {:?}",
            result.err().unwrap()
        );
    }
}