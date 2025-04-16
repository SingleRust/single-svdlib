use crate::error::SvdLibError;
use crate::utils::determine_chunk_size;
use crate::{Diagnostics, SMat, SvdFloat, SvdRec};
use nalgebra_sparse::na::{ComplexField, DMatrix, DVector, RealField};
use ndarray::Array1;
use nshare::IntoNdarray2;
use rand::prelude::{Distribution, StdRng};
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
use std::ops::Mul;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerIterationNormalizer {
    QR,
    LU,
    None,
}

const PARALLEL_THRESHOLD_ROWS: usize = 5000;
const PARALLEL_THRESHOLD_COLS: usize = 1000;
const PARALLEL_THRESHOLD_ELEMENTS: usize = 100_000;

pub fn randomized_svd<T, M>(
    m: &M,
    target_rank: usize,
    n_oversamples: usize,
    n_power_iters: usize,
    power_iteration_normalizer: PowerIterationNormalizer,
    mean_center: bool,
    seed: Option<u64>,
) -> anyhow::Result<SvdRec<T>>
where
    T: SvdFloat + RealField,
    M: SMat<T>,
    T: ComplexField,
{
    let m_rows = m.nrows();
    let m_cols = m.ncols();

    let rank = target_rank.min(m_rows.min(m_cols));
    let l = rank + n_oversamples;

    let column_means = if mean_center {
        compute_column_means(m)
    } else {
        None
    };

    let mut omega = generate_random_matrix(m_cols, l, seed);

    let mut y = DMatrix::<T>::zeros(m_rows, l);
    multiply_matrix_centered(m, &omega, &mut y, false, &column_means);

    if n_power_iters > 0 {
        let mut z = DMatrix::<T>::zeros(m_cols, l);

        for _ in 0..n_power_iters {
            multiply_matrix_centered(m, &y, &mut z, true, &column_means);
            match power_iteration_normalizer {
                PowerIterationNormalizer::QR => {
                    let qr = z.qr();
                    z = qr.q();
                }
                PowerIterationNormalizer::LU => {
                    normalize_columns(&mut z);
                }
                PowerIterationNormalizer::None => {}
            }

            multiply_matrix_centered(m, &z, &mut y, false, &column_means);
            match power_iteration_normalizer {
                PowerIterationNormalizer::QR => {
                    let qr = y.qr();
                    y = qr.q();
                }
                PowerIterationNormalizer::LU => normalize_columns(&mut y),
                PowerIterationNormalizer::None => {}
            }
        }
    }

    let qr = y.qr();
    let y = qr.q();

    let mut b = DMatrix::<T>::zeros(y.ncols(), m_cols);
    multiply_transposed_by_matrix_centered(&y, m, &mut b, &column_means);
    let svd = b.svd(true, true);
    let u_b = svd
        .u
        .ok_or_else(|| SvdLibError::Las2Error("SVD U computation failed".to_string()))?;
    let singular_values = svd.singular_values;
    let vt = svd
        .v_t
        .ok_or_else(|| SvdLibError::Las2Error("SVD V_t computation failed".to_string()))?;

    let u = y.mul(&u_b);
    let actual_rank = target_rank.min(singular_values.len());

    let u_subset = u.columns(0, actual_rank);
    let s = convert_singular_values(
        <DVector<T>>::from(singular_values.rows(0, actual_rank)),
        actual_rank,
    );
    let vt_subset = vt.rows(0, actual_rank).into_owned();
    let u = u_subset.into_owned().into_ndarray2();
    let vt = vt_subset.into_ndarray2();
    Ok(SvdRec {
        d: actual_rank,
        u,
        s,
        vt,
        diagnostics: create_diagnostics(
            m,
            actual_rank,
            target_rank,
            n_power_iters,
            seed.unwrap_or(0) as u32,
        ),
    })
}

fn convert_singular_values<T: SvdFloat + ComplexField>(
    values: DVector<T::RealField>,
    size: usize,
) -> Array1<T> {
    let mut array = Array1::zeros(size);

    for i in 0..size {
        array[i] = T::from_real(values[i].clone());
    }

    array
}

fn compute_column_means<T, M>(m: &M) -> Option<DVector<T>>
where
    T: SvdFloat + RealField,
    M: SMat<T>,
{
    let m_rows = m.nrows();
    let m_cols = m.ncols();

    let mut means = DVector::zeros(m_cols);

    for j in 0..m_cols {
        let mut col_vec = vec![T::zero(); m_cols];
        let mut result_vec = vec![T::zero(); m_rows];

        col_vec[j] = T::one();

        m.svd_opa(&col_vec, &mut result_vec, false);

        let mut sum = T::zero();
        for &val in &result_vec {
            sum += val;
        }

        means[j] = sum / T::from_f64(m_rows as f64).unwrap();
    }

    Some(means)
}

fn create_diagnostics<T, M: SMat<T>>(
    a: &M,
    d: usize,
    target_rank: usize,
    power_iterations: usize,
    seed: u32,
) -> Diagnostics<T>
where
    T: SvdFloat,
{
    Diagnostics {
        non_zero: a.nnz(),
        dimensions: target_rank,
        iterations: power_iterations,
        transposed: false,
        lanczos_steps: 0, // we dont do that
        ritz_values_stabilized: d,
        significant_values: d,
        singular_values: d,
        end_interval: [T::from(-1e-30).unwrap(), T::from(1e-30).unwrap()],
        kappa: T::from(1e-6).unwrap(),
        random_seed: seed,
    }
}

fn normalize_columns<T: SvdFloat + RealField + Send + Sync>(matrix: &mut DMatrix<T>) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();

    if rows < PARALLEL_THRESHOLD_ROWS && cols < PARALLEL_THRESHOLD_COLS {
        for j in 0..cols {
            let mut norm = T::zero();

            // Calculate column norm
            for i in 0..rows {
                norm += ComplexField::powi(matrix[(i, j)], 2);
            }
            norm = ComplexField::sqrt(norm);

            if norm > T::from_f64(1e-10).unwrap() {
                let scale = T::one() / norm;
                for i in 0..rows {
                    matrix[(i, j)] *= scale;
                }
            }
        }
        return;
    }

    let norms: Vec<T> = (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut norm = T::zero();
            for i in 0..rows {
                let val = unsafe { *matrix.get_unchecked((i, j)) };
                norm += ComplexField::powi(val, 2);
            }
            ComplexField::sqrt(norm)
        })
        .collect();

    let scales: Vec<(usize, T)> = norms
        .into_iter()
        .enumerate()
        .filter_map(|(j, norm)| {
            if norm > T::from_f64(1e-10).unwrap() {
                Some((j, T::one() / norm))
            } else {
                None // Skip columns with too small norms
            }
        })
        .collect();

    scales.iter().for_each(|(j, scale)| {
        for i in 0..rows {
            let value = matrix.get_mut((i, *j)).unwrap();
            *value = value.clone() * scale.clone();
        }
    });
}

// ----------------------------------------
// Utils Functions
// ----------------------------------------

fn generate_random_matrix<T: SvdFloat + RealField>(
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> DMatrix<T> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(0),
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    DMatrix::from_fn(rows, cols, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    })
}

fn multiply_matrix<T: SvdFloat, M: SMat<T>>(
    sparse: &M,
    dense: &DMatrix<T>,
    result: &mut DMatrix<T>,
    transpose_sparse: bool,
) {
    let cols = dense.ncols();

    let results: Vec<(usize, Vec<T>)> = (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut col_vec = vec![T::zero(); dense.nrows()];
            let mut result_vec = vec![T::zero(); result.nrows()];

            for i in 0..dense.nrows() {
                col_vec[i] = dense[(i, j)];
            }

            sparse.svd_opa(&col_vec, &mut result_vec, transpose_sparse);

            (j, result_vec)
        })
        .collect();

    for (j, col_result) in results {
        for i in 0..result.nrows() {
            result[(i, j)] = col_result[i];
        }
    }
}

fn multiply_transposed_by_matrix<T: SvdFloat, M: SMat<T>>(
    q: &DMatrix<T>,
    sparse: &M,
    result: &mut DMatrix<T>,
) {
    let q_rows = q.nrows();
    let q_cols = q.ncols();
    let sparse_rows = sparse.nrows();
    let sparse_cols = sparse.ncols();

    eprintln!("Q dimensions: {} x {}", q_rows, q_cols);
    eprintln!("Sparse dimensions: {} x {}", sparse_rows, sparse_cols);
    eprintln!("Result dimensions: {} x {}", result.nrows(), result.ncols());

    assert_eq!(
        q_rows, sparse_rows,
        "Dimension mismatch: Q has {} rows but sparse has {} rows",
        q_rows, sparse_rows
    );

    assert_eq!(
        result.nrows(),
        q_cols,
        "Result matrix has incorrect row count: expected {}, got {}",
        q_cols,
        result.nrows()
    );
    assert_eq!(
        result.ncols(),
        sparse_cols,
        "Result matrix has incorrect column count: expected {}, got {}",
        sparse_cols,
        result.ncols()
    );

    let chunk_size = determine_chunk_size(q_cols);

    let chunk_results: Vec<Vec<(usize, Vec<T>)>> = (0..q_cols)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_results = Vec::with_capacity(chunk.len());

            for &col_idx in &chunk {
                let mut q_col = vec![T::zero(); q_rows];
                for i in 0..q_rows {
                    q_col[i] = q[(i, col_idx)];
                }

                let mut result_row = vec![T::zero(); sparse_cols];

                sparse.svd_opa(&q_col, &mut result_row, true);

                chunk_results.push((col_idx, result_row));
            }
            chunk_results
        })
        .collect();

    for chunk_result in chunk_results {
        for (row_idx, row_values) in chunk_result {
            for j in 0..sparse_cols {
                result[(row_idx, j)] = row_values[j];
            }
        }
    }
}

pub fn svd_flip<T: SvdFloat + 'static>(
    u: Option<&mut DMatrix<T>>,
    v: Option<&mut DMatrix<T>>,
    u_based_decision: bool,
) -> Result<(), SvdLibError> {
    if u.is_none() && v.is_none() {
        return Err(SvdLibError::Las2Error(
            "Both u and v cannot be None".to_string(),
        ));
    }

    if u_based_decision {
        if u.is_none() {
            return Err(SvdLibError::Las2Error(
                "u cannot be None when u_based_decision is true".to_string(),
            ));
        }

        let u = u.unwrap();
        let ncols = u.ncols();
        let nrows = u.nrows();

        let mut signs = DVector::from_element(ncols, T::one());

        for j in 0..ncols {
            let mut max_abs = T::zero();
            let mut max_idx = 0;

            for i in 0..nrows {
                let abs_val = u[(i, j)].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                    max_idx = i;
                }
            }

            if u[(max_idx, j)] < T::zero() {
                signs[j] = -T::one();
            }
        }

        for j in 0..ncols {
            for i in 0..nrows {
                u[(i, j)] *= signs[j];
            }
        }

        if let Some(v) = v {
            let v_nrows = v.nrows();
            let v_ncols = v.ncols();

            for i in 0..v_nrows.min(signs.len()) {
                for j in 0..v_ncols {
                    v[(i, j)] *= signs[i];
                }
            }
        }
    } else {
        if v.is_none() {
            return Err(SvdLibError::Las2Error(
                "v cannot be None when u_based_decision is false".to_string(),
            ));
        }

        let v = v.unwrap();
        let nrows = v.nrows();
        let ncols = v.ncols();

        let mut signs = DVector::from_element(nrows, T::one());

        for i in 0..nrows {
            let mut max_abs = T::zero();
            let mut max_idx = 0;

            for j in 0..ncols {
                let abs_val = v[(i, j)].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                    max_idx = j;
                }
            }

            if v[(i, max_idx)] < T::zero() {
                signs[i] = -T::one();
            }
        }

        for i in 0..nrows {
            for j in 0..ncols {
                v[(i, j)] *= signs[i];
            }
        }

        if let Some(u) = u {
            let u_nrows = u.nrows();
            let u_ncols = u.ncols();

            for j in 0..u_ncols.min(signs.len()) {
                for i in 0..u_nrows {
                    u[(i, j)] *= signs[j];
                }
            }
        }
    }

    Ok(())
}

fn multiply_matrix_centered<T: SvdFloat, M: SMat<T>>(
    sparse: &M,
    dense: &DMatrix<T>,
    result: &mut DMatrix<T>,
    transpose_sparse: bool,
    column_means: &Option<DVector<T>>,
) {
    if column_means.is_none() {
        multiply_matrix(sparse, dense, result, transpose_sparse);
        return;
    }

    let means = column_means.as_ref().unwrap();
    let cols = dense.ncols();

    let results: Vec<(usize, Vec<T>)> = (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut col_vec = vec![T::zero(); dense.nrows()];
            let mut result_vec = vec![T::zero(); result.nrows()];

            for i in 0..dense.nrows() {
                col_vec[i] = dense[(i, j)];
            }

            sparse.svd_opa(&col_vec, &mut result_vec, transpose_sparse);

            if !transpose_sparse {
                let mut dot_product = T::zero();
                for &val in &col_vec {
                    dot_product += val;
                }

                for i in 0..result_vec.len() {
                    for (j, &mean) in means.iter().enumerate() {
                        if !transpose_sparse {
                            result_vec[i] -= mean * dot_product;
                        }
                    }
                }
            } else {
                let mut sum_x = T::zero();
                for &val in &col_vec {
                    sum_x += val;
                }

                for (i, mean) in means.iter().enumerate() {
                    result_vec[i] -= *mean * sum_x;
                }
            }

            (j, result_vec)
        })
        .collect();

    for (j, col_result) in results {
        for i in 0..result.nrows() {
            result[(i, j)] = col_result[i];
        }
    }
}

fn multiply_transposed_by_matrix_centered<T: SvdFloat, M: SMat<T>>(
    q: &DMatrix<T>,
    sparse: &M,
    result: &mut DMatrix<T>,
    column_means: &Option<DVector<T>>,
) {
    if column_means.is_none() {
        multiply_transposed_by_matrix(q, sparse, result);
        return;
    }

    let means = column_means.as_ref().unwrap();
    let q_rows = q.nrows();
    let q_cols = q.ncols();
    let sparse_rows = sparse.nrows();
    let sparse_cols = sparse.ncols();

    assert_eq!(
        q_rows, sparse_rows,
        "Dimension mismatch: Q has {} rows but sparse has {} rows",
        q_rows, sparse_rows
    );

    assert_eq!(
        result.nrows(),
        q_cols,
        "Result matrix has incorrect row count: expected {}, got {}",
        q_cols,
        result.nrows()
    );
    assert_eq!(
        result.ncols(),
        sparse_cols,
        "Result matrix has incorrect column count: expected {}, got {}",
        sparse_cols,
        result.ncols()
    );

    let chunk_size = determine_chunk_size(q_cols);

    let chunk_results: Vec<Vec<(usize, Vec<T>)>> = (0..q_cols)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_results = Vec::with_capacity(chunk.len());

            for &col_idx in &chunk {
                let mut q_col = vec![T::zero(); q_rows];
                for i in 0..q_rows {
                    q_col[i] = q[(i, col_idx)];
                }

                let mut result_row = vec![T::zero(); sparse_cols];

                sparse.svd_opa(&q_col, &mut result_row, true);

                let mut q_sum = T::zero();
                for &val in &q_col {
                    q_sum += val;
                }

                for j in 0..sparse_cols {
                    result_row[j] -= means[j] * q_sum;
                }

                chunk_results.push((col_idx, result_row));
            }
            chunk_results
        })
        .collect();

    for chunk_result in chunk_results {
        for (row_idx, row_values) in chunk_result {
            for j in 0..sparse_cols {
                result[(row_idx, j)] = row_values[j];
            }
        }
    }
}

#[cfg(test)]
mod randomized_svd_tests {
    use super::*;
    use crate::randomized::{randomized_svd, PowerIterationNormalizer};
    use nalgebra_sparse::coo::CooMatrix;
    use nalgebra_sparse::CsrMatrix;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rayon::ThreadPoolBuilder;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn setup_thread_pool() {
        INIT.call_once(|| {
            ThreadPoolBuilder::new()
                .num_threads(16)
                .build_global()
                .expect("Failed to build global thread pool");

            println!("Initialized thread pool with {} threads", 16);
        });
    }

    fn create_sparse_matrix(
        rows: usize,
        cols: usize,
        density: f64,
    ) -> nalgebra_sparse::coo::CooMatrix<f64> {
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
                    if v.abs() > 1e-10 {
                        break v;
                    }
                };

                coo.push(i, j, val);
            }
        }

        let actual_density = coo.nnz() as f64 / (rows as f64 * cols as f64);
        println!("Created sparse matrix: {} x {}", rows, cols);
        println!("  - Requested density: {:.6}", density);
        println!("  - Actual density: {:.6}", actual_density);
        println!("  - Sparsity: {:.4}%", (1.0 - actual_density) * 100.0);
        println!("  - Non-zeros: {}", coo.nnz());

        coo
    }

    #[test]
    fn test_randomized_svd_accuracy() {
        setup_thread_pool();

        let mut coo = CooMatrix::<f64>::new(20, 15);

        for i in 0..20 {
            for j in 0..5 {
                let val = (i as f64) * 0.5 + (j as f64) * 2.0;
                coo.push(i, j, val);
            }
        }

        let csr = CsrMatrix::from(&coo);

        let mut std_svd = crate::lanczos::svd_dim(&csr, 10).unwrap();

        let rand_svd = randomized_svd(
            &csr,
            10,
            5,
            2,
            PowerIterationNormalizer::QR,
            false,
            Some(42),
        )
        .unwrap();

        assert_eq!(rand_svd.d, 10, "Expected rank of 10");

        let rel_tol = 0.3;
        let compare_count = std::cmp::min(2, std::cmp::min(std_svd.d, rand_svd.d));
        println!("Standard SVD has {} dimensions", std_svd.d);
        println!("Randomized SVD has {} dimensions", rand_svd.d);

        for i in 0..compare_count {
            let rel_diff = (std_svd.s[i] - rand_svd.s[i]).abs() / std_svd.s[i];
            println!(
                "Singular value {}: standard={}, randomized={}, rel_diff={}",
                i, std_svd.s[i], rand_svd.s[i], rel_diff
            );
            assert!(
                rel_diff < rel_tol,
                "Dominant singular value {} differs too much: rel diff = {}, standard = {}, randomized = {}",
                i, rel_diff, std_svd.s[i], rand_svd.s[i]
            );
        }

        std_svd.u = std_svd.u.t().into_owned();
        let std_recon = std_svd.recompose();
        let rand_recon = rand_svd.recompose();

        let mut diff_norm = 0.0;
        let mut orig_norm = 0.0;

        for i in 0..20 {
            for j in 0..15 {
                diff_norm += (std_recon[[i, j]] - rand_recon[[i, j]]).powi(2);
                orig_norm += std_recon[[i, j]].powi(2);
            }
        }

        diff_norm = diff_norm.sqrt();
        orig_norm = orig_norm.sqrt();

        let rel_error = diff_norm / orig_norm;
        assert!(
            rel_error < 0.2,
            "Reconstruction difference too large: {}",
            rel_error
        );
    }

    // Test with mean centering
    #[test]
    fn test_randomized_svd_with_mean_centering() {
        setup_thread_pool();

        let mut coo = CooMatrix::<f64>::new(30, 10);
        let mut rng = StdRng::seed_from_u64(123);

        let column_means: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();

        let mut u = vec![vec![0.0; 3]; 30]; // 3 factors
        let mut v = vec![vec![0.0; 3]; 10];

        for i in 0..30 {
            for j in 0..3 {
                u[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        for i in 0..10 {
            for j in 0..3 {
                v[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        for i in 0..30 {
            for j in 0..10 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += u[i][k] * v[j][k];
                }
                val = val + column_means[j] + rng.gen_range(-0.1..0.1);
                coo.push(i, j, val);
            }
        }

        let csr = CsrMatrix::from(&coo);

        let svd_no_center =
            randomized_svd(&csr, 3, 3, 2, PowerIterationNormalizer::QR, false, Some(42)).unwrap();

        let svd_with_center =
            randomized_svd(&csr, 3, 3, 2, PowerIterationNormalizer::QR, true, Some(42)).unwrap();

        println!("Singular values without centering: {:?}", svd_no_center.s);
        println!("Singular values with centering: {:?}", svd_with_center.s);
    }

    #[test]
    fn test_randomized_svd_large_sparse() {
        setup_thread_pool();

        let test_matrix = create_sparse_matrix(5000, 1000, 0.01);

        let csr = CsrMatrix::from(&test_matrix);

        let result = randomized_svd(
            &csr,
            20,
            10,
            2,
            PowerIterationNormalizer::QR,
            false,
            Some(42),
        );

        assert!(
            result.is_ok(),
            "Randomized SVD failed on large sparse matrix: {:?}",
            result.err().unwrap()
        );

        let svd = result.unwrap();
        assert_eq!(svd.d, 20, "Expected rank of 20");
        assert_eq!(svd.u.ncols(), 20, "Expected 20 left singular vectors");
        assert_eq!(svd.u.nrows(), 5000, "Expected 5000 columns in U transpose");
        assert_eq!(svd.vt.nrows(), 20, "Expected 20 right singular vectors");
        assert_eq!(svd.vt.ncols(), 1000, "Expected 1000 columns in V transpose");

        for i in 1..svd.s.len() {
            assert!(svd.s[i] > 0.0, "Singular values should be positive");
            assert!(
                svd.s[i - 1] >= svd.s[i],
                "Singular values should be in descending order"
            );
        }
    }

    // Test with different power iteration settings
    #[test]
    fn test_power_iteration_impact() {
        setup_thread_pool();

        let mut coo = CooMatrix::<f64>::new(100, 50);
        let mut rng = StdRng::seed_from_u64(987);

        let mut u = vec![vec![0.0; 10]; 100];
        let mut v = vec![vec![0.0; 10]; 50];

        for i in 0..100 {
            for j in 0..10 {
                u[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        for i in 0..50 {
            for j in 0..10 {
                v[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        for i in 0..100 {
            for j in 0..50 {
                let mut val = 0.0;
                for k in 0..10 {
                    val += u[i][k] * v[j][k];
                }
                val += rng.gen_range(-0.01..0.01);
                coo.push(i, j, val);
            }
        }

        let csr = CsrMatrix::from(&coo);

        let powers = [0, 1, 3, 5];
        let mut errors = Vec::new();

        let mut dense_mat = Array2::<f64>::zeros((100, 50));
        for (i, j, val) in csr.triplet_iter() {
            dense_mat[[i, j]] = *val;
        }
        let matrix_norm = dense_mat.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        for &power in &powers {
            let svd = randomized_svd(
                &csr,
                10,
                5,
                power,
                PowerIterationNormalizer::QR,
                false,
                Some(42),
            )
            .unwrap();

            let recon = svd.recompose();
            let mut error = 0.0;

            for i in 0..100 {
                for j in 0..50 {
                    error += (dense_mat[[i, j]] - recon[[i, j]]).powi(2);
                }
            }

            error = error.sqrt() / matrix_norm;
            errors.push(error);

            println!("Power iterations: {}, Relative error: {}", power, error);
        }

        let mut improved = false;
        for i in 1..errors.len() {
            if errors[i] < errors[0] * 0.9 {
                improved = true;
                break;
            }
        }

        assert!(
            improved,
            "Power iterations did not improve accuracy as expected"
        );
    }
}
