use crate::error::SvdLibError;
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
use crate::utils::determine_chunk_size;

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
    seed: Option<u64>,
) -> anyhow::Result<SvdRec<T>>
where
    T: SvdFloat + RealField,
    M: SMat<T>,
    T: ComplexField,
{
    let start = std::time::Instant::now(); // only for debugging
    let m_rows = m.nrows();
    let m_cols = m.ncols();

    let rank = target_rank.min(m_rows.min(m_cols));
    let l = rank + n_oversamples;
    println!("Basic statistics: {:?}", start.elapsed());

    let omega = generate_random_matrix(m_cols, l, seed);
    println!("Generated Random Matrix here: {:?}", start.elapsed());

    let mut y = DMatrix::<T>::zeros(m_rows, l);
    multiply_matrix(m, &omega, &mut y, false);
    println!(
        "First multiplication took: {:?}, Continuing for power iterations:",
        start.elapsed()
    );

    if n_power_iters > 0 {
        let mut z = DMatrix::<T>::zeros(m_cols, l);

        for w in 0..n_power_iters {
            multiply_matrix(m, &y, &mut z, true);
            println!("{}-nd power-iteration forward: {:?}", w, start.elapsed());
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
            println!(
                "{}-nd power-iteration forward, normalization: {:?}",
                w,
                start.elapsed()
            );

            multiply_matrix(m, &z, &mut y, false);
            println!("{}-nd power-iteration backward: {:?}", w, start.elapsed());
            match power_iteration_normalizer {
                PowerIterationNormalizer::QR => {
                    let qr = y.qr();
                    y = qr.q();
                }
                PowerIterationNormalizer::LU => normalize_columns(&mut y),
                PowerIterationNormalizer::None => {}
            }
            println!(
                "{}-nd power-iteration backward, normalization: {:?}",
                w,
                start.elapsed()
            );
        }
    }
    println!(
        "Finished power-iteration, continuing QR: {:?}",
        start.elapsed()
    );
    let qr = y.qr();
    println!("QR finished: {:?}", start.elapsed());
    let q = qr.q();

    let mut b = DMatrix::<T>::zeros(q.ncols(), m_cols);
    multiply_transposed_by_matrix(&q, m, &mut b);
    println!(
        "QMB matrix multiplication transposed: {:?}",
        start.elapsed()
    );

    let svd = b.svd(true, true);
    println!("SVD decomposition took: {:?}", start.elapsed());
    let u_b = svd
        .u
        .ok_or_else(|| SvdLibError::Las2Error("SVD U computation failed".to_string()))?;
    let singular_values = svd.singular_values;
    let vt = svd
        .v_t
        .ok_or_else(|| SvdLibError::Las2Error("SVD V_t computation failed".to_string()))?;

    let actual_rank = target_rank.min(singular_values.len());

    let u_b_subset = u_b.columns(0, actual_rank);
    let u = q.mul(&u_b_subset);

    let vt_subset = vt.rows(0, actual_rank).into_owned();

    // Convert to the format required by SvdRec
    let d = actual_rank;
    println!("SVD Result Cropping: {:?}", start.elapsed());

    let ut = u.transpose().into_ndarray2();
    let s = convert_singular_values(
        <DVector<T>>::from(singular_values.rows(0, actual_rank)),
        actual_rank,
    );
    let vt = vt_subset.into_ndarray2();
    println!("Translation to ndarray: {:?}", start.elapsed());

    Ok(SvdRec {
        d,
        ut,
        s,
        vt,
        diagnostics: create_diagnostics(m, d, target_rank, n_power_iters, seed.unwrap_or(0) as u32),
    })
}

fn convert_singular_values<T: SvdFloat + ComplexField>(
    values: DVector<T::RealField>,
    size: usize,
) -> Array1<T> {
    let mut array = Array1::zeros(size);

    for i in 0..size {
        // Convert from RealField to T using f64 as intermediate
        array[i] = T::from_real(values[i].clone());
    }

    array
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

    // Use sequential processing for small matrices
    if rows < PARALLEL_THRESHOLD_ROWS && cols < PARALLEL_THRESHOLD_COLS {
        for j in 0..cols {
            let mut norm = T::zero();

            // Calculate column norm
            for i in 0..rows {
                norm += ComplexField::powi(matrix[(i, j)], 2);
            }
            norm = ComplexField::sqrt(norm);

            // Normalize the column if the norm is not too small
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

    // Now create a vector of (column_index, scale) pairs
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

    // Apply normalization
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
