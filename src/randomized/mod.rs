use rayon::iter::ParallelIterator;
use crate::error::SvdLibError;
use crate::{Diagnostics, SMat, SvdFloat, SvdRec};
use nalgebra_sparse::na::{ComplexField, DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use nshare::IntoNdarray2;
use rand::prelude::{Distribution, StdRng};
use rand::SeedableRng;
use rand_distr::Normal;
use std::ops::Mul;
use rayon::current_num_threads;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};

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
    let m_rows = m.nrows();
    let m_cols = m.ncols();

    let rank = target_rank.min(m_rows.min(m_cols));
    let l = rank + n_oversamples;

    let omega = generate_random_matrix(m_cols, l, seed);

    let mut y = DMatrix::<T>::zeros(m_rows, l);
    multiply_matrix(m, &omega, &mut y, false);

    if n_power_iters > 0 {
        let mut z = DMatrix::<T>::zeros(m_cols, l);

        for _ in 0..n_power_iters {
            multiply_matrix(m, &y, &mut z, true);
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

            multiply_matrix(m, &z, &mut y, false);
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
    let q = qr.q();

    let mut b = DMatrix::<T>::zeros(q.ncols(), m_cols);
    multiply_transposed_by_matrix(&q, m, &mut b);

    let svd = b.svd(true, true);
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

    let ut = u.transpose().into_ndarray2();
    let s = convert_singular_values(<DVector<T>>::from(singular_values.rows(0, actual_rank)), actual_rank);
    let vt = vt_subset.into_ndarray2();

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
    scales
        .iter()
        .for_each(|(j, scale)| {
            for i in 0..rows {
                let value = matrix.get_mut((i,*j)).unwrap();
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
    //if rows < PARALLEL_THRESHOLD_ROWS && cols < PARALLEL_THRESHOLD_COLS && rows * cols < PARALLEL_THRESHOLD_ELEMENTS {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(0),
        };

        let normal = Normal::new(0.0, 1.0).unwrap();
        return DMatrix::from_fn(rows, cols, |_, _| {
            T::from_f64(normal.sample(&mut rng)).unwrap()
        });
    //}

    /*let seed_value = seed.unwrap_or(0);
    let mut matrix = DMatrix::<T>::zeros(rows, cols);
    let num_threads = current_num_threads();
    let chunk_size = (rows * cols + num_threads - 1) / num_threads;

    (0..(rows * cols)).into_par_iter()
        .chunks(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, indices)| {
            let thread_seed = seed_value.wrapping_add(chunk_idx as u64);
            let mut rng = StdRng::seed_from_u64(thread_seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            for idx in indices {
                let i = idx / cols;
                let j = idx % cols;
                unsafe {
                    *matrix.get_unchecked_mut((i, j)) = T::from_f64(normal.sample(&mut rng)).unwrap();
                }
            }
        });
    matrix*/

}

fn multiply_matrix<T: SvdFloat, M: SMat<T>>(
    sparse: &M,
    dense: &DMatrix<T>,
    result: &mut DMatrix<T>,
    transpose_sparse: bool,
) {
    let cols = dense.ncols();
    //let matrix_rows = if transpose_sparse { sparse.ncols() } else { sparse.nrows() };

    //if matrix_rows < PARALLEL_THRESHOLD_ROWS && cols < PARALLEL_THRESHOLD_COLS {
        let mut col_vec = vec![T::zero(); dense.nrows()];
        let mut result_vec = vec![T::zero(); result.nrows()];

        for j in 0..cols {
            // Extract column from dense matrix
            for i in 0..dense.nrows() {
                col_vec[i] = dense[(i, j)];
            }

            // Perform sparse matrix operation
            sparse.svd_opa(&col_vec, &mut result_vec, transpose_sparse);

            // Store results
            for i in 0..result.nrows() {
                result[(i, j)] = result_vec[i];
            }

            // Clear result vector for reuse
            result_vec.iter_mut().for_each(|v| *v = T::zero());
        }
        return;
    //}


}

fn multiply_transposed_by_matrix<T: SvdFloat, M: SMat<T>>(
    q: &DMatrix<T>,
    sparse: &M,
    result: &mut DMatrix<T>,
) {
    for j in 0..sparse.ncols() {
        let mut unit_vec = vec![T::zero(); sparse.ncols()];
        unit_vec[j] = T::one();

        let mut col_vec = vec![T::zero(); sparse.nrows()];
        sparse.svd_opa(&unit_vec, &mut col_vec, false);

        for i in 0..q.ncols() {
            let mut sum = T::zero();
            for k in 0..q.nrows() {
                sum += q[(k, i)] * col_vec[k];
            }
            result[(i, j)] = sum;
        }
    }
}
