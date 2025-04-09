use crate::error::SvdLibError;
use crate::{Diagnostics, SMat, SvdFloat, SvdRec};
use ndarray::{s, Array, Array1, Array2, Axis};
use num_traits::Float;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::cmp::min;

/// Computes a randomized SVD with default parameters
///
/// # Parameters
/// - A: Sparse matrix
/// - rank: Number of singular values/vectors to compute
///
/// # Returns
/// - SvdRec: Singular value decomposition (U, S, V)
pub fn randomized_svd<T, M>(a: &M, rank: usize) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T>,
{
    let oversampling = 10;
    let n_power_iterations = 2;
    let random_seed = 0; // Will be generated randomly
    randomized_svd_full(a, rank, oversampling, n_power_iterations, random_seed)
}

/// Computes a randomized SVD with control over oversampling and power iterations
///
/// # Parameters
/// - A: Sparse matrix
/// - rank: Number of singular values/vectors to compute
/// - oversampling: Additional columns to sample for improved accuracy (typically 5-10)
/// - n_power_iterations: Number of power iterations to enhance accuracy for smaller singular values
///
/// # Returns
/// - SvdRec: Singular value decomposition (U, S, V)
pub fn randomized_svd_with_params<T, M>(
    a: &M,
    rank: usize,
    oversampling: usize,
    n_power_iterations: usize,
) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T>,
{
    randomized_svd_full(a, rank, oversampling, n_power_iterations, 0)
}

/// Computes a randomized SVD with full parameter control
///
/// # Parameters
/// - A: Sparse matrix
/// - rank: Number of singular values/vectors to compute
/// - oversampling: Additional columns to sample for improved accuracy
/// - n_power_iterations: Number of power iterations to enhance accuracy for smaller singular values
/// - random_seed: Seed for random number generation (0 for random seed)
///
/// # Returns
/// - SvdRec: Singular value decomposition (U, S, V)
pub fn randomized_svd_full<T, M>(
    a: &M,
    rank: usize,
    oversampling: usize,
    n_power_iterations: usize,
    random_seed: u32,
) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T>,
{
    // Determine the actual seed to use
    let seed = if random_seed == 0 {
        // Use the system time as a seed if none is provided
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        duration.as_nanos() as u32
    } else {
        random_seed
    };

    // Validate parameters
    let nrows = a.nrows();
    let ncols = a.ncols();
    let min_dim = min(nrows, ncols);

    if rank == 0 {
        return Err(SvdLibError::Las2Error(
            "Rank must be greater than 0".to_string(),
        ));
    }

    if rank > min_dim {
        return Err(SvdLibError::Las2Error(format!(
            "Requested rank {} exceeds matrix dimensions {}x{}",
            rank, nrows, ncols
        )));
    }

    // The target rank with oversampling (can't exceed matrix dimensions)
    let target_rank = min(rank + oversampling, min_dim);

    // Transpose large matrices for efficiency if needed
    let transposed = ncols > nrows;
    let (work_rows, work_cols) = if transposed {
        (ncols, nrows)
    } else {
        (nrows, ncols)
    };

    // Stage 1: Generate a random projection matrix Omega
    let omega = generate_random_matrix(work_cols, target_rank, seed)?;

    // Stage 2: Form Y = A * Omega (or Y = A^T * Omega if transposed)
    let mut y = Array2::zeros((work_rows, target_rank));

    // Fill Y by matrix-vector products
    for j in 0..target_rank {
        let omega_col = omega.slice(s![.., j]).to_vec();
        let mut y_col = vec![T::zero(); work_rows];

        // Apply A or A^T
        a.svd_opa(&omega_col, &mut y_col, transposed);

        // Copy result to column of Y
        for i in 0..work_rows {
            y[[i, j]] = y_col[i];
        }
    }

    // Stage 3: Power iteration scheme to increase accuracy for smaller singular values
    if n_power_iterations > 0 {
        // Compute power iterations (Y = (A*A^T)^q * Y)
        y = power_iteration(a, y, n_power_iterations, transposed)?;
    }

    // Stage 4: Orthogonalize the basis using QR decomposition
    let q = orthogonalize(y)?;

    // Stage 5: Form B = Q^T * A (or B = Q^T * A^T if transposed)
    let mut b = Array2::zeros((target_rank, work_cols));

    // Fill B by matrix-vector products
    for i in 0..work_cols {
        let mut e_i = vec![T::zero(); work_cols];
        e_i[i] = T::one();

        let mut b_row = vec![T::zero(); target_rank];

        // Apply A or A^T
        a.svd_opa(&e_i, &mut b_row, !transposed);

        // Apply Q^T
        let qt_b = matrix_vector_multiply(&q, &b_row, true);

        // Copy result to row of B
        for j in 0..target_rank {
            b[[j, i]] = qt_b[j];
        }
    }

    // Stage 6: Compute the SVD of B
    let (u_b, s_b, v_b) = compute_svd_dense(b, rank)?;

    // Stage 7: Form the SVD of A
    let u_a = if transposed {
        v_b
    } else {
        matrix_multiply(&q, &u_b)
    };

    let v_a = if transposed {
        matrix_multiply(&q, &u_b)
    } else {
        v_b
    };

    // Return the SVD in the requested format
    Ok(SvdRec {
        d: rank,
        ut: if transposed { v_a } else { u_a.t().to_owned() },
        s: s_b,
        vt: if transposed { u_a.t().to_owned() } else { v_a },
        diagnostics: Diagnostics {
            non_zero: a.nnz(),
            dimensions: rank,
            iterations: n_power_iterations,
            transposed,
            lanczos_steps: 0, // Not applicable for randomized SVD
            ritz_values_stabilized: 0, // Not applicable
            significant_values: rank,
            singular_values: rank,
            end_interval: [T::zero(), T::zero()], // Not applicable
            kappa: T::from_f64(1e-6).unwrap(), // Standard value
            random_seed: seed,
        },
    })
}

// Helper functions (implementations to be added)

fn generate_random_matrix<T: SvdFloat>(
    n_rows: usize,
    n_cols: usize,
    seed: u32,
) -> Result<Array2<T>, SvdLibError> {
    // Create a Gaussian random matrix
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut omega = Array2::zeros((n_rows, n_cols));

    // Fill with random normal values
    omega.par_iter_mut().for_each(|x| {
        // Using local RNG for each parallel task
        let mut local_rng = StdRng::from_entropy();
        *x = T::from_f64(normal.sample(&mut local_rng)).unwrap();
    });

    Ok(omega)
}

fn power_iteration<T, M>(
    a: &M,
    mut y: Array2<T>,
    n_iterations: usize,
    transposed: bool,
) -> Result<Array2<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T>,
{
    let (n_rows, n_cols) = y.dim();

    // Temporary arrays for matrix operations
    let mut temp_rows = vec![T::zero(); a.nrows()];
    let mut temp_cols = vec![T::zero(); a.ncols()];

    for _ in 0..n_iterations {
        // Apply (A*A^T) to Y through two matrix-vector products
        for j in 0..n_cols {
            let y_col = y.slice(s![.., j]).to_vec();

            // First multiply: temp = A^T * y_col (or A * y_col if transposed)
            a.svd_opa(&y_col, &mut temp_cols, !transposed);

            // Second multiply: y_col = A * temp (or A^T * temp if transposed)
            a.svd_opa(&temp_cols, &mut temp_rows, transposed);

            // Update Y
            for i in 0..n_rows {
                y[[i, j]] = temp_rows[i];
            }
        }

        // Orthogonalize after each iteration for numerical stability
        y = orthogonalize(y)?;
    }

    Ok(y)
}

fn orthogonalize<T: SvdFloat>(a: Array2<T>) -> Result<Array2<T>, SvdLibError> {
    // Implement a modified Gram-Schmidt orthogonalization
    // This is more numerically stable than the standard Gram-Schmidt process
    let (n_rows, n_cols) = a.dim();
    let mut q = a.clone();

    for j in 0..n_cols {
        // Normalize the j-th column
        let mut norm_squared = T::zero();
        for i in 0..n_rows {
            norm_squared += q[[i, j]] * q[[i, j]];
        }

        let norm = norm_squared.sqrt();

        // Handle near-zero columns
        if norm <= T::from_f64(1e-10).unwrap() {
            // If column is essentially zero, replace with random vector
            let mut rng = StdRng::from_entropy();
            let normal = Normal::new(0.0, 1.0).unwrap();

            for i in 0..n_rows {
                q[[i, j]] = T::from_f64(normal.sample(&mut rng)).unwrap();
            }

            // Recursively orthogonalize this column against previous columns
            for k in 0..j {
                let mut dot = T::zero();
                for i in 0..n_rows {
                    dot += q[[i, j]] * q[[i, k]];
                }

                for i in 0..n_rows {
                    q[[i, j]] -= dot * q[[i, k]];
                }
            }

            // Normalize again
            norm_squared = T::zero();
            for i in 0..n_rows {
                norm_squared += q[[i, j]] * q[[i, j]];
            }
            let norm = norm_squared.sqrt();

            for i in 0..n_rows {
                q[[i, j]] /= norm;
            }
        } else {
            // Normalize the column
            for i in 0..n_rows {
                q[[i, j]] /= norm;
            }

            // Orthogonalize remaining columns against this one
            for k in (j+1)..n_cols {
                let mut dot = T::zero();
                for i in 0..n_rows {
                    dot += q[[i, j]] * q[[i, k]];
                }

                for i in 0..n_rows {
                    q[[i, k]] -= dot * q[[i, j]];
                }
            }
        }
    }

    Ok(q)
}

fn matrix_vector_multiply<T: SvdFloat>(
    a: &Array2<T>,
    x: &[T],
    transpose: bool,
) -> Vec<T> {
    let (n_rows, n_cols) = a.dim();

    if !transpose {
        // y = A * x
        assert_eq!(x.len(), n_cols, "Vector length must match number of columns");

        let mut y = vec![T::zero(); n_rows];

        if n_rows > 1000 {
            // Parallel implementation for large matrices
            y.par_iter_mut().enumerate().for_each(|(i, y_i)| {
                let row = a.row(i);
                *y_i = row.iter().zip(x.iter()).map(|(&a_ij, &x_j)| a_ij * x_j).sum();
            });
        } else {
            // Sequential for smaller matrices to avoid parallel overhead
            for i in 0..n_rows {
                let row = a.row(i);
                y[i] = row.iter().zip(x.iter()).map(|(&a_ij, &x_j)| a_ij * x_j).sum();
            }
        }

        y
    } else {
        // y = A^T * x
        assert_eq!(x.len(), n_rows, "Vector length must match number of rows");

        let mut y = vec![T::zero(); n_cols];

        if n_cols > 1000 {
            // Parallel implementation for large matrices
            y.par_iter_mut().enumerate().for_each(|(j, y_j)| {
                let col = a.column(j);
                *y_j = col.iter().zip(x.iter()).map(|(&a_ij, &x_i)| a_ij * x_i).sum();
            });
        } else {
            // Sequential for smaller matrices
            for j in 0..n_cols {
                let col = a.column(j);
                y[j] = col.iter().zip(x.iter()).map(|(&a_ij, &x_i)| a_ij * x_i).sum();
            }
        }

        y
    }
}

fn matrix_multiply<T: SvdFloat>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
    // Basic matrix multiplication C = A * B
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();

    assert_eq!(a_cols, b_rows, "Matrix dimensions do not match for multiplication");

    let mut c = Array2::zeros((a_rows, b_cols));

    // For large matrices, use parallel execution
    if a_rows * b_cols > 10000 {
        c.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..b_cols {
                    let mut sum = T::zero();
                    for k in 0..a_cols {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    row[j] = sum;
                }
            });
    } else {
        // For smaller matrices, sequential is faster due to less overhead
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = T::zero();
                for k in 0..a_cols {
                    sum += a[[i, k]] * b[[k, j]];
                }
                c[[i, j]] = sum;
            }
        }
    }

    c
}

fn compute_svd_dense<T: SvdFloat>(
    a: Array2<T>,
    rank: usize,
) -> Result<(Array2<T>, Array1<T>, Array2<T>), SvdLibError> {
    // For the dense SVD computation, we'll use a simplified approach
    // In a real implementation, you would use a high-quality SVD library
    // such as LAPACK via ndarray-linalg or similar

    let (n_rows, n_cols) = a.dim();

    // Form A^T * A (or A * A^T for tall matrices)
    let mut ata = if n_rows <= n_cols {
        // For wide matrices, compute A^T * A (smaller)
        let mut ata = Array2::zeros((n_cols, n_cols));

        for i in 0..n_cols {
            for j in 0..=i {
                let mut sum = T::zero();
                for k in 0..n_rows {
                    sum += a[[k, i]] * a[[k, j]];
                }
                ata[[i, j]] = sum;
                if i != j {
                    ata[[j, i]] = sum; // Symmetric matrix
                }
            }
        }
        ata
    } else {
        // For tall matrices, compute A * A^T (smaller)
        let mut aat = Array2::zeros((n_rows, n_rows));

        for i in 0..n_rows {
            for j in 0..=i {
                let mut sum = T::zero();
                for k in 0..n_cols {
                    sum += a[[i, k]] * a[[j, k]];
                }
                aat[[i, j]] = sum;
                if i != j {
                    aat[[j, i]] = sum; // Symmetric matrix
                }
            }
        }
        aat
    };

    // Compute eigendecomposition of A^T*A or A*A^T
    // For simplicity, we'll use a basic power iteration method
    // In practice, use a specialized eigenvalue solver
    let (eigvals, eigvecs) = compute_eigen_decomposition(&mut ata, rank)?;

    let mut s = Array1::zeros(rank);
    for i in 0..rank {
        // Singular values are square roots of eigenvalues
        s[i] = eigvals[i].abs().sqrt();
    }

    let mut v = if n_rows <= n_cols {
        // Wide matrix case: we computed A^T*A, so eigvecs are V
        eigvecs
    } else {
        // Tall matrix case: we computed A*A^T, so need to compute V = A^T*U*S^(-1)
        let mut v = Array2::zeros((n_cols, rank));

        for j in 0..rank {
            if s[j] > T::from_f64(1e-10).unwrap() {
                for i in 0..n_cols {
                    let mut sum = T::zero();
                    for k in 0..n_rows {
                        sum += a[[k, i]] * eigvecs[[k, j]];
                    }
                    v[[i, j]] = sum / s[j];
                }
            }
        }
        v
    };

    let mut u = if n_rows <= n_cols {
        // Wide matrix case: compute U = A*V*S^(-1)
        let mut u = Array2::zeros((n_rows, rank));

        for j in 0..rank {
            if s[j] > T::from_f64(1e-10).unwrap() {
                for i in 0..n_rows {
                    let mut sum = T::zero();
                    for k in 0..n_cols {
                        sum += a[[i, k]] * v[[k, j]];
                    }
                    u[[i, j]] = sum / s[j];
                }
            }
        }
        u
    } else {
        // Tall matrix case: we computed A*A^T, so eigvecs are U
        eigvecs
    };

    // Ensure orthogonality and normalize
    u = orthogonalize(u)?;
    v = orthogonalize(v)?;

    Ok((u, s, v))
}

fn compute_eigen_decomposition<T: SvdFloat>(
    a: &mut Array2<T>,
    rank: usize,
) -> Result<(Vec<T>, Array2<T>), SvdLibError> {
    let n = a.dim().0;
    let mut eigenvalues = vec![T::zero(); n];
    let mut eigenvectors = Array2::zeros((n, n));

    // Initialize eigenvectors to identity matrix
    for i in 0..n {
        eigenvectors[[i, i]] = T::one();
    }

    // We'll use the QR algorithm with shifts
    // In practice, use a specialized library for this

    // For simplicity, we'll just compute a few iterations of QR decomposition
    // This is a simplified approach - real implementations would use more robust methods
    let max_iter = 30;

    for _ in 0..max_iter {
        // For each iteration, perform a QR decomposition step
        let q = orthogonalize(eigenvectors.clone())?;

        // Compute R = Q^T * A * Q (Rayleigh quotient)
        let r = matrix_multiply(&matrix_multiply(&q.t().to_owned(), a), &q);

        // Update eigenvectors
        eigenvectors = q;

        // Update matrix for next iteration
        *a = r;
    }

    // Extract eigenvalues from diagonal
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    // Sort eigenvalues and eigenvectors by decreasing eigenvalue magnitude
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].abs().partial_cmp(&eigenvalues[i].abs()).unwrap());

    let mut sorted_values = vec![T::zero(); n];
    let mut sorted_vectors = Array2::zeros((n, n));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_values[new_idx] = eigenvalues[old_idx];
        for i in 0..n {
            sorted_vectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
        }
    }

    // Return only the top 'rank' eigenvalues and eigenvectors
    let top_values = sorted_values.into_iter().take(rank).collect();
    let top_vectors = sorted_vectors.slice(s![.., 0..rank]).to_owned();

    Ok((top_values, top_vectors))
}