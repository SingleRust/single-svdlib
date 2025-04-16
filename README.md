# Single-SVDLib: Singular Value Decomposition for Sparse Matrices

[![Crate](https://img.shields.io/crates/v/single-svdlib.svg)](https://crates.io/crates/single-svdlib)
[![Documentation](https://docs.rs/single-svdlib/badge.svg)](https://docs.rs/single-svdlib)
[![License](https://img.shields.io/crates/l/single-svdlib.svg)](LICENSE)

A high-performance Rust library for computing Singular Value Decomposition (SVD) on sparse matrices, with support for both Lanczos and randomized SVD algorithms.

## Features

- **Multiple SVD algorithms**:
    - Lanczos algorithm (based on SVDLIBC)
    - Randomized SVD for very large and sparse matrices
- **Sparse matrix support**:
    - Compressed Sparse Row (CSR) format
    - Compressed Sparse Column (CSC) format
    - Coordinate (COO) format
- **Performance optimizations**:
    - Parallel execution with Rayon
    - Adaptive tuning for highly sparse matrices
    - Column masking for subspace SVD
- **Generic interface**:
    - Works with both `f32` and `f64` precision
- **Comprehensive error handling and diagnostics**

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-svdlib = "0.6.0"
```

## Quick Start

```rust
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use single_svdlib::laczos::svd_dim_seed;

// Create a matrix in COO format
let mut coo = CooMatrix::<f64>::new(3, 3);
coo.push(0, 0, 1.0); coo.push(0, 1, 16.0); coo.push(0, 2, 49.0);
coo.push(1, 0, 4.0); coo.push(1, 1, 25.0); coo.push(1, 2, 64.0);
coo.push(2, 0, 9.0); coo.push(2, 1, 36.0); coo.push(2, 2, 81.0);

// Convert to CSR for better performance
let csr = CsrMatrix::from(&coo);

// Compute SVD with a fixed random seed
let svd = svd_dim_seed(&csr, 3, 42).unwrap();

// Access the results
let singular_values = &svd.s;
let left_singular_vectors = &svd.ut;  // Note: These are transposed
let right_singular_vectors = &svd.vt; // Note: These are transposed

// Reconstruct the original matrix
let reconstructed = svd.recompose();
```

## SVD Methods

### Lanczos Algorithm (LAS2)

The Lanczos algorithm is well-suited for sparse matrices of moderate size:

```rust
use single_svdlib::laczos;

// Basic SVD computation (uses defaults)
let svd = laczos::svd(&matrix)?;

// SVD with specified target rank
let svd = laczos::svd_dim(&matrix, 10)?;

// SVD with specified target rank and fixed random seed
let svd = laczos::svd_dim_seed(&matrix, 10, 42)?;

// Full control over SVD parameters
let svd = laczos::svd_las2(
    &matrix,
    dimensions,    // upper limit of desired number of dimensions
    iterations,    // number of Lanczos iterations
    end_interval,  // interval containing unwanted eigenvalues, e.g. [-1e-30, 1e-30]
    kappa,         // relative accuracy of eigenvalues, e.g. 1e-6
    random_seed,   // random seed (0 for automatic)
)?;
```

### Randomized SVD

For very large sparse matrices, the randomized SVD algorithm offers better performance:

```rust
use single_svdlib::randomized;

let svd = randomized::randomized_svd(
    &matrix,
    target_rank,                         // desired rank
    n_oversamples,                       // oversampling parameter (typically 5-10)
    n_power_iterations,                  // number of power iterations (typically 2-4)
    randomized::PowerIterationNormalizer::QR,  // normalization method
    Some(42),                           // random seed (None for automatic)
)?;
```

### Column Masking

For operations on specific columns of a matrix:

```rust
use single_svdlib::laczos::masked::MaskedCSRMatrix;

// Create a mask for selected columns
let columns = vec![0, 2, 5, 7];  // Only use these columns
let masked_matrix = MaskedCSRMatrix::with_columns(&csr_matrix, &columns);

// Compute SVD on the masked matrix
let svd = laczos::svd(&masked_matrix)?;
```

## Result Structure

The SVD result contains:

```rust
struct SvdRec<T> {
    d: usize,              // Rank (number of singular values)
    ut: Array2<T>,         // Transpose of left singular vectors (d x m)
    s: Array1<T>,          // Singular values (d)
    vt: Array2<T>,         // Transpose of right singular vectors (d x n)
    diagnostics: Diagnostics<T>,  // Computation diagnostics
}
```

Note that `ut` and `vt` are returned in transposed form.

## Diagnostics

Each SVD computation returns detailed diagnostics:

```rust
let svd = laczos::svd(&matrix)?;
println!("Non-zero elements: {}", svd.diagnostics.non_zero);
println!("Transposed during computation: {}", svd.diagnostics.transposed);
println!("Lanczos steps: {}", svd.diagnostics.lanczos_steps);
println!("Significant values found: {}", svd.diagnostics.significant_values);
```

## Performance Tips

1. **Choose the right algorithm**:
    - For matrices up to ~10,000 x 10,000 with moderate sparsity, use the Lanczos algorithm
    - For larger matrices or very high sparsity (>99%), use randomized SVD

2. **Matrix format matters**:
    - Convert COO matrices to CSR or CSC for computation
    - CSR typically performs better for row-oriented operations

3. **Adjust parameters for very sparse matrices**:
    - Increase power iterations in randomized SVD (e.g., 5-7)
    - Use a higher `kappa` value in Lanczos for very sparse matrices

4. **Consider column masking** for operations that only need a subset of the data

## License

This crate is licensed under the BSD License, the same as the original SVDLIBC implementation. See the `SVDLIBC-LICENSE.txt` file for details.

## Credits

- Original SVDLIBC implementation by Doug Rohde
- Rust port maintainer of SVDLIBC: Dave Farnham
- Extensions and modifications of the original algorithm: Ian F. Diks
