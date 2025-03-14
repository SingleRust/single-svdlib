# single_svdlib

A Rust library for performing Singular Value Decomposition (SVD) on sparse matrices using the Lanczos algorithm. It is build on the original library and expan

## Overview

`svdlibrs` is a Rust port of LAS2 from SVDLIBC, originally developed by Doug Rohde. This library efficiently computes SVD on sparse matrices, particularly large ones, and returns the decomposition as ndarray components.

This implementation extends the original [svdlibrs](https://github.com/dfarnham/svdlibrs) by Dave Farnham with:
- Updated dependency versions
- Support for a broader range of numeric types (f64, f32, others)
- Column masking capabilities for analyzing specific subsets of data

## Features

- Performs SVD on sparse matrices using the Lanczos algorithm
- Works with various input formats: CSR, CSC, or COO matrices
- Column masking for dimension selection without data copying
- Generic implementation supporting different numeric types
- High numerical precision for critical calculations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-svdlib = "0.1.0"
nalgebra-sparse = "0.10.0"
ndarray = "0.16.1"
```

## Basic Usage

```rust
use single_svdlib::{svd, svd_dim, svd_dim_seed};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

// Create a sparse matrix
let mut coo = CooMatrix::<f64>::new(3, 3);
coo.push(0, 0, 1.0); coo.push(0, 1, 16.0); coo.push(0, 2, 49.0);
coo.push(1, 0, 4.0); coo.push(1, 1, 25.0); coo.push(1, 2, 64.0);
coo.push(2, 0, 9.0); coo.push(2, 1, 36.0); coo.push(2, 2, 81.0);

let csr = CsrMatrix::from(&coo);

// Compute SVD
let svd_result = svd(&csr)?;

// Access the results
println!("Rank: {}", svd_result.d);
println!("Singular values: {:?}", svd_result.s);
println!("Left singular vectors (U): {:?}", svd_result.ut.t());
println!("Right singular vectors (V): {:?}", svd_result.vt.t());

// Reconstruct the original matrix
let reconstructed = svd_result.recompose();
```

## Column Masking

The library supports analyzing specific columns without copying the data:

```rust
use single_svdlib::{svd, MaskedCSRMatrix};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

// Create a sparse matrix
let mut coo = CooMatrix::<f64>::new(3, 5);
coo.push(0, 0, 1.0); coo.push(0, 2, 2.0); coo.push(0, 4, 3.0);
coo.push(1, 1, 4.0); coo.push(1, 3, 5.0);
coo.push(2, 0, 6.0); coo.push(2, 2, 7.0); coo.push(2, 4, 8.0);

let csr = CsrMatrix::from(&coo);

// Method 1: Using a boolean mask (true = include column)
let mask = vec![true, false, true, false, true]; // Only columns 0, 2, 4
let masked_matrix = MaskedCSRMatrix::new(&csr, mask);

// Method 2: Specifying which columns to include
let columns = vec![0, 2, 4];
let masked_matrix = MaskedCSRMatrix::with_columns(&csr, &columns);

// Run SVD on the masked matrix
let svd_result = svd(&masked_matrix)?;
```

## Support for Different Numeric Types

The library supports various numeric types:

```rust
// With f64 (double precision)
let csr_f64 = CsrMatrix::<f64>::from(&coo);
let svd_result = svd(&csr_f64)?;

// With f32 (single precision)
let csr_f32 = CsrMatrix::<f32>::from(&coo);
let svd_result = svd(&csr_f32)?;

// With integer types (converted internally)
let csr_i32 = CsrMatrix::<i32>::from(&coo);
let masked_i32 = MaskedCSRMatrix::with_columns(&csr_i32, &columns);
let svd_result = svd(&masked_i32)?;
```

## Advanced Usage

For more control over the SVD computation:

```rust
use single_svdlib::{svdLAS2, SvdRec};

// Customize the SVD calculation
let svd: SvdRec = svdLAS2(
    &matrix,        // sparse matrix
    dimensions,     // upper limit of desired dimensions (0 = max)
    iterations,     // number of algorithm iterations (0 = auto)
    &[-1.0e-30, 1.0e-30], // interval for unwanted eigenvalues
    1.0e-6,         // relative accuracy threshold
    random_seed,    // random seed (0 = auto-generate)
)?;
```

## SVD Results and Diagnostics

The SVD results are returned in a `SvdRec` struct:

```rust
pub struct SvdRec {
    pub d: usize,        // Dimensionality (rank)
    pub ut: Array2<f64>, // Transpose of left singular vectors
    pub s: Array1<f64>,  // Singular values
    pub vt: Array2<f64>, // Transpose of right singular vectors
    pub diagnostics: Diagnostics, // Computational diagnostics
}
```

The `Diagnostics` struct provides detailed information about the computation:

```rust
pub struct Diagnostics {
    pub non_zero: usize,   // Number of non-zeros in the input matrix
    pub dimensions: usize, // Number of dimensions attempted
    pub iterations: usize, // Number of iterations attempted
    pub transposed: bool,  // True if the matrix was transposed internally
    pub lanczos_steps: usize,          // Number of Lanczos steps
    pub ritz_values_stabilized: usize, // Number of ritz values
    pub significant_values: usize,     // Number of significant values
    pub singular_values: usize,        // Number of singular values
    pub end_interval: [f64; 2], // Interval for unwanted eigenvalues
    pub kappa: f64,             // Relative accuracy threshold
    pub random_seed: u32,       // Random seed used
}
```

## License

This library is provided under the BSD License, as per the original SVDLIBC implementation.

## Acknowledgments

- Dave Farnham for the original Rust port
- Doug Rohde for the original SVDLIBC implementation
- University of Tennessee Research Foundation for the underlying mathematical library

[Latest Version]: https://img.shields.io/crates/v/single-svdlib.svg
[crates.io]: https://crates.io/crates/single-svdlib