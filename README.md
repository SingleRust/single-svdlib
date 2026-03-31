# Single-SVDLib: Singular Value Decomposition for Sparse Matrices

[![Crate](https://img.shields.io/crates/v/single-svdlib.svg)](https://crates.io/crates/single-svdlib)
[![Documentation](https://docs.rs/single-svdlib/badge.svg)](https://docs.rs/single-svdlib)
[![License](https://img.shields.io/crates/l/single-svdlib.svg)](LICENSE)

A high-performance Rust library for computing Singular Value Decomposition (SVD) on sparse matrices, with support for both Lanczos and randomized SVD algorithms.

## Features

- **Multiple SVD algorithms**:
    - Lanczos algorithm (LAS2 port of SVDLIBC)
    - Randomized SVD for very large and sparse matrices
- **Sparse matrix support**:
    - Native support for `nalgebra-sparse` (`CsrMatrix`, `CscMatrix`, `CooMatrix`)
    - Native support for `sprs` (`CsMatI`)
- **Memory & Performance optimizations**:
    - Parallel execution with Rayon across all sparse multiplication paths
    - Memory-efficient Lanczos subspace streaming (eliminates double-buffering)
    - Cache-optimized bidiagonal solver
    - Column masking for subspace SVD without data copying
- **Generic interface**:
    - Works with both `f32` and `f64` precision
- **Algebraic Consistency**:
    - Standardized output orientation across all algorithms: $A \approx U S V^T$
    - Built-in reconstruction via `svd.recompose()`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-svdlib = "2.0.0"
```

## Quick Start

```rust
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use single_svdlib::lanczos::svd_dim_seed;

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
let singular_values = &svd.s;      // Descending order [d]
let u = &svd.u;                    // Left singular vectors [M x d] (vectors as columns)
let vt = &svd.vt;                  // Transpose of right singular vectors [d x N] (vectors as rows)

// Reconstruct the original matrix: A ≈ U * S * VT
let reconstructed = svd.recompose();
```

## SVD Methods

### Lanczos Algorithm (LAS2)

The Lanczos algorithm is well-suited for sparse matrices of moderate size:

```rust
use single_svdlib::lanczos;

// Basic SVD computation (uses defaults)
let svd = lanczos::svd(&matrix)?;

// SVD with specified target rank
let svd = lanczos::svd_dim(&matrix, 10)?;

// SVD with specified target rank and fixed random seed
let svd = lanczos::svd_dim_seed(&matrix, 10, 42)?;
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
    false,                              // mean centering
)?;
```

## Result Structure

The SVD result `SvdRec<T>` is designed for maximum interoperability:

```rust
struct SvdRec<T> {
    d: usize,              // Rank (number of singular values found)
    u: Array2<T>,          // Left singular vectors (M x d)
    s: Array1<T>,          // Singular values (d), sorted descending
    vt: Array2<T>,         // Right singular vectors (d x N)
    diagnostics: Diagnostics<T>,
}
```

The orientations match standard linear algebra conventions ($A = U S V^T$):
- `u` stores singular vectors as **columns**.
- `vt` stores singular vectors as **rows**.

## Performance Tips

1. **Leverage Rayon**: The library automatically detects and uses the available Rayon thread pool for sparse matrix multiplications and singular vector combinations.
2. **Matrix Format**: Use `CsrMatrix` for row-major heavy operations. Both `nalgebra-sparse` and `sprs` are natively supported via the `SMat` trait.
3. **Memory Efficiency**: The Lanczos implementation has been optimized to stream vectors directly from storage, avoiding the $O(d \times N)$ memory spikes seen in standard ports.

## License

This crate is licensed under the BSD License, the same as the original SVDLIBC implementation. See the `SVDLIBC-LICENSE.txt` file for details.

## Credits

- Original SVDLIBC implementation by Doug Rohde
- Rust port maintainer of SVDLIBC: Dave Farnham
- Performance optimizations and algebraic fixes: Ian F. Diks
