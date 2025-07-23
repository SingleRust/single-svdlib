use rayon::iter::ParallelIterator;
use nalgebra_sparse::na::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use num_traits::{Float, Zero};
use rayon::prelude::{IntoParallelIterator, IndexedParallelIterator};
use single_utilities::traits::FloatOpsTS;
use std::fmt::Debug;

pub fn determine_chunk_size(nrows: usize) -> usize {
    let num_threads = rayon::current_num_threads();

    let min_rows_per_thread = 64;
    let desired_chunks_per_thread = 4;

    let target_total_chunks = num_threads * desired_chunks_per_thread;
    let chunk_size = nrows.div_ceil(target_total_chunks);

    chunk_size.max(min_rows_per_thread)
}

pub trait SMat<T: Float> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn nnz(&self) -> usize;
    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool); // y = A*x
    fn compute_column_means(&self) -> Vec<T>;
    fn multiply_with_dense(&self, dense: &DMatrix<T>, result: &mut DMatrix<T>, transpose_self: bool);
    fn multiply_with_dense_centered(&self, dense: &DMatrix<T>, result: &mut DMatrix<T>, transpose_self: bool, means: &DVector<T>);

    fn multiply_transposed_by_dense(&self, q: &DMatrix<T>, result: &mut DMatrix<T>);
    fn multiply_transposed_by_dense_centered(&self, q: &DMatrix<T>, result: &mut DMatrix<T>, means: &DVector<T>);
}

/// Singular Value Decomposition Components
///
/// # Fields
/// - d:  Dimensionality (rank), the number of rows of both `ut`, `vt` and the length of `s`
/// - ut: Transpose of left singular vectors, the vectors are the rows of `ut`
/// - s:  Singular values (length `d`)
/// - vt: Transpose of right singular vectors, the vectors are the rows of `vt`
/// - diagnostics: Computational diagnostics
#[derive(Debug, Clone, PartialEq)]
pub struct SvdRec<T: Float> {
    pub d: usize,
    pub u: Array2<T>,
    pub s: Array1<T>,
    pub vt: Array2<T>,
    pub diagnostics: Diagnostics<T>,
}

/// Computational Diagnostics
///
/// # Fields
/// - non_zero:  Number of non-zeros in the matrix
/// - dimensions: Number of dimensions attempted (bounded by matrix shape)
/// - iterations: Number of iterations attempted (bounded by dimensions and matrix shape)
/// - transposed:  True if the matrix was transposed internally
/// - lanczos_steps: Number of Lanczos steps performed
/// - ritz_values_stabilized: Number of ritz values
/// - significant_values: Number of significant values discovered
/// - singular_values: Number of singular values returned
/// - end_interval: left, right end of interval containing unwanted eigenvalues
/// - kappa: relative accuracy of ritz values acceptable as eigenvalues
/// - random_seed: Random seed provided or the seed generated
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics<T: Float> {
    pub non_zero: usize,
    pub dimensions: usize,
    pub iterations: usize,
    pub transposed: bool,
    pub lanczos_steps: usize,
    pub ritz_values_stabilized: usize,
    pub significant_values: usize,
    pub singular_values: usize,
    pub end_interval: [T; 2],
    pub kappa: T,
    pub random_seed: u32,
}

pub trait SvdFloat: FloatOpsTS {
    fn eps() -> Self;
    fn eps34() -> Self;
    fn compare(a: Self, b: Self) -> bool;
}

impl SvdFloat for f32 {
    fn eps() -> Self {
        f32::EPSILON
    }

    fn eps34() -> Self {
        f32::EPSILON.powf(0.75)
    }

    fn compare(a: Self, b: Self) -> bool {
        (b - a).abs() < f32::EPSILON
    }
}

impl SvdFloat for f64 {
    fn eps() -> Self {
        f64::EPSILON
    }

    fn eps34() -> Self {
        f64::EPSILON.powf(0.75)
    }

    fn compare(a: Self, b: Self) -> bool {
        (b - a).abs() < f64::EPSILON
    }
}
