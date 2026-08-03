//! Deterministic fixtures shared by the unit tests.
//!
//! The generators use a self-contained LCG so fixtures stay stable across `rand`
//! upgrades, and so the unit tests and the integration tests agree on what
//! `gen_sparse(200, 80, 0.1, 7)` means.

#![cfg(test)]
// Numeric kernels index several arrays in step from one loop variable, and
// offset arithmetic is load-bearing; iterator rewrites obscure which array an
// index belongs to.
#![allow(clippy::needless_range_loop)]

use crate::matrix::SvdMat;
use crate::types::SvdFloat;
use ndarray::Array2;
use sprs::{SpIndex, TriMatI};

/// Deterministic, dependency-free PRNG.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() % (1 << 53)) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[-1, 1)`.
    pub fn signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
    pub fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Densify a sparse matrix, for comparing against a reference computation.
pub fn dense_of<T, I, Iptr>(m: &SvdMat<T, I, Iptr>) -> Array2<T>
where
    T: SvdFloat,
    I: SpIndex,
    Iptr: SpIndex,
{
    let mut d = Array2::zeros((m.rows(), m.cols()));
    for (v, (i, j)) in m.iter() {
        d[[i.index(), j.index()]] = *v;
    }
    d
}

/// A sparse matrix with an exact non-zero count at the requested density.
pub fn gen_sparse(rows: usize, cols: usize, density: f64, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);
    let target = ((rows as f64 * cols as f64 * density).round() as usize).max(1);
    let mut seen = std::collections::HashSet::new();
    let mut t = TriMatI::<f64, u32>::new((rows, cols));
    let mut attempts = 0usize;
    while seen.len() < target && attempts < target * 100 {
        attempts += 1;
        let i = rng.range(rows);
        let j = rng.range(cols);
        if seen.insert((i, j)) {
            let v = rng.signed() * 10.0;
            t.add_triplet(i, j, if v.abs() < 1e-6 { 1.0 } else { v });
        }
    }
    t.to_csr()
}

/// A dense-ish matrix with a genuine low-rank structure and a decaying spectrum,
/// which is what exercises convergence behaviour.
pub fn gen_lowrank(rows: usize, cols: usize, rank: usize, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);
    let u: Vec<Vec<f64>> = (0..rows)
        .map(|_| (0..rank).map(|_| rng.signed()).collect())
        .collect();
    let v: Vec<Vec<f64>> = (0..cols)
        .map(|_| (0..rank).map(|_| rng.signed()).collect())
        .collect();
    let mut t = TriMatI::<f64, u32>::new((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let mut val = 0.0;
            for k in 0..rank {
                // 1/(k+1) weighting gives the singular values a real gap structure.
                val += u[i][k] * v[j][k] / (k as f64 + 1.0);
            }
            val += rng.signed() * 0.001;
            t.add_triplet(i, j, val);
        }
    }
    t.to_csr()
}

/// Reference singular values via a dense Jacobi SVD of `AᵀA`, independent of any
/// code under test.
pub fn reference_singular_values(a: &Array2<f64>) -> Vec<f64> {
    let m = nalgebra::DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]]);
    let mut s: Vec<f64> = m.singular_values().iter().copied().collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}
