//! Fixtures shared by the benchmark targets.
//!
//! Each bench target compiles its own copy, so not every builder is used by both.
#![allow(dead_code)]

use single_svdlib::SvdMat;
use sprs::CsMatI;

/// Self-contained LCG so fixtures don't shift with `rand` versions or across machines.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(s: u64) -> Self {
        Lcg(s
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
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() % (1 << 53)) as f64 / (1u64 << 53) as f64
    }
    pub fn signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
    pub fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Cells by genes with latent type structure, so the spectrum decays like real data's
/// instead of staying flat.
pub fn counts(rows: usize, cols: usize, per_row: usize, types: usize, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);
    let block = (cols / types.max(1)).max(1);
    let mut indptr: Vec<u64> = Vec::with_capacity(rows + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(rows * per_row);
    let mut data: Vec<f64> = Vec::with_capacity(rows * per_row);
    let mut row: Vec<(u32, f64)> = Vec::with_capacity(per_row);

    indptr.push(0);
    for _ in 0..rows {
        let ty = rng.range(types.max(1));
        row.clear();
        for _ in 0..per_row {
            let (g, w) = if rng.next_f64() < 0.7 {
                (ty * block + rng.range(block), 8.0)
            } else {
                (rng.range(cols), 1.0)
            };
            row.push((g as u32, (rng.next_f64() * w).floor() + 1.0));
        }
        row.sort_unstable_by_key(|&(g, _)| g);
        row.dedup_by_key(|&mut (g, _)| g);
        for &(g, v) in &row {
            indices.push(g);
            data.push(v);
        }
        indptr.push(indices.len() as u64);
    }
    CsMatI::new((rows, cols), indptr, indices, data)
}

/// A dense tall-skinny block — what TSQR and reorthogonalization see.
pub fn block(rows: usize, cols: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut rng = Lcg::new(seed);
    ndarray::Array2::from_shape_fn((rows, cols), |_| rng.signed())
}

/// Every `stride`-th column, the "highly variable gene" selection pattern.
pub fn every_nth(cols: usize, stride: usize) -> Vec<usize> {
    (0..cols).step_by(stride.max(1)).collect()
}
