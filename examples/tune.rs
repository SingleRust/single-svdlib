//! Which configuration to use for a large cells-by-genes PCA.
//!
//! Builds one matrix, then times every reasonable way to get the same components out of
//! it, checking each against the most accurate result. Run it on a shape resembling your
//! own data and use the table it prints.
//!
//! ```text
//! cargo run --release --example tune                       # 400k x 30k, 2k genes, 50 PCs
//! cargo run --release --example tune 1000000 30000 150 2000 50
//! ```

use single_svdlib::{irlba, randomized, MaskedCsMat, SparseMat, SparseMatDense, SvdMat, SvdRec};
use sprs::CsMatI;
use std::time::{Duration, Instant};

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg(s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % (1 << 53)) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Cells with a latent type structure, so the spectrum decays like real data's.
fn build(cells: usize, genes: usize, per_cell: usize, types: usize, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);
    let block = (genes / types.max(1)).max(1);
    let mut indptr: Vec<u64> = Vec::with_capacity(cells + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(cells * per_cell);
    let mut data: Vec<f64> = Vec::with_capacity(cells * per_cell);
    let mut row: Vec<(u32, f64)> = Vec::with_capacity(per_cell);

    indptr.push(0);
    for _ in 0..cells {
        let ty = rng.range(types.max(1));
        row.clear();
        for _ in 0..per_cell {
            let (g, w) = if rng.next_f64() < 0.7 {
                (ty * block + rng.range(block), 8.0)
            } else {
                (rng.range(genes), 1.0)
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
    CsMatI::new((cells, genes), indptr, indices, data)
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let cells = a.first().copied().unwrap_or(400_000);
    let genes = a.get(1).copied().unwrap_or(30_000);
    let per_cell = a.get(2).copied().unwrap_or(150);
    let n_sel = a.get(3).copied().unwrap_or(2_000);
    let k = a.get(4).copied().unwrap_or(50);

    println!("building {cells} x {genes}, ~{per_cell} nnz/cell ...");
    let matrix = build(cells, genes, per_cell, 24, 7);
    println!("  nnz = {}", matrix.nnz());

    let stride = (genes / n_sel).max(1);
    let mut sel: Vec<usize> = (0..genes).step_by(stride).collect();
    sel.extend(0..256);
    sel.sort_unstable();
    sel.dedup();

    let view = MaskedCsMat::with_columns(&matrix, &sel);
    let t = Instant::now();
    let sub = view.to_sparse();
    let extract = t.elapsed();
    println!(
        "  mask keeps {} genes, {} nnz ({:.1}%); extraction took {:?}\n",
        sel.len(),
        view.nnz(),
        100.0 * view.nnz() as f64 / matrix.nnz() as f64,
        extract
    );

    // The reference: IRLBA converges to a residual tolerance, so trust it.
    println!("computing a reference (irlba, tight tolerance) ...");
    let reference = irlba::svd_with(
        &sub,
        &irlba::IrlbaConfig::new(k)
            .seed(42)
            .tol(1e-12)
            .mean_center(true),
        Some(sub.col_means()),
    )
    .expect("reference failed");
    println!(
        "  sigma[0] = {:.2}, sigma[{}] = {:.2}\n",
        reference.s[0],
        k - 1,
        reference.s[k - 1]
    );

    let err = |r: &SvdRec<f64>| {
        (0..k)
            .map(|i| (r.s[i] - reference.s[i]).abs() / reference.s[0])
            .fold(0.0f64, f64::max)
    };

    let mut rows: Vec<(String, Duration, f64, usize)> = Vec::new();
    let mut run = |label: String, f: &dyn Fn() -> SvdRec<f64>| {
        let t = Instant::now();
        let r = f();
        let d = t.elapsed();
        println!("  {label:<34} {:>9.2?}  rel {:.2e}", d, err(&r));
        rows.push((label, d, err(&r), r.diagnostics.matvecs));
    };

    println!("on the masked VIEW (no copy):");
    run("irlba, default work".into(), &|| {
        irlba::svd_centered(&view, k, Some(42)).unwrap()
    });

    println!("\non the EXTRACTED submatrix (one sparse copy):");
    for w in [k + 7, k + 30, 2 * k, 3 * k] {
        run(format!("irlba, work={w}"), &|| {
            irlba::svd_with(
                &sub,
                &irlba::IrlbaConfig::new(k)
                    .seed(42)
                    .work(w)
                    .mean_center(true),
                Some(sub.col_means()),
            )
            .unwrap()
        });
    }
    for q in [2usize, 4, 7] {
        run(format!("randomized, {q} power iterations"), &|| {
            randomized::svd_with(
                &sub,
                &randomized::RandomizedConfig::new(k)
                    .seed(42)
                    .power_iterations(q)
                    .mean_center(true),
                None,
            )
            .unwrap()
        });
    }
    for b in [2usize, 3] {
        run(format!("randomized, block krylov x{b}"), &|| {
            randomized::svd_with(
                &sub,
                &randomized::RandomizedConfig::new(k)
                    .seed(42)
                    .block_krylov(b)
                    .mean_center(true),
                None,
            )
            .unwrap()
        });
    }

    rows.sort_by_key(|r| r.1);
    println!("\nfastest first (relative error against the reference):");
    for (label, d, e, mv) in &rows {
        println!("  {label:<34} {:>9.2?}  rel {:.2e}  matvecs {mv}", d, e);
    }
}
