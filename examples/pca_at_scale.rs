//! PCA on a very large sparse matrix, restricted to a subset of columns, without ever
//! densifying or modifying it.
//!
//! Shaped like a single-cell workload: cells x genes, a few hundred non-zeros per cell,
//! restricted to a "highly variable gene" subset, reduced to 50 components.
//!
//! ```text
//! cargo run --release --example pca_at_scale
//! cargo run --release --example pca_at_scale 500000 30000 200 2000 50
//! ```
//!
//! Arguments: `cells genes nnz_per_cell selected_genes components`.

use single_svdlib::{irlba, MaskedCsMat, SparseMat, SparseMatDense, SvdMat};
use sprs::CsMatI;
use std::time::Instant;

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

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Build the CSR arrays directly rather than going through a triplet matrix, which
/// would need a second full copy of the data before conversion.
///
/// The cells carry a latent factor structure — a handful of "types", each with its own
/// gene programme — so the spectrum decays the way real data's does. Uniform noise would
/// give a flat Marchenko-Pastur spectrum, which is a far harder case for any Krylov
/// method than anything measured on real counts.
fn build(cells: usize, genes: usize, per_cell: usize, types: usize, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);

    // Each type prefers a contiguous block of marker genes.
    let block = (genes / types.max(1)).max(1);

    let nnz_est = cells * per_cell;
    let mut indptr: Vec<u64> = Vec::with_capacity(cells + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(nnz_est);
    let mut data: Vec<f64> = Vec::with_capacity(nnz_est);
    let mut row: Vec<(u32, f64)> = Vec::with_capacity(per_cell);

    indptr.push(0);
    for _ in 0..cells {
        let ty = rng.range(types.max(1));
        row.clear();
        for _ in 0..per_cell {
            // Most counts land in this cell type's programme; the rest is background.
            let (g, weight) = if rng.next_f64() < 0.7 {
                let lo = ty * block;
                (lo + rng.range(block), 8.0)
            } else {
                (rng.range(genes), 1.0)
            };
            let count = (rng.next_f64() * weight).floor() + 1.0;
            row.push((g as u32, count));
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
    let cells = a.first().copied().unwrap_or(1_000_000);
    let genes = a.get(1).copied().unwrap_or(30_000);
    let per_cell = a.get(2).copied().unwrap_or(150);
    let n_selected = a.get(3).copied().unwrap_or(2_000);
    let components = a.get(4).copied().unwrap_or(50);

    println!("building {cells} cells x {genes} genes, ~{per_cell} nnz/cell, 24 latent types ...");
    let t0 = Instant::now();
    let types = 24;
    let matrix = build(cells, genes, per_cell, types, 7);
    let build_time = t0.elapsed();

    let bytes = matrix.indptr().len() * 8 + matrix.indices().len() * 4 + matrix.data().len() * 8;
    let as_usize = matrix.indptr().len() * 8 + matrix.indices().len() * 8 + matrix.data().len() * 8;
    let as_dense = cells * genes * 8;
    println!("  {:?}, nnz = {}", build_time, matrix.nnz());
    println!(
        "  sparse (u32/u64 indices) : {:>8.2} GiB",
        gib(bytes)
    );
    println!(
        "  sparse (usize indices)   : {:>8.2} GiB   (+{:.0}%)",
        gib(as_usize),
        100.0 * (as_usize as f64 / bytes as f64 - 1.0)
    );
    println!(
        "  the same matrix dense    : {:>8.2} GiB   ({:.0}x larger)",
        gib(as_dense),
        as_dense as f64 / bytes as f64
    );

    // "Highly variable genes": every k-th gene, plus the marker block.
    let stride = (genes / n_selected).max(1);
    let mut selected: Vec<usize> = (0..genes).step_by(stride).collect();
    selected.extend(0..256);
    selected.sort_unstable();
    selected.dedup();
    println!("\nselecting {} of {genes} genes", selected.len());

    // ---- the view: no copy, source untouched ----
    let t = Instant::now();
    let view = MaskedCsMat::with_columns(&matrix, &selected);
    let view_build = t.elapsed();
    println!(
        "  view built in {:?}: {} x {}, {} nnz in mask ({:.1}% of the source)",
        view_build,
        view.rows(),
        view.cols(),
        view.nnz(),
        100.0 * view.nnz() as f64 / matrix.nnz() as f64
    );

    println!("\nPCA on the view ({components} components, mean-centered) ...");
    let t = Instant::now();
    let pca_view = irlba::svd_centered(&view, components, Some(42)).expect("PCA on view failed");
    let view_time = t.elapsed();
    let (restarts, converged) = match pca_view.diagnostics.detail {
        single_svdlib::Detail::Irlba {
            restarts,
            converged,
            ..
        } => (restarts, converged),
        _ => unreachable!(),
    };
    println!(
        "  {:?}  restarts={restarts} converged={converged} matvecs={}",
        view_time, pca_view.diagnostics.matvecs
    );
    println!(
        "  scores (cells x PCs) = {:?}, loadings (PCs x genes) = {:?}",
        pca_view.u.dim(),
        pca_view.vt.dim()
    );
    println!(
        "  sigma[0] = {:.4}   sigma[{}] = {:.4}",
        pca_view.s[0],
        components - 1,
        pca_view.s[components - 1]
    );

    // ---- the extraction: one sparse copy, then every product is cheaper ----
    println!("\nextracting the same submatrix (still sparse) ...");
    let t = Instant::now();
    let extracted = view.to_sparse();
    let extract_time = t.elapsed();
    let ex_bytes =
        extracted.indptr().len() * 8 + extracted.indices().len() * 4 + extracted.data().len() * 8;
    println!(
        "  {:?}, {} x {}, {} nnz, {:.2} GiB",
        extract_time,
        extracted.rows(),
        extracted.cols(),
        extracted.nnz(),
        gib(ex_bytes)
    );

    let t = Instant::now();
    let pca_copy =
        irlba::svd_centered(&extracted, components, Some(42)).expect("PCA on extraction failed");
    let copy_time = t.elapsed();
    println!("  PCA {:?}  matvecs={}", copy_time, pca_copy.diagnostics.matvecs);

    // ---- agreement ----
    let worst = (0..components)
        .map(|i| (pca_view.s[i] - pca_copy.s[i]).abs() / pca_view.s[0])
        .fold(0.0f64, f64::max);
    println!("\nview vs extraction: max relative difference = {worst:.3e}");

    // Independent check that this really is the PCA of the centered submatrix:
    // ||(A_sub - 1*mean^T) v_i - sigma_i u_i|| must vanish.
    let means = view.col_means();
    let mut worst_resid: f64 = 0.0;
    for i in 0..components.min(5) {
        let vi: Vec<f64> = pca_view.vt.row(i).to_vec();
        let mut av = vec![0.0; view.rows()];
        view.mul_vec(&vi, &mut av, false);
        let shift: f64 = means.iter().zip(vi.iter()).map(|(&m, &v)| m * v).sum();
        let resid: f64 = av
            .iter()
            .zip(pca_view.u.column(i).iter())
            .map(|(&x, &ui)| {
                let e = (x - shift) - pca_view.s[i] * ui;
                e * e
            })
            .sum::<f64>()
            .sqrt();
        worst_resid = worst_resid.max(resid / pca_view.s[0]);
    }
    println!("centered residual ||A_c v - s u|| / sigma_max = {worst_resid:.3e}");

    println!(
        "\nsummary: view {:?} (no copy) | extract {:?} + PCA {:?} = {:?}",
        view_time,
        extract_time,
        copy_time,
        extract_time + copy_time
    );
}
