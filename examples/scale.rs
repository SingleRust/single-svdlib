//! Scale check at single-cell-like dimensions.
//!
//! Reports matrix footprint, wall time and agreement between the solvers on a
//! 200k × 30k operand. Run under `/usr/bin/time -l` (macOS) or `/usr/bin/time -v`
//! (Linux) to see peak RSS.
//!
//! Pass a solver name to measure one in isolation, so peak RSS is attributable:
//!
//! ```text
//! cargo run --release --example scale            # all three
//! cargo run --release --example scale irlba
//! cargo run --release --example scale randomized
//! cargo run --release --example scale krylov
//! ```

use single_svdlib::{irlba, randomized, SvdMat};
use sprs::TriMatI;
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

fn mib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let run = |name: &str| which == "all" || which == name;

    let (rows, cols) = (200_000usize, 30_000usize);
    let nnz_per_row = 25usize; // ~0.083% dense, typical of a count matrix
    let rank = 50usize;

    println!("building {rows} x {cols}, ~{} nnz ...", rows * nnz_per_row);
    let t0 = Instant::now();
    let mut tri = TriMatI::<f64, u32>::new((rows, cols));
    let mut rng = Lcg::new(7);
    for i in 0..rows {
        // A handful of "marker" columns carry extra weight, so the spectrum has real
        // structure rather than being flat noise.
        for _ in 0..nnz_per_row {
            let j = if rng.next_f64() < 0.3 {
                rng.range(64)
            } else {
                rng.range(cols)
            };
            tri.add_triplet(i, j, rng.next_f64() * 10.0);
        }
    }
    let a: SvdMat<f64> = tri.to_csr::<u64>();
    // Release the triplet buffers before solving so the reported peak reflects the
    // solver, not the loader.
    drop(tri);
    println!("  built in {:?}", t0.elapsed());

    let idx_bytes = std::mem::size_of_val(a.indices());
    let ptr_bytes = a.indptr().len() * std::mem::size_of::<u64>();
    let val_bytes = std::mem::size_of_val(a.data());
    let total = idx_bytes + ptr_bytes + val_bytes;
    let usize_equiv = a.indices().len() * std::mem::size_of::<usize>()
        + a.indptr().len() * std::mem::size_of::<usize>()
        + val_bytes;
    println!(
        "  nnz = {}, matrix = {:.1} MiB (u32/u64 indices)",
        a.nnz(),
        mib(total)
    );
    println!(
        "  the same matrix with usize indices = {:.1} MiB  ({:.0}% larger)",
        mib(usize_equiv),
        100.0 * (usize_equiv as f64 / total as f64 - 1.0)
    );

    // Basis memory IRLBA will hold, known before the solve starts.
    let work = rank + 7;
    let basis = ((work + 1) * cols + work * rows) * std::mem::size_of::<f64>();
    println!(
        "  irlba basis (work = {work}) = {:.1} MiB, fixed\n",
        mib(basis)
    );

    let mut by_irlba = None;
    if run("irlba") {
        println!("irlba rank {rank} ...");
        let t = Instant::now();
        let rec = irlba::svd_seed(&a, rank, 42).expect("irlba failed");
        let elapsed = t.elapsed();
        let (restarts, converged) = match rec.diagnostics.detail {
            single_svdlib::Detail::Irlba {
                restarts,
                converged,
                ..
            } => (restarts, converged),
            _ => unreachable!(),
        };
        println!(
            "  {elapsed:?}  restarts={restarts} converged={converged} matvecs={}",
            rec.diagnostics.matvecs
        );
        println!(
            "  sigma[0]={:.6}  sigma[{}]={:.6}",
            rec.s[0],
            rank - 1,
            rec.s[rank - 1]
        );
        by_irlba = Some(rec);
    }

    let mut others: Vec<(&str, single_svdlib::SvdRec<f64>)> = Vec::new();
    if run("randomized") {
        println!("\nrandomized rank {rank}, 2 power iterations ...");
        let t = Instant::now();
        let rec = randomized::svd_seed(&a, rank, 42).expect("randomized failed");
        println!("  {:?}  matvecs={}", t.elapsed(), rec.diagnostics.matvecs);
        others.push(("randomized", rec));
    }
    if run("krylov") {
        println!("\nrandomized rank {rank}, block krylov x3 ...");
        let t = Instant::now();
        let rec = randomized::svd_block_krylov(&a, rank, 3, Some(42)).expect("block krylov failed");
        println!("  {:?}  matvecs={}", t.elapsed(), rec.diagnostics.matvecs);
        others.push(("block krylov", rec));
    }

    // IRLBA converged to a residual tolerance, so treat it as the reference.
    if let Some(reference) = &by_irlba {
        if !others.is_empty() {
            println!("\nagreement against irlba (relative, over the top {rank}):");
            for (name, rec) in &others {
                let worst = (0..rank)
                    .map(|i| (rec.s[i] - reference.s[i]).abs() / reference.s[i])
                    .fold(0.0f64, f64::max);
                println!("  {name:<14} max rel diff = {worst:.3e}");
            }
        }
    }
}
