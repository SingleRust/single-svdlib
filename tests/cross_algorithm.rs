//! Cross-algorithm agreement, exercised through the public API only.
//!
//! The unit tests inside each module check that module. These check the contract a
//! consumer actually depends on: that every solver agrees with a dense reference and
//! with each other, over a shared set of fixtures.

#![allow(clippy::needless_range_loop)]

use ndarray::{Array2, Axis};
use single_svdlib::{irlba, randomized, MaskedCsMat, SparseMat, SvdMat, SvdRec};
use sprs::{SpIndex, TriMatI};

// ---------------------------------------------------------------------------
// Fixtures. A self-contained LCG so they are stable across `rand` versions.
// ---------------------------------------------------------------------------

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
    fn signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn dense_of(m: &SvdMat<f64>) -> Array2<f64> {
    let mut d = Array2::zeros((m.rows(), m.cols()));
    for (v, (i, j)) in m.iter() {
        d[[i.index(), j.index()]] = *v;
    }
    d
}

/// Reference singular values from a dense LAPACK-grade factorization.
fn reference(a: &Array2<f64>) -> Vec<f64> {
    let m = nalgebra::DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]]);
    let mut s: Vec<f64> = m.singular_values().iter().copied().collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}

fn sparse(rows: usize, cols: usize, density: f64, seed: u64) -> SvdMat<f64> {
    let mut rng = Lcg::new(seed);
    let target = ((rows as f64 * cols as f64 * density).round() as usize).max(1);
    let mut seen = std::collections::HashSet::new();
    let mut t = TriMatI::<f64, u32>::new((rows, cols));
    let mut attempts = 0usize;
    while seen.len() < target && attempts < target * 100 {
        attempts += 1;
        let (i, j) = (rng.range(rows), rng.range(cols));
        if seen.insert((i, j)) {
            let v = rng.signed() * 10.0;
            t.add_triplet(i, j, if v.abs() < 1e-6 { 1.0 } else { v });
        }
    }
    t.to_csr::<u64>()
}

fn lowrank(rows: usize, cols: usize, rank: usize, seed: u64) -> SvdMat<f64> {
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
                val += u[i][k] * v[j][k] / (k as f64 + 1.0);
            }
            t.add_triplet(i, j, val + rng.signed() * 0.001);
        }
    }
    t.to_csr::<u64>()
}

fn diagonal(n: usize) -> SvdMat<f64> {
    let mut t = TriMatI::<f64, u32>::new((n, n));
    for i in 0..n {
        t.add_triplet(i, i, (n - i) as f64);
    }
    t.to_csr::<u64>()
}

/// Every fixture, with the rank to request from each.
fn fixtures() -> Vec<(&'static str, SvdMat<f64>, usize)> {
    vec![
        ("diagonal_50", diagonal(50), 10),
        ("sparse_tall_500x60", sparse(500, 60, 0.08, 7), 12),
        ("sparse_wide_60x500", sparse(60, 500, 0.08, 11), 12),
        ("sparse_square_200x200", sparse(200, 200, 0.03, 13), 15),
        ("lowrank_300x90_r12", lowrank(300, 90, 12, 17), 12),
        ("lowrank_90x300_r8", lowrank(90, 300, 8, 19), 8),
    ]
}

fn max_rel(got: &SvdRec<f64>, want: &[f64]) -> f64 {
    got.s
        .iter()
        .enumerate()
        .map(|(i, &g)| (g - want[i]).abs() / want[i].abs().max(1e-30))
        .fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------------------

/// IRLBA — and therefore the top-level `svd` — must match LAPACK on every fixture.
#[test]
fn irlba_matches_lapack_everywhere() {
    for (name, a, rank) in fixtures() {
        let want = reference(&dense_of(&a));
        let got = irlba::svd_seed(&a, rank, 42).unwrap_or_else(|e| panic!("{name}: {e}"));
        let err = max_rel(&got, &want);
        assert!(err < 1e-9, "{name}: max relative error {err:.3e}");
    }
}

#[test]
fn top_level_svd_matches_irlba() {
    for (name, a, rank) in fixtures() {
        let via_top = single_svdlib::svd_seed(&a, rank, 42).unwrap();
        let via_mod = irlba::svd_seed(&a, rank, 42).unwrap();
        assert_eq!(via_top.s, via_mod.s, "{name}");
    }
}

/// Randomized SVD must land within its accuracy class on every fixture, and block
/// Krylov must never be worse than power iteration.
#[test]
fn randomized_lands_within_its_accuracy_class() {
    for (name, a, rank) in fixtures() {
        let want = reference(&dense_of(&a));

        let power = randomized::svd_with(
            &a,
            &randomized::RandomizedConfig::new(rank)
                .seed(42)
                .power_iterations(7),
            None,
        )
        .unwrap_or_else(|e| panic!("{name} power: {e}"));

        let krylov = randomized::svd_block_krylov(&a, rank, 4, Some(42))
            .unwrap_or_else(|e| panic!("{name} krylov: {e}"));

        let e_power = max_rel(&power, &want);
        let e_krylov = max_rel(&krylov, &want);

        // A loose absolute bar: randomized methods cannot be held to LAPACK precision
        // on a flat spectrum, but they must be in the right ballpark.
        assert!(e_power < 0.2, "{name}: power iteration error {e_power:.3e}");
        assert!(e_krylov < 0.2, "{name}: block krylov error {e_krylov:.3e}");
        // Block Krylov exists to dominate power iteration; allow a little slack for
        // cases where both are already at machine precision.
        assert!(
            e_krylov <= e_power * 1.5 + 1e-12,
            "{name}: block krylov {e_krylov:.3e} worse than power iteration {e_power:.3e}"
        );
    }
}

/// Storage order must not change the answer.
#[test]
fn csr_and_csc_agree_across_solvers() {
    for (name, a, rank) in fixtures() {
        let csc = a.to_other_storage();
        assert!(a.is_csr() && csc.is_csc(), "{name}: storage setup");

        let i_csr = irlba::svd_seed(&a, rank, 42).unwrap();
        let i_csc = irlba::svd_seed(&csc, rank, 42).unwrap();
        for (x, y) in i_csr.s.iter().zip(i_csc.s.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-10);
        }

        let r_csr = randomized::svd_seed(&a, rank, 42).unwrap();
        let r_csc = randomized::svd_seed(&csc, rank, 42).unwrap();
        for (x, y) in r_csr.s.iter().zip(r_csc.s.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-9);
        }
    }
}

/// Index width is a memory choice, not a numerical one.
#[test]
fn index_widths_agree() {
    let a32 = sparse(300, 100, 0.05, 23);

    let mut t64 = TriMatI::<f64, u64>::new((300, 100));
    for (v, (i, j)) in a32.iter() {
        t64.add_triplet(i as usize, j as usize, *v);
    }
    let a64: SvdMat<f64, u64, u64> = t64.to_csr::<u64>();

    let x = irlba::svd_seed(&a32, 10, 42).unwrap();
    let y = irlba::svd_seed(&a64, 10, 42).unwrap();
    for (p, q) in x.s.iter().zip(y.s.iter()) {
        approx::assert_relative_eq!(p, q, max_relative = 1e-12);
    }
}

/// Rank-`k` truncation error must equal the reference spectral tail exactly:
/// `||A - A_k||_F = sqrt(sum_{i>k} sigma_i^2)`. This is a much stronger check than
/// comparing singular values, because it tests the vectors too.
#[test]
fn truncation_error_matches_spectral_tail() {
    for (name, a, rank) in fixtures() {
        let dense = dense_of(&a);
        let refs = reference(&dense);
        let tail: f64 = refs[rank..].iter().map(|v| v * v).sum::<f64>().sqrt();

        let got = irlba::svd_seed(&a, rank, 42).unwrap();
        let err: f64 = (&got.recompose() - &dense)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        approx::assert_relative_eq!(err, tail, max_relative = 1e-6);
        assert!(err.is_finite(), "{name}: non-finite reconstruction error");
    }
}

/// `A·vᵢ = σᵢ·uᵢ` for every returned triplet, which is the definition.
#[test]
fn triplets_satisfy_the_defining_relation() {
    for (name, a, rank) in fixtures() {
        let got = irlba::svd_seed(&a, rank, 42).unwrap();
        for i in 0..got.d {
            let vi: Vec<f64> = got.vt.row(i).to_vec();
            let mut av = vec![0.0; a.rows()];
            SparseMat::mul_vec(&a, &vi, &mut av, false);
            let resid: f64 = av
                .iter()
                .zip(got.u.column(i).iter())
                .map(|(&x, &ui)| {
                    let d = x - got.s[i] * ui;
                    d * d
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                resid / got.s[0] < 1e-8,
                "{name} triplet {i}: ||A v - s u|| / s_max = {:.3e}",
                resid / got.s[0]
            );
        }
    }
}

/// PCA on a masked matrix, checked against explicitly building the submatrix and
/// centering it densely — the composition 1.x got wrong in three separate places.
#[test]
fn masked_pca_matches_dense_reference() {
    let a = sparse(400, 40, 0.15, 29);
    let cols: Vec<usize> = (0..40).filter(|c| c % 3 == 0).collect();
    let masked = MaskedCsMat::with_columns(&a, &cols);

    // The submatrix, densely.
    let full = dense_of(&a);
    let mut sub = Array2::<f64>::zeros((400, cols.len()));
    for (new, &old) in cols.iter().enumerate() {
        sub.column_mut(new).assign(&full.column(old));
    }
    let means = sub.mean_axis(Axis(0)).unwrap();
    let centered = &sub - &means.view().insert_axis(Axis(0));
    let want = reference(&centered);

    let rank = 6;
    let got = irlba::svd_centered(&masked, rank, Some(42)).unwrap();
    assert_eq!(got.u.nrows(), 400);
    assert_eq!(got.vt.ncols(), cols.len());
    for (i, &g) in got.s.iter().enumerate() {
        let rel = (g - want[i]).abs() / want[i].abs().max(1e-30);
        assert!(
            rel < 1e-8,
            "masked PCA singular value {i}: {g:.9e} vs {:.9e} (rel {rel:.3e})",
            want[i]
        );
    }
}

/// f32 must work end to end and land at f32 precision.
#[test]
fn f32_end_to_end() {
    let a64 = lowrank(200, 60, 8, 31);
    let want = reference(&dense_of(&a64));

    let mut t = TriMatI::<f32, u32>::new((200, 60));
    for (v, (i, j)) in a64.iter() {
        t.add_triplet(i as usize, j as usize, *v as f32);
    }
    let a32: SvdMat<f32> = t.to_csr::<u64>();

    let got = irlba::svd_seed(&a32, 8, 42).unwrap();
    for (i, &g) in got.s.iter().enumerate() {
        let rel = ((g as f64) - want[i]).abs() / want[i].abs().max(1e-30);
        assert!(rel < 1e-4, "f32 singular value {i}: rel {rel:.3e}");
    }
}

/// Seeded runs must be bit-reproducible; unseeded ones must actually differ.
#[test]
fn seeding_behaves() {
    let a = sparse(200, 80, 0.08, 37);

    let x = irlba::svd_seed(&a, 8, 7).unwrap();
    let y = irlba::svd_seed(&a, 8, 7).unwrap();
    assert_eq!(x.s, y.s);
    assert_eq!(x.u, y.u);
    assert_eq!(x.vt, y.vt);

    let cfg = randomized::RandomizedConfig::new(6).power_iterations(0);
    let p = randomized::svd_with(&a, &cfg, None).unwrap();
    let q = randomized::svd_with(&a, &cfg, None).unwrap();
    assert_ne!(
        p.diagnostics.random_seed, q.diagnostics.random_seed,
        "unseeded runs reused a seed"
    );
}

/// Requesting more triplets than the matrix can supply must be an error, not a panic
/// or silent truncation.
#[test]
fn out_of_range_rank_is_an_error() {
    let a = sparse(30, 12, 0.3, 41);
    assert!(irlba::svd(&a, 13).is_err());
    assert!(randomized::svd(&a, 13).is_err());
    assert!(single_svdlib::svd(&a, 0).is_err());
}

/// The deprecated module must still compile and run — 2.0 does not remove the API.
#[test]
#[allow(deprecated)]
fn deprecated_lanczos_still_callable() {
    let a = sparse(120, 50, 0.1, 43);
    // Only that it runs; its accuracy is documented as unreliable.
    let got = single_svdlib::lanczos::svd_dim_seed(&a, 8, 42);
    assert!(
        got.is_ok(),
        "deprecated path failed to run: {:?}",
        got.err()
    );
}
