//! Property-based tests.
//!
//! The hand-written suites check cases someone thought of. These check invariants that
//! must hold for *any* operand, over shapes, densities and value distributions that
//! proptest chooses — including ones nobody would think to write down.
//!
//! # Tolerance policy
//!
//! Every comparison is relative to `σ_max`, never to the individual singular value.
//! A trailing singular value can legitimately be `1e-18` on a rank-deficient operand,
//! where its own relative error is meaningless but its absolute error against the
//! matrix scale is exactly what matters. This is the same convention LAPACK's own test
//! suite uses.

#![allow(clippy::needless_range_loop)]

use ndarray::{Array2, Axis};
use proptest::prelude::*;
use single_svdlib::{
    dense, irlba, matrix::kernels, randomized, MaskedCsMat, SparseMat, SparseMatDense, SvdMat,
};
use sprs::TriMatI;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Value distributions that stress different failure modes.
fn value() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Ordinary magnitudes.
        4 => -10.0f64..10.0,
        // Wide dynamic range: exercises cancellation and scaling.
        2 => (-8i32..8, 1.0f64..10.0, any::<bool>())
            .prop_map(|(e, m, neg)| {
                let v = m * 10f64.powi(e);
                if neg { -v } else { v }
            }),
        // Small integers are exact in f64, so they exercise the exact-arithmetic paths
        // and produce genuine rank deficiency far more often than continuous values.
        2 => (-8i64..8).prop_map(|v| v as f64),
        // Exact zeros, to produce structural sparsity and empty rows/columns.
        1 => Just(0.0),
    ]
}

/// A sparse matrix and the dense array holding exactly the same content.
///
/// Both are built from one buffer, so any disagreement is a bug in the code under
/// test rather than in the fixture.
fn matrix(max_dim: usize) -> impl Strategy<Value = (SvdMat<f64>, Array2<f64>)> {
    (2usize..=max_dim, 2usize..=max_dim).prop_flat_map(|(rows, cols)| {
        proptest::collection::vec(value(), rows * cols).prop_map(move |vals| {
            let mut tri = TriMatI::<f64, u32>::new((rows, cols));
            let mut dense = Array2::<f64>::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    let v = vals[i * cols + j];
                    dense[[i, j]] = v;
                    if v != 0.0 {
                        tri.add_triplet(i, j, v);
                    }
                }
            }
            (tri.to_csr::<u64>(), dense)
        })
    })
}

/// A dense tall-skinny matrix, for the QR properties.
fn tall(max_rows: usize, max_cols: usize) -> impl Strategy<Value = Array2<f64>> {
    (1usize..=max_cols)
        .prop_flat_map(move |cols| (Just(cols), cols..=max_rows.max(cols)))
        .prop_flat_map(|(cols, rows)| {
            proptest::collection::vec(value(), rows * cols)
                .prop_map(move |v| Array2::from_shape_vec((rows, cols), v).unwrap())
        })
}

/// A matrix paired with a rank that is always valid for it, so the strategy never has
/// to reject — filtering `rank` after the fact exhausts proptest's global reject budget.
fn matrix_and_rank(max_dim: usize) -> impl Strategy<Value = (SvdMat<f64>, Array2<f64>, usize)> {
    matrix(max_dim).prop_flat_map(|(a, d)| {
        let min_dim = a.rows().min(a.cols());
        (Just(a), Just(d), 1usize..=min_dim)
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn reference(a: &Array2<f64>) -> Vec<f64> {
    let m = nalgebra::DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]]);
    let mut s: Vec<f64> = m.singular_values().iter().copied().collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}

fn frob(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The matrix scale every tolerance is measured against.
fn scale(want: &[f64]) -> f64 {
    want.first().copied().unwrap_or(0.0).max(f64::MIN_POSITIVE)
}

// ---------------------------------------------------------------------------
// Core solver invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 2000, ..ProptestConfig::default() })]

    /// IRLBA's singular values must match a dense reference, for any operand.
    #[test]
    fn irlba_matches_dense_reference((a, d) in matrix(22)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let want = reference(&d);
        let s = scale(&want);

        let got = irlba::svd_seed(&a, rank, 42);
        prop_assume!(got.is_ok(), "solver declined: {:?}", got.err());
        let got = got.unwrap();

        let (converged, max_residual, restarts) = match got.diagnostics.detail {
            single_svdlib::Detail::Irlba { converged, max_residual, restarts, .. } => (converged, max_residual, restarts),
            _ => unreachable!(),
        };
        for i in 0..got.d {
            let err = (got.s[i] - want[i]).abs();
            prop_assert!(
                err <= 1e-8 * s,
                "triplet {i}: got {:.12e}, want {:.12e}, abs err {:.3e} vs scale {:.3e} \
                 [converged={converged} restarts={restarts} max_residual={max_residual:.3e} rank={} d={} shape={:?}]\n\
                 got : {:?}\n want: {:?}",
                got.s[i], want[i], err, s, rank, got.d, (a.rows(), a.cols()),
                got.s.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>(),
                want.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>()
            );
        }
    }

    /// Shapes, ordering and finiteness — the contract every caller relies on.
    #[test]
    fn irlba_output_is_well_formed((a, _d) in matrix(22)) {
        let (rows, cols) = (a.rows(), a.cols());
        let min_dim = rows.min(cols);
        let rank = (min_dim - 1).max(1);

        let got = irlba::svd_seed(&a, rank, 42);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        prop_assert_eq!(got.u.dim(), (rows, got.d), "u shape");
        prop_assert_eq!(got.vt.dim(), (got.d, cols), "vt shape");
        prop_assert_eq!(got.s.len(), got.d, "s length");

        prop_assert!(got.s.iter().all(|v| v.is_finite() && *v >= 0.0), "s: {:?}", got.s);
        prop_assert!(got.u.iter().all(|v| v.is_finite()), "u has non-finite entries");
        prop_assert!(got.vt.iter().all(|v| v.is_finite()), "vt has non-finite entries");

        for w in got.s.to_vec().windows(2) {
            prop_assert!(w[0] >= w[1], "not descending: {:?}", got.s);
        }
    }

    /// Singular vectors must be orthonormal, including on rank-deficient operands
    /// where the trailing directions are arbitrary but still must form a basis.
    #[test]
    fn irlba_vectors_are_orthonormal((a, _d) in matrix(20)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let got = irlba::svd_seed(&a, rank, 42);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        let ou = dense::orthogonality_error(&got.u.view());
        prop_assert!(ou < 1e-8, "||U^T U - I||_F = {ou:.3e}");

        let vt_t = got.vt.t().to_owned();
        let ov = dense::orthogonality_error(&vt_t.view());
        prop_assert!(ov < 1e-8, "||V^T V - I||_F = {ov:.3e}");
    }

    /// The defining relation `A·vᵢ = σᵢ·uᵢ`, measured against the matrix scale.
    #[test]
    fn irlba_triplets_satisfy_definition((a, d) in matrix(20)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let want = reference(&d);
        let s = scale(&want);

        let got = irlba::svd_seed(&a, rank, 42);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        for i in 0..got.d {
            let vi: Vec<f64> = got.vt.row(i).to_vec();
            let mut av = vec![0.0; a.rows()];
            SparseMat::mul_vec(&a, &vi, &mut av, false);
            let resid: f64 = av
                .iter()
                .zip(got.u.column(i).iter())
                .map(|(&x, &ui)| { let e = x - got.s[i] * ui; e * e })
                .sum::<f64>()
                .sqrt();
            prop_assert!(
                resid <= 1e-8 * s,
                "triplet {i}: ||A v - s u|| = {resid:.3e} vs scale {s:.3e}"
            );
        }
    }

    /// Rank-`k` truncation error must equal the reference spectral tail. This tests the
    /// singular *vectors*, not just the values — a wrong subspace shows up here even
    /// when the values happen to be right.
    #[test]
    fn irlba_truncation_matches_spectral_tail((a, d) in matrix(18)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let want = reference(&d);
        let tail: f64 = want[rank..].iter().map(|v| v * v).sum::<f64>().sqrt();
        let s = scale(&want);

        let got = irlba::svd_seed(&a, rank, 42);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        let err = frob(&(&got.recompose() - &d));
        prop_assert!(
            (err - tail).abs() <= 1e-7 * s * (d.nrows() as f64).sqrt(),
            "truncation error {err:.6e} vs spectral tail {tail:.6e} (scale {s:.3e})"
        );
    }

    /// Storage order is an implementation detail and must not change the answer.
    #[test]
    fn storage_order_is_irrelevant((a, d) in matrix(20)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let want = reference(&d);
        let s = scale(&want);
        let csc = a.to_other_storage();

        let x = irlba::svd_seed(&a, rank, 42);
        let y = irlba::svd_seed(&csc, rank, 42);
        prop_assume!(x.is_ok() && y.is_ok());
        let (x, y) = (x.unwrap(), y.unwrap());

        for i in 0..x.d {
            prop_assert!(
                (x.s[i] - y.s[i]).abs() <= 1e-9 * s,
                "CSR vs CSC differ at {i}: {:.12e} vs {:.12e}", x.s[i], y.s[i]
            );
        }
    }

    /// A fixed seed must reproduce byte-identical output.
    #[test]
    fn seeded_runs_are_reproducible((a, _d) in matrix(18)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let x = irlba::svd_seed(&a, rank, 7);
        let y = irlba::svd_seed(&a, rank, 7);
        prop_assume!(x.is_ok() && y.is_ok());
        let (x, y) = (x.unwrap(), y.unwrap());
        prop_assert_eq!(x.s, y.s);
        prop_assert_eq!(x.u, y.u);
        prop_assert_eq!(x.vt, y.vt);
    }

    /// Whatever the operand, the solver returns — no panic, no hang, no non-finite
    /// output smuggled out as success. Ranks beyond what the method supports must be
    /// typed errors.
    #[test]
    fn solvers_always_terminate_cleanly((a, _d, rank) in matrix_and_rank(16)) {
        // A typed error is always acceptable; silently returning garbage is not.
        if let Ok(rec) = irlba::svd_seed(&a, rank, 42) {
            prop_assert!(rec.s.iter().all(|v| v.is_finite()), "irlba leaked non-finite values");
            prop_assert_eq!(rec.d, rank);
        }
        if let Ok(rec) = randomized::svd_seed(&a, rank, 42) {
            prop_assert!(
                rec.s.iter().all(|v| v.is_finite()),
                "randomized leaked non-finite values"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Randomized solvers
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 120, max_shrink_iters: 2000, ..ProptestConfig::default() })]

    /// Randomized SVD can never *overestimate* a singular value by more than rounding:
    /// it computes the SVD of a projection of A onto a subspace, and a projection
    /// cannot have larger singular values than the original. Underestimation is
    /// expected and is the method's approximation error.
    #[test]
    fn randomized_never_overestimates((a, d) in matrix(20)) {
        let min_dim = a.rows().min(a.cols());
        let rank = (min_dim - 1).max(1);
        let want = reference(&d);
        let s = scale(&want);

        let cfg = randomized::RandomizedConfig::new(rank).seed(42).power_iterations(2);
        let got = randomized::svd_with(&a, &cfg, None);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        for i in 0..got.d {
            prop_assert!(
                got.s[i] <= want[i] + 1e-9 * s,
                "triplet {i}: randomized {:.12e} exceeds true {:.12e}",
                got.s[i], want[i]
            );
        }
    }

    /// With enough power iterations the dominant singular value must be close, whatever
    /// the operand. Trailing values are not held to this — that is the method's
    /// documented weakness on flat spectra, not a defect.
    #[test]
    fn randomized_captures_the_dominant_value((a, d) in matrix(20)) {
        let want = reference(&d);
        let s = scale(&want);
        prop_assume!(s > 1e-12);

        let cfg = randomized::RandomizedConfig::new(1).seed(42).power_iterations(12);
        let got = randomized::svd_with(&a, &cfg, None);
        prop_assume!(got.is_ok());
        let got = got.unwrap();

        prop_assert!(
            (got.s[0] - want[0]).abs() <= 1e-3 * s,
            "dominant value {:.9e} vs true {:.9e}", got.s[0], want[0]
        );
    }

    /// Block Krylov spans a superset of the power-iteration subspace at equal block
    /// count, so it must never do worse on the dominant value.
    #[test]
    fn block_krylov_is_no_worse_than_power_iteration((a, d) in matrix(18)) {
        let want = reference(&d);
        let s = scale(&want);
        prop_assume!(s > 1e-12);

        let power = randomized::svd_with(
            &a, &randomized::RandomizedConfig::new(1).seed(42).power_iterations(3), None);
        let krylov = randomized::svd_block_krylov(&a, 1, 4, Some(42));
        prop_assume!(power.is_ok() && krylov.is_ok());

        let e_power = (power.unwrap().s[0] - want[0]).abs() / s;
        let e_krylov = (krylov.unwrap().s[0] - want[0]).abs() / s;
        prop_assert!(
            e_krylov <= e_power + 1e-9,
            "block krylov {e_krylov:.3e} worse than power iteration {e_power:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Kernels and dense helpers
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, max_shrink_iters: 2000, ..ProptestConfig::default() })]

    /// The parallel sparse x dense kernels must equal the dense products exactly enough
    /// that only summation order distinguishes them.
    #[test]
    fn kernels_match_dense_products(
        (a, d) in matrix(24),
        k in 1usize..6,
        budget in prop_oneof![Just(0usize), Just(1024), Just(usize::MAX)],
    ) {
        let (rows, cols) = (a.rows(), a.cols());
        let scale_a = frob(&d).max(1.0);

        // A · D
        let rhs = Array2::from_shape_fn((cols, k), |(i, j)| ((i * 7 + j * 3) % 11) as f64 - 5.0);
        let mut out = Array2::zeros((rows, k));
        kernels::gather_mul(a.view(), rhs.view(), out.view_mut());
        let want = d.dot(&rhs);
        let err = frob(&(&out - &want));
        prop_assert!(err <= 1e-9 * scale_a * (k as f64), "gather_mul err {err:.3e}");

        // Aᵀ · D, at every scratch budget
        let rhs_t = Array2::from_shape_fn((rows, k), |(i, j)| ((i * 5 + j) % 9) as f64 - 4.0);
        let mut out_t = Array2::zeros((cols, k));
        kernels::scatter_mul(a.view(), rhs_t.view(), out_t.view_mut(), budget);
        let want_t = d.t().dot(&rhs_t);
        let err_t = frob(&(&out_t - &want_t));
        prop_assert!(err_t <= 1e-9 * scale_a * (k as f64), "scatter_mul err {err_t:.3e} at budget {budget}");
    }

    /// Mean centering as a rank-1 correction must equal explicitly building the
    /// centered dense matrix.
    #[test]
    fn centering_matches_explicit_dense((a, d) in matrix(20), k in 1usize..4) {
        let (rows, cols) = (a.rows(), a.cols());
        let means = SparseMatDense::col_means(&a);
        let want_means = d.mean_axis(Axis(0)).unwrap();
        for (g, w) in means.iter().zip(want_means.iter()) {
            prop_assert!((g - w).abs() <= 1e-9 * (w.abs() + 1.0), "col mean {g} vs {w}");
        }

        let centered = &d - &want_means.view().insert_axis(Axis(0));
        let scale_c = frob(&centered).max(1.0);

        let rhs = Array2::from_shape_fn((cols, k), |(i, j)| ((i * 3 + j) % 7) as f64 - 3.0);
        let mut out = Array2::zeros((rows, k));
        SparseMatDense::mul_dense_centered(&a, rhs.view(), out.view_mut(), false, means.view());
        let err = frob(&(&out - &centered.dot(&rhs)));
        prop_assert!(err <= 1e-8 * scale_c * (k as f64), "centered A·D err {err:.3e}");

        let rhs_t = Array2::from_shape_fn((rows, k), |(i, j)| ((i + j * 2) % 5) as f64 - 2.0);
        let mut out_t = Array2::zeros((cols, k));
        SparseMatDense::mul_dense_centered(&a, rhs_t.view(), out_t.view_mut(), true, means.view());
        let err_t = frob(&(&out_t - &centered.t().dot(&rhs_t)));
        prop_assert!(err_t <= 1e-8 * scale_c * (k as f64), "centered Aᵀ·D err {err_t:.3e}");
    }

    /// A masked view must behave exactly like the physically extracted submatrix.
    #[test]
    fn masked_view_matches_physical_subset(
        (a, d) in matrix(20),
        picks in proptest::collection::vec(any::<bool>(), 1..21),
        anchor in any::<prop::sample::Index>(),
    ) {
        let cols = a.cols();
        let mut selected: Vec<usize> =
            (0..cols).filter(|c| *picks.get(*c % picks.len()).unwrap_or(&true)).collect();
        // Guarantee a non-empty mask instead of rejecting the empty draw — rejection
        // here burns proptest's global reject budget and aborts the run.
        if selected.is_empty() {
            selected.push(anchor.index(cols));
        }

        let masked = MaskedCsMat::with_columns(&a, &selected);
        prop_assert_eq!(masked.cols(), selected.len());

        // The physical submatrix.
        let mut sub = Array2::<f64>::zeros((a.rows(), selected.len()));
        for (new, &old) in selected.iter().enumerate() {
            sub.column_mut(new).assign(&d.column(old));
        }
        let scale_s = frob(&sub).max(1.0);

        // Matvec, both directions.
        let x: Vec<f64> = (0..selected.len()).map(|i| (i % 5) as f64 - 2.0).collect();
        let mut y = vec![0.0; a.rows()];
        masked.mul_vec(&x, &mut y, false);
        let want = sub.dot(&ndarray::Array1::from_vec(x));
        for (g, w) in y.iter().zip(want.iter()) {
            prop_assert!((g - w).abs() <= 1e-9 * scale_s, "masked A·x {g} vs {w}");
        }

        let xt: Vec<f64> = (0..a.rows()).map(|i| (i % 3) as f64 - 1.0).collect();
        let mut yt = vec![0.0; selected.len()];
        masked.mul_vec(&xt, &mut yt, true);
        let want_t = sub.t().dot(&ndarray::Array1::from_vec(xt));
        for (g, w) in yt.iter().zip(want_t.iter()) {
            prop_assert!((g - w).abs() <= 1e-9 * scale_s, "masked Aᵀ·x {g} vs {w}");
        }
    }
}

// ---------------------------------------------------------------------------
// TSQR
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 2000, ..ProptestConfig::default() })]

    /// `A == Q·R` with `Q` orthonormal and `R` upper triangular, for any tall operand.
    #[test]
    fn tsqr_factorizes_correctly(a in tall(40, 8)) {
        let original = a.clone();
        let mut q = a;
        let r = dense::tsqr(&mut q);
        prop_assume!(r.is_ok());
        let r = r.unwrap();
        let (m, n) = original.dim();

        prop_assert_eq!(q.dim(), (m, n));
        prop_assert_eq!(r.dim(), (n, n));
        prop_assert!(q.iter().all(|v| v.is_finite()), "Q has non-finite entries");
        prop_assert!(r.iter().all(|v| v.is_finite()), "R has non-finite entries");

        // R upper triangular.
        for i in 1..n {
            for j in 0..i {
                prop_assert!(r[[i, j]].abs() < 1e-10 * frob(&original).max(1.0),
                    "R not upper triangular at ({i},{j}): {}", r[[i, j]]);
            }
        }

        // Q orthonormal.
        let orth = dense::orthogonality_error(&q.view());
        prop_assert!(orth < 1e-9, "||Q^T Q - I||_F = {orth:.3e} for {m}x{n}");

        // A == Q R.
        let scale_a = frob(&original).max(f64::MIN_POSITIVE);
        let err = frob(&(&q.dot(&r) - &original)) / scale_a;
        prop_assert!(err < 1e-9, "||A - QR||/||A|| = {err:.3e} for {m}x{n}");
    }

    /// The small dense SVD must reconstruct its operand and come back ordered.
    #[test]
    fn small_svd_reconstructs(a in tall(16, 10)) {
        let svd = dense::small_svd(a.view());
        prop_assume!(svd.is_ok());
        let svd = svd.unwrap();

        for w in svd.s.to_vec().windows(2) {
            prop_assert!(w[0] >= w[1], "singular values not descending");
        }
        prop_assert!(svd.s.iter().all(|v| v.is_finite() && *v >= 0.0));

        let scaled = &svd.u * &svd.s.view().insert_axis(Axis(0));
        let err = frob(&(&scaled.dot(&svd.vt) - &a)) / frob(&a).max(f64::MIN_POSITIVE);
        prop_assert!(err < 1e-9, "relative reconstruction {err:.3e}");
    }
}
