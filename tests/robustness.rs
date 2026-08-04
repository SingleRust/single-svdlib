//! Adversarial and degenerate inputs.
//!
//! Every test here asserts that a hostile input produces either a correct answer or a
//! typed error — never a panic, a hang, or a silently wrong result.

#![allow(clippy::needless_range_loop)]

use single_svdlib::{irlba, randomized, MaskedCsMat, SparseMat, SvdLibError, SvdMat};
use sprs::TriMatI;

fn from_triplets(rows: usize, cols: usize, t: &[(usize, usize, f64)]) -> SvdMat<f64> {
    let mut tri = TriMatI::<f64, u32>::new((rows, cols));
    for &(i, j, v) in t {
        tri.add_triplet(i, j, v);
    }
    tri.to_csr::<u64>()
}

fn dense_like(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> SvdMat<f64> {
    let mut tri = TriMatI::<f64, u32>::new((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let v = f(i, j);
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    tri.to_csr::<u64>()
}

// ---------------------------------------------------------------------------
// Structurally degenerate operands
// ---------------------------------------------------------------------------

/// An all-zero matrix has every singular value zero. It must not panic or hang.
#[test]
fn all_zero_matrix() {
    let a = from_triplets(20, 10, &[]);
    assert_eq!(a.nnz(), 0);

    match irlba::svd_seed(&a, 3, 42) {
        Ok(rec) => {
            for &s in rec.s.iter() {
                assert!(s.abs() < 1e-10, "expected zero singular values, got {s}");
            }
        }
        // A typed error is also acceptable: there is no meaningful Krylov subspace.
        Err(SvdLibError::Failed { .. }) => {}
        Err(e) => panic!("unexpected error kind: {e}"),
    }

    match randomized::svd_seed(&a, 3, 42) {
        Ok(rec) => {
            for &s in rec.s.iter() {
                assert!(s.abs() < 1e-10, "expected zero singular values, got {s}");
            }
        }
        Err(SvdLibError::Failed { .. }) | Err(SvdLibError::DenseFactorization { .. }) => {}
        Err(e) => panic!("unexpected error kind: {e}"),
    }
}

/// A matrix of exact rank 1 asked for more triplets than it has.
#[test]
fn rank_deficient_beyond_actual_rank() {
    // Every row is a multiple of [1, 2, 3, 4].
    let a = dense_like(30, 4, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0));

    let rec = irlba::svd_seed(&a, 3, 42).expect("rank-1 matrix, 3 requested");
    assert!(rec.s[0] > 1.0, "dominant value should be substantial");
    for &s in rec.s.iter().skip(1) {
        assert!(s < 1e-8 * rec.s[0], "trailing values should be ~0, got {s}");
    }
    assert!(rec.s.iter().all(|s| s.is_finite()));
}

/// Entirely empty rows and columns.
#[test]
fn zero_rows_and_columns() {
    // Only rows 3 and 7, columns 1 and 5 carry anything.
    let a = from_triplets(
        10,
        8,
        &[(3, 1, 2.0), (3, 5, 3.0), (7, 1, 1.0), (7, 5, -4.0)],
    );
    let rec = irlba::svd_seed(&a, 2, 42).expect("should handle empty rows/cols");
    assert!(rec.s.iter().all(|s| s.is_finite() && *s >= 0.0));
    assert!(rec.s[0] >= rec.s[1]);
    assert!(rec.u.iter().all(|v| v.is_finite()));
    assert!(rec.vt.iter().all(|v| v.is_finite()));
}

/// Duplicate rows make the operand exactly singular in a way that stresses
/// reorthogonalization.
#[test]
fn duplicated_rows() {
    let a = dense_like(40, 12, |i, j| {
        let base = i % 4; // only 4 distinct rows
        ((base * 7 + j * 3) % 11) as f64
    });
    let rec = irlba::svd_seed(&a, 6, 42).expect("duplicated rows");
    assert!(rec.s.iter().all(|s| s.is_finite()));
    // Rank is at most 4, so trailing values must collapse.
    for &s in rec.s.iter().skip(4) {
        assert!(s < 1e-8 * rec.s[0], "value beyond true rank was {s}");
    }
}

/// The smallest operand the API accepts.
#[test]
fn minimal_shapes() {
    let a = from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
    let rec = irlba::svd_seed(&a, 1, 42).expect("2x2 rank 1");
    approx::assert_relative_eq!(rec.s[0], 2.0, max_relative = 1e-10);

    // A single row or column is a valid matrix, if a degenerate one.
    let row = from_triplets(1, 5, &[(0, 0, 3.0), (0, 3, 4.0)]);
    match irlba::svd_seed(&row, 1, 42) {
        Ok(rec) => approx::assert_relative_eq!(rec.s[0], 5.0, max_relative = 1e-8),
        Err(e) => panic!("1xN should work or error cleanly, got {e}"),
    }
}

// ---------------------------------------------------------------------------
// Numerically hostile values
// ---------------------------------------------------------------------------

/// Extreme dynamic range within one matrix.
#[test]
fn wide_dynamic_range() {
    let a = dense_like(50, 20, |i, j| {
        if i == 0 {
            1e12
        } else if i == 1 {
            1e-12
        } else {
            ((i * 3 + j) % 7) as f64
        }
    });
    let rec = irlba::svd_seed(&a, 5, 42).expect("wide dynamic range");
    assert!(rec.s.iter().all(|s| s.is_finite()), "{:?}", rec.s);
    assert!(
        rec.s[0] > 1e11,
        "dominant scale should survive: {}",
        rec.s[0]
    );
    for w in rec.s.to_vec().windows(2) {
        assert!(w[0] >= w[1], "not descending under wide range");
    }
}

/// A NaN anywhere in the operand must not hang or be silently reported as converged.
#[test]
fn nan_input_does_not_hang_or_claim_convergence() {
    let a = from_triplets(20, 10, &[(0, 0, 1.0), (1, 1, f64::NAN), (2, 2, 3.0)]);
    let cfg = irlba::IrlbaConfig::new(2).seed(42).max_restarts(20);
    match irlba::svd_with(&a, &cfg, None) {
        Err(_) => {} // a typed error is the ideal outcome
        Ok(rec) => {
            let poisoned = rec.s.iter().any(|s| s.is_nan())
                || rec.u.iter().any(|v| v.is_nan())
                || rec.vt.iter().any(|v| v.is_nan());
            if poisoned {
                match rec.diagnostics.detail {
                    single_svdlib::Detail::Irlba { converged, .. } => assert!(
                        !converged,
                        "NaN propagated into the result but convergence was reported"
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// The randomized path reaches the dense factorization by a different route than
/// IRLBA, so it needs its own non-finite check.
#[test]
fn randomized_survives_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = from_triplets(30, 12, &[(0, 0, 1.0), (1, 1, bad), (2, 2, 3.0)]);
        let cfg = randomized::RandomizedConfig::new(3)
            .seed(42)
            .power_iterations(2);
        match randomized::svd_with(&a, &cfg, None) {
            Err(_) => {}
            Ok(rec) => assert!(
                rec.s.iter().all(|s| s.is_finite()),
                "returned non-finite singular values for input {bad}"
            ),
        }
        // Block Krylov too.
        let _ = randomized::svd_block_krylov(&a, 3, 2, Some(42));
    }
}

/// An infinity in the operand, same contract.
#[test]
fn infinity_input_does_not_hang() {
    let a = from_triplets(20, 10, &[(0, 0, f64::INFINITY), (1, 1, 2.0)]);
    let cfg = irlba::IrlbaConfig::new(2).seed(42).max_restarts(20);
    let _ = irlba::svd_with(&a, &cfg, None); // must return, either way
}

// ---------------------------------------------------------------------------
// Configuration boundaries
// ---------------------------------------------------------------------------

/// The largest rank each solver will actually accept, and what happens at the boundary.
#[test]
fn maximum_supported_rank() {
    let a = dense_like(30, 20, |i, j| ((i * 5 + j * 3) % 13) as f64 + 0.5);
    let min_dim = 20;

    // One below the dimension must work.
    assert!(
        irlba::svd_seed(&a, min_dim - 1, 42).is_ok(),
        "rank = min_dim - 1 should be supported"
    );

    // Exactly min_dim: whatever the behaviour, it must be a typed error and not a
    // panic, and the message must explain itself.
    match irlba::svd_seed(&a, min_dim, 42) {
        Ok(rec) => assert_eq!(rec.d, min_dim),
        Err(SvdLibError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("work") || msg.contains("rank"),
                "unhelpful message at the rank boundary: {msg}"
            );
        }
        Err(e) => panic!("unexpected error kind at rank = min_dim: {e}"),
    }

    // Randomized has no such restriction and should reach full rank.
    assert!(
        randomized::svd_seed(&a, min_dim, 42).is_ok(),
        "randomized should support rank = min_dim"
    );
}

/// A restart budget too small to converge must be an error, not a quietly degraded
/// result — a pipeline that forgets to inspect the diagnostics would otherwise feed a
/// best-effort decomposition into whatever comes next.
#[test]
fn exhausted_restart_budget_fails_loudly() {
    // A near-flat spectrum needs many restarts.
    let a = dense_like(200, 100, |i, j| (((i * 31 + j * 17) % 101) as f64) - 50.0);
    let cfg = irlba::IrlbaConfig::new(30)
        .seed(42)
        .tol(1e-14)
        .max_restarts(1);

    match irlba::svd_with(&a, &cfg, None) {
        Err(SvdLibError::Failed { stage, message }) => {
            assert_eq!(stage, "irlba");
            assert!(
                message.contains("did not converge") && message.contains("residual"),
                "error should say what happened and what to do: {message}"
            );
        }
        Err(e) => panic!("wrong error kind: {e}"),
        // Converging in one restart is acceptable; silently returning garbage is not.
        Ok(rec) => assert!(rec.converged(), "returned an unconverged result as Ok"),
    }

    // Opting in to a best effort must still work, and must say so.
    let lax = cfg.clone().allow_unconverged();
    let rec = irlba::svd_with(&a, &lax, None).expect("best effort should be available");
    if !rec.converged() {
        let resid = rec.max_residual().expect("irlba tracks a residual");
        assert!(
            resid > 0.0,
            "unconverged result must carry a positive residual"
        );
    }
}

/// `converged()` must be reachable without matching on the diagnostics enum.
#[test]
fn convergence_is_visible_without_pattern_matching() {
    let a = dense_like(120, 40, |i, j| ((i * 3 + j) % 17) as f64);
    let rec = irlba::svd_seed(&a, 8, 42).unwrap();
    assert!(rec.converged());
    assert!(rec.max_residual().is_some());

    // The randomized methods complete by construction.
    let r = randomized::svd_seed(&a, 8, 42).unwrap();
    assert!(r.converged());
    assert!(r.max_residual().is_none());
}

/// `work` smaller than `rank` is incoherent and must be rejected, not silently fixed
/// into something that returns the wrong number of triplets.
#[test]
fn incoherent_work_size() {
    let a = dense_like(60, 30, |i, j| ((i + j) % 5) as f64);
    let cfg = irlba::IrlbaConfig::new(10).seed(42).work(3);
    match irlba::svd_with(&a, &cfg, None) {
        Ok(rec) => assert_eq!(rec.d, 10, "returned a different rank than requested"),
        Err(SvdLibError::InvalidArgument(_)) => {}
        Err(e) => panic!("unexpected error kind: {e}"),
    }
}

/// Zero oversampling is legal but marginal.
#[test]
fn zero_oversampling() {
    let a = dense_like(100, 40, |i, j| ((i * 7 + j) % 11) as f64);
    let cfg = randomized::RandomizedConfig::new(5).seed(42).oversamples(0);
    let rec = randomized::svd_with(&a, &cfg, None).expect("zero oversampling");
    assert_eq!(rec.d, 5);
    assert!(rec.s.iter().all(|s| s.is_finite()));
}

/// An empty mask leaves a zero-column operand.
#[test]
fn empty_column_mask() {
    let a = dense_like(30, 10, |i, j| ((i + j) % 3) as f64);
    let masked = MaskedCsMat::with_columns(&a, &[]);
    assert_eq!(masked.cols(), 0);
    // Any rank is out of range for a zero-column matrix.
    assert!(irlba::svd(&masked, 1).is_err());
}

/// A mask selecting a single column.
#[test]
fn single_column_mask() {
    let a = dense_like(30, 10, |i, j| ((i * 3 + j) % 7) as f64 + 1.0);
    let masked = MaskedCsMat::with_columns(&a, &[4]);
    assert_eq!(masked.cols(), 1);
    match irlba::svd(&masked, 1) {
        Ok(rec) => {
            assert_eq!(rec.d, 1);
            assert!(rec.s[0].is_finite() && rec.s[0] > 0.0);
        }
        Err(e) => panic!("single-column mask should work, got {e}"),
    }
}

// ---------------------------------------------------------------------------
// Execution environment
// ---------------------------------------------------------------------------

/// Results must not depend on the thread count — the parallel kernels reduce in a
/// different order per thread count, so this pins that the difference stays at
/// rounding level.
#[test]
fn results_are_stable_across_thread_counts() {
    let a = dense_like(400, 80, |i, j| (((i * 13 + j * 7) % 23) as f64) - 11.0);

    let mut spectra = Vec::new();
    for threads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let rec = pool.install(|| irlba::svd_seed(&a, 10, 42).unwrap());
        spectra.push(rec.s.to_vec());
    }
    for k in 1..spectra.len() {
        for i in 0..spectra[0].len() {
            let rel = (spectra[k][i] - spectra[0][i]).abs() / spectra[0][i].abs().max(1e-30);
            assert!(
                rel < 1e-8,
                "thread count changed singular value {i}: {:.12e} vs {:.12e}",
                spectra[k][i],
                spectra[0][i]
            );
        }
    }
}

/// The scatter kernel's column blocking must not change the answer, at any budget.
#[test]
fn scratch_budget_does_not_change_results() {
    use ndarray::Array2;
    use single_svdlib::matrix::kernels::{scatter_mul, DEFAULT_SCRATCH_BUDGET};

    let a = dense_like(500, 120, |i, j| (((i * 11 + j * 5) % 17) as f64) - 8.0);
    let rhs = Array2::from_shape_fn((500, 24), |(i, j)| ((i * 3 + j) % 9) as f64 - 4.0);

    let mut reference = Array2::zeros((120, 24));
    scatter_mul(a.view(), rhs.view(), reference.view_mut(), usize::MAX);

    for budget in [0usize, 1, 1024, DEFAULT_SCRATCH_BUDGET] {
        let mut got = Array2::zeros((120, 24));
        scatter_mul(a.view(), rhs.view(), got.view_mut(), budget);
        for (x, y) in got.iter().zip(reference.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-12);
        }
    }
}
