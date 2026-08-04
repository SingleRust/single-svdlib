//! One-sided Jacobi SVD.
//!
//! # Why not the bidiagonal QR the linear-algebra backend provides
//!
//! Golub–Reinsch (what `nalgebra::SVD` implements, and LAPACK's `gesvd`) first reduces
//! to bidiagonal form. That reduction mixes columns of wildly different norm, so the
//! small singular values inherit an absolute error proportional to `‖A‖` rather than to
//! themselves. On a `5 × 2` operand with `κ ≈ 8·10⁶` this crate measured
//! `‖A − UΣVᵀ‖ / ‖A‖ ≈ 1.3·10⁻⁹` — nine orders worse than a backward-stable
//! factorization should give, and independent of the iteration limit or tolerance.
//!
//! One-sided Jacobi never forms a bidiagonal. It rotates *pairs of columns* until they
//! are mutually orthogonal, at which point the column norms are the singular values.
//! Demmel & Veselić showed this is accurate to `O(ε · κ(A·D⁻¹))` — the condition number
//! *after* optimal column scaling — so a matrix that is merely badly scaled, which is
//! exactly what an ill-conditioned Krylov basis looks like, is factored to full relative
//! accuracy.
//!
//! That matters here because [`small_svd`](super::small_svd) is applied to the reduced
//! `B` (IRLBA) and `R` (randomized) factors, whose conditioning is inherited from the
//! operand and decides the accuracy of everything downstream.
//!
//! The operands are `l × l` with `l` on the order of the requested rank, so the extra
//! sweeps cost nothing measurable next to the sparse products.

use ndarray::{Array1, Array2};

/// Sweeps before giving up. Jacobi converges quadratically; more than a dozen sweeps
/// means the operand is pathological.
const MAX_SWEEPS: usize = 60;

/// A thin SVD computed by one-sided Jacobi: `a = u · diag(s) · vᵀ`, `s` descending.
pub struct JacobiSvd {
    /// `m × n` with orthonormal columns.
    pub u: Array2<f64>,
    /// Length `n`, descending, non-negative.
    pub s: Array1<f64>,
    /// `n × n` orthogonal; the *right* vectors as columns.
    pub v: Array2<f64>,
}

/// One-sided Jacobi SVD of a tall-or-square matrix (`m >= n`).
///
/// Returns `None` only on a non-finite operand. Exhausting the sweep budget is not an
/// error: the iterate in hand is a valid partial factorization with very nearly
/// orthogonal columns, and returning it beats failing the caller's whole decomposition.
pub fn jacobi_svd(a: &Array2<f64>) -> Option<JacobiSvd> {
    let (m, n) = a.dim();
    debug_assert!(
        m >= n,
        "one-sided Jacobi needs at least as many rows as columns"
    );

    let mut w = a.clone(); // becomes U·Σ
    let mut v = Array2::<f64>::eye(n);
    // Convergence threshold on the inter-column cosine.
    let tol = f64::EPSILON * (m as f64).sqrt();

    // Flush columns that are negligible against the largest to exact zero.
    //
    // A column with norm around `1e-156` next to one around `1e0` carries no information
    // — every singular value it could contribute is far below the rounding floor of the
    // rest — but it does drive the rotation arithmetic into the denormal range, where
    // the pair can never satisfy any orthogonality test and the sweeps churn forever.
    // Zeroing it here routes it through the rank-deficiency path instead, which gives it
    // a proper orthonormal left vector.
    {
        let norms: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| w[[i, j]] * w[[i, j]]).sum::<f64>().sqrt())
            .collect();
        let biggest = norms.iter().copied().fold(0.0f64, f64::max);
        if biggest > 0.0 {
            let cutoff = biggest * f64::EPSILON * f64::EPSILON;
            for j in 0..n {
                if norms[j] <= cutoff {
                    for i in 0..m {
                        w[[i, j]] = 0.0;
                    }
                }
            }
        }
    }

    // Rotate every column pair until they are mutually orthogonal.
    for _ in 0..MAX_SWEEPS {
        let mut rotations = 0usize;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let mut app = 0.0;
                let mut aqq = 0.0;
                let mut apq = 0.0;
                for i in 0..m {
                    let (x, y) = (w[[i, p]], w[[i, q]]);
                    app += x * x;
                    aqq += y * y;
                    apq += x * y;
                }
                if !apq.is_finite() || !app.is_finite() || !aqq.is_finite() {
                    return None;
                }
                // A zero column is orthogonal to everything by definition, and the
                // relative test below would divide by its (zero) norm.
                if app <= 0.0 || aqq <= 0.0 {
                    continue;
                }
                // Orthogonal enough already? The test is on the *cosine* between the two
                // columns, relative to their norms — that is what makes the method
                // relatively accurate, and it is the quantity that becomes an entry of
                // `UᵀU` once the columns are normalised.
                //
                // The threshold carries a `sqrt(m)` factor rather than being bare `eps`.
                // At exactly `eps` the cosine can hover on the boundary, each rotation
                // re-injecting O(eps) error into the pair it just fixed, and the sweep
                // never reports zero rotations. Do *not* instead skip on a small rotation
                // angle: `s ≈ 1/(2·tau)` is tiny whenever the two columns differ greatly
                // in norm, even when their cosine is nowhere near zero, so that test
                // silently abandons badly scaled pairs and leaves `U` orthogonal to only
                // ~1e-10.
                //
                // `sqrt(app) * sqrt(aqq)`, never `sqrt(app * aqq)`: the product
                // underflows to exactly zero once the column norms are small enough
                // (`app = 9e-312` against `aqq = 2e-158` is reachable from ordinary
                // input), which turns the test into `|apq| <= 0`. No non-zero cosine can
                // satisfy that, so the pair rotates on every sweep forever and the
                // factorization reports non-convergence.
                if apq == 0.0 || apq.abs() <= tol * app.sqrt() * aqq.sqrt() {
                    continue;
                }
                rotations += 1;

                // The rotation that zeroes the (p, q) entry of the 2x2 Gram matrix.
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m {
                    let (x, y) = (w[[i, p]], w[[i, q]]);
                    w[[i, p]] = c * x - s * y;
                    w[[i, q]] = s * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[[i, p]], v[[i, q]]);
                    v[[i, p]] = c * x - s * y;
                    v[[i, q]] = s * x + c * y;
                }
            }
        }
        if rotations == 0 {
            break;
        }
    }
    // The column norms are the singular values; normalising gives U.
    let mut sigma = Array1::<f64>::zeros(n);
    for j in 0..n {
        let norm = (0..m).map(|i| w[[i, j]] * w[[i, j]]).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return None;
        }
        sigma[j] = norm;
    }

    // Descending, carrying the vectors along.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // A column whose norm has fallen to the rounding floor holds noise, not direction:
    // its entries are the residue of cancellation and are *not* orthogonal to the other
    // columns. Normalising it would amplify that noise into a unit vector and destroy
    // the orthonormality of `u` — on a matrix with a duplicated column this produced
    // `||UᵀU − I|| = 0.6`. The test has to be relative to the largest singular value.
    let smax = sigma.iter().copied().fold(0.0f64, f64::max);
    let floor = f64::EPSILON * smax * (m as f64).sqrt();

    let mut u = Array2::<f64>::zeros((m, n));
    let mut s_out = Array1::<f64>::zeros(n);
    let mut v_out = Array2::<f64>::zeros((n, n));
    let mut deficient = Vec::new();
    for (new, &old) in order.iter().enumerate() {
        s_out[new] = sigma[old];
        for i in 0..n {
            v_out[[i, new]] = v[[i, old]];
        }
        if sigma[old] > floor {
            let inv = 1.0 / sigma[old];
            for i in 0..m {
                u[[i, new]] = w[[i, old]] * inv;
            }
        } else {
            deficient.push(new);
        }
    }

    // A numerically zero singular value leaves its left vector undefined. Callers rotate
    // bases with `u`, so it has to be a genuine orthonormal basis rather than have holes
    // in it: complete the deficient columns against the ones that are defined.
    complete_orthonormal_basis(&mut u, &deficient);

    Some(JacobiSvd {
        u,
        s: s_out,
        v: v_out,
    })
}

/// Replace the listed columns of `u` with unit vectors orthogonal to every other column.
fn complete_orthonormal_basis(u: &mut Array2<f64>, deficient: &[usize]) {
    let (m, n) = u.dim();
    if deficient.is_empty() {
        return;
    }
    // For each hole, test *every* canonical direction and take the one that survives
    // projection best. Accepting the first merely-nonzero candidate is not enough: a
    // direction that is nearly dependent on the existing columns leaves a tiny residual,
    // and normalising it scales the Gram-Schmidt error up by the reciprocal of that
    // residual. Accepting a residual of `1e-8` therefore admits an orthogonality error
    // of the same order, which showed up as `||UᵀU - I|| = 2e-7`. Picking the largest
    // residual keeps the amplification at O(1).
    let mut used = vec![false; m];
    for &j in deficient {
        let mut best: Option<(f64, Array1<f64>, usize)> = None;
        for cand in 0..m {
            if used[cand] {
                continue;
            }
            let mut trial = Array1::<f64>::zeros(m);
            trial[cand] = 1.0;

            // Orthogonalise against every column already established, twice.
            for _ in 0..2 {
                for k in 0..n {
                    if k == j {
                        continue;
                    }
                    let dot: f64 = (0..m).map(|i| u[[i, k]] * trial[i]).sum();
                    if dot != 0.0 {
                        for i in 0..m {
                            trial[i] -= dot * u[[i, k]];
                        }
                    }
                }
            }
            let norm = trial.iter().map(|x| x * x).sum::<f64>().sqrt();
            if best.as_ref().is_none_or(|(b, _, _)| norm > *b) {
                best = Some((norm, trial, cand));
            }
            // A residual this large cannot be improved on meaningfully; stop early.
            if norm > 0.9 {
                break;
            }
        }
        if let Some((norm, trial, cand)) = best {
            if norm > 0.0 {
                used[cand] = true;
                for i in 0..m {
                    u[[i, j]] = trial[i] / norm;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Lcg;

    fn frob(a: &Array2<f64>) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn recompose(svd: &JacobiSvd) -> Array2<f64> {
        let scaled = &svd.u * &svd.s.view().insert_axis(ndarray::Axis(0));
        scaled.dot(&svd.v.t())
    }

    /// The exact counterexample proptest shrank to, on which nalgebra's Golub-Reinsch
    /// reconstructs to only ~1.3e-9 relative. Jacobi must do far better.
    #[test]
    fn beats_bidiagonal_qr_on_the_ill_conditioned_counterexample() {
        let a = ndarray::arr2(&[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.8977478193857099, -1.0],
            [0.0, 7255691.862956913],
        ]);
        let svd = jacobi_svd(&a).expect("should converge");
        let err = frob(&(&recompose(&svd) - &a)) / frob(&a);
        assert!(
            err < 1e-15,
            "Jacobi reconstruction {err:.3e} is no better than bidiagonal QR's 1.3e-9"
        );
        let orth = crate::dense::orthogonality_error(&svd.u.view());
        assert!(orth < 1e-13, "||U^T U - I|| = {orth:.3e}");
    }

    #[test]
    fn matches_a_reference_on_well_conditioned_input() {
        let mut rng = Lcg::new(3);
        let a = Array2::from_shape_fn((12, 8), |_| rng.signed());
        let svd = jacobi_svd(&a).unwrap();

        let m = nalgebra::DMatrix::from_fn(12, 8, |i, j| a[[i, j]]);
        let mut want: Vec<f64> = m.singular_values().iter().copied().collect();
        want.sort_by(|x, y| y.partial_cmp(x).unwrap());

        for (i, &g) in svd.s.iter().enumerate() {
            approx::assert_relative_eq!(g, want[i], max_relative = 1e-12);
        }
        let err = frob(&(&recompose(&svd) - &a)) / frob(&a);
        assert!(err < 1e-14, "reconstruction {err:.3e}");
    }

    /// A rank-deficient operand: the zero singular values must still come with an
    /// orthonormal `u`, since callers rotate bases with it.
    #[test]
    fn rank_deficient_still_yields_an_orthonormal_basis() {
        // Column 2 is a copy of column 0, column 4 is zero.
        let mut rng = Lcg::new(5);
        let mut a = Array2::from_shape_fn((10, 5), |_| rng.signed());
        let c0 = a.column(0).to_owned();
        a.column_mut(2).assign(&c0);
        a.column_mut(4).fill(0.0);

        let svd = jacobi_svd(&a).unwrap();
        assert!(
            svd.s[4] < 1e-14,
            "expected a zero singular value, got {}",
            svd.s[4]
        );

        let orth = crate::dense::orthogonality_error(&svd.u.view());
        assert!(
            orth < 1e-12,
            "||U^T U - I|| = {orth:.3e} on a deficient operand"
        );

        let err = frob(&(&recompose(&svd) - &a)) / frob(&a);
        assert!(err < 1e-14, "reconstruction {err:.3e}");
    }

    #[test]
    fn all_zero_operand() {
        let a = Array2::<f64>::zeros((6, 3));
        let svd = jacobi_svd(&a).unwrap();
        assert!(svd.s.iter().all(|&v| v == 0.0));
        let orth = crate::dense::orthogonality_error(&svd.u.view());
        assert!(
            orth < 1e-12,
            "||U^T U - I|| = {orth:.3e} on the zero matrix"
        );
    }

    #[test]
    fn single_column() {
        let a = ndarray::arr2(&[[3.0], [4.0]]);
        let svd = jacobi_svd(&a).unwrap();
        approx::assert_relative_eq!(svd.s[0], 5.0, max_relative = 1e-14);
    }

    #[test]
    fn rejects_non_finite() {
        let a = ndarray::arr2(&[[1.0, 0.0], [0.0, f64::NAN]]);
        assert!(jacobi_svd(&a).is_none());
    }
}
