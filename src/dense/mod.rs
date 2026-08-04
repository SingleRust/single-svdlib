//! Dense helpers: tall-skinny QR, and small factorizations on the reduced matrices the
//! Krylov and randomized methods produce.

pub mod jacobi;
pub mod tsqr;

pub use tsqr::{orthogonality_error, orthonormalize, tsqr};

use crate::error::{Result, SvdLibError};
use crate::types::SvdFloat;
use ndarray::{Array1, Array2, ArrayView2};

/// A thin SVD of a small dense matrix: `a ≈ u · diag(s) · vt`, `s` descending.
pub struct SmallSvd<T> {
    /// `m × k`, `k = min(m, n)`.
    pub u: Array2<T>,
    /// Length `k`, descending.
    pub s: Array1<T>,
    /// `k × n`.
    pub vt: Array2<T>,
}

/// Thin SVD of a small dense matrix.
///
/// Always computed in `f64` and cast back, whatever `T` is. These operands are `l × l`
/// or `l × n` with `l` on the order of the requested rank, so the widening costs
/// nothing measurable and it keeps `f32` callers from losing accuracy in the one place
/// where the whole result's conditioning is decided.
///
/// Uses [one-sided Jacobi](jacobi) rather than the bidiagonal QR a linear-algebra
/// backend would provide. Jacobi is accurate to the condition number *after* column
/// scaling, so a badly scaled reduced factor — which is what an ill-conditioned Krylov
/// basis produces — is still factored to full relative accuracy. Golub–Reinsch is not:
/// on a `5 × 2` operand with `κ ≈ 8·10⁶`, `nalgebra`'s implementation reconstructed to
/// only `1.3·10⁻⁹` relative, against Jacobi's `< 1·10⁻¹⁵`.
pub fn small_svd<T: SvdFloat>(a: ArrayView2<T>) -> Result<SmallSvd<T>> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(SvdLibError::shape(format!(
            "small_svd needs a non-empty matrix, got {m}x{n}"
        )));
    }
    let k = m.min(n);

    // Fail fast and legibly on a poisoned operand, rather than letting it reach the
    // factorization and surface as an opaque non-convergence.
    if let Some((i, j)) = a
        .indexed_iter()
        .find(|(_, v)| !num_traits::Float::is_finite(**v))
        .map(|(idx, _)| idx)
    {
        return Err(SvdLibError::DenseFactorization {
            factorization: "SVD",
            message: format!("operand is not finite at ({i}, {j})"),
        });
    }

    let a64 = Array2::<f64>::from_shape_fn((m, n), |(i, j)| a[[i, j]].to_f64());
    let fail = || SvdLibError::DenseFactorization {
        factorization: "SVD",
        message: format!("one-sided Jacobi did not converge on a {m}x{n} operand"),
    };

    // Jacobi needs at least as many rows as columns. For a wide operand factor the
    // transpose instead: `Aᵀ = U·Σ·Vᵀ` gives `A = V·Σ·Uᵀ`.
    let (u64, s64, vt64) = if m >= n {
        let j = jacobi::jacobi_svd(&a64).ok_or_else(fail)?;
        (j.u, j.s, j.v.t().to_owned())
    } else {
        let j = jacobi::jacobi_svd(&a64.t().to_owned()).ok_or_else(fail)?;
        (j.v, j.s, j.u.t().to_owned())
    };
    debug_assert_eq!(u64.dim(), (m, k));
    debug_assert_eq!(vt64.dim(), (k, n));

    // Jacobi already returns the singular values descending.
    let mut out_u = Array2::<T>::zeros((m, k));
    let mut out_s = Array1::<T>::zeros(k);
    let mut out_vt = Array2::<T>::zeros((k, n));
    for idx in 0..k {
        out_s[idx] = T::from_f64_val(s64[idx]);
        for i in 0..m {
            out_u[[i, idx]] = T::from_f64_val(u64[[i, idx]]);
        }
        for j in 0..n {
            out_vt[[idx, j]] = T::from_f64_val(vt64[[idx, j]]);
        }
    }
    Ok(SmallSvd {
        u: out_u,
        s: out_s,
        vt: out_vt,
    })
}

/// Flip the sign of each singular-vector pair so the dominant entry of every column of
/// `u` is positive.
///
/// The SVD is only unique up to a per-triplet sign, so two runs can disagree on it for
/// no numerical reason. Pinning the sign makes results reproducible and comparable —
/// this is what `sklearn`'s `svd_flip` is for.
pub fn svd_flip<T: SvdFloat>(u: &mut Array2<T>, vt: &mut Array2<T>) {
    let k = u.ncols().min(vt.nrows());
    for j in 0..k {
        // Locate the largest-magnitude entry of column j.
        let mut best = T::zero();
        let mut sign = T::one();
        for i in 0..u.nrows() {
            let v = u[[i, j]];
            let mag = num_traits::Float::abs(v);
            if mag > best {
                best = mag;
                sign = if v < T::zero() { -T::one() } else { T::one() };
            }
        }
        if sign < T::zero() {
            for i in 0..u.nrows() {
                u[[i, j]] = -u[[i, j]];
            }
            for j2 in 0..vt.ncols() {
                vt[[j, j2]] = -vt[[j, j2]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Lcg;

    #[test]
    fn small_svd_reconstructs_and_is_ordered() {
        let mut rng = Lcg::new(11);
        let a = Array2::from_shape_fn((9, 6), |_| rng.signed());
        let svd = small_svd(a.view()).unwrap();

        assert_eq!(svd.u.dim(), (9, 6));
        assert_eq!(svd.s.len(), 6);
        assert_eq!(svd.vt.dim(), (6, 6));

        for w in svd.s.to_vec().windows(2) {
            assert!(w[0] >= w[1], "singular values not descending: {:?}", svd.s);
        }

        let scaled = &svd.u * &svd.s.view().insert_axis(ndarray::Axis(0));
        let recon = scaled.dot(&svd.vt);
        let err: f64 = (&recon - &a).iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err / scale < 1e-12,
            "relative reconstruction {}",
            err / scale
        );
    }

    /// `f32` input must still be factored at `f64` precision internally.
    #[test]
    fn small_svd_promotes_f32() {
        let mut rng = Lcg::new(13);
        let a = Array2::from_shape_fn((7, 5), |_| rng.signed() as f32);
        let svd = small_svd(a.view()).unwrap();
        let scaled = &svd.u * &svd.s.view().insert_axis(ndarray::Axis(0));
        let recon = scaled.dot(&svd.vt);
        let err: f32 = (&recon - &a).iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            err / scale < 1e-5,
            "relative reconstruction {}",
            err / scale
        );
    }

    #[test]
    fn svd_flip_makes_signs_deterministic() {
        let mut u = ndarray::arr2(&[[-3.0f64, 1.0], [1.0, -4.0]]);
        let mut vt = ndarray::arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let before = u.dot(&vt);
        svd_flip(&mut u, &mut vt);
        // Dominant entry of each column of u is now positive.
        assert!(u[[0, 0]] > 0.0);
        assert!(u[[1, 1]] > 0.0);
        // The product is unchanged: flipping a column of u and the matching row of vt
        // cancels.
        let after = u.dot(&vt);
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-14);
        }
    }

    #[test]
    fn small_svd_rejects_empty() {
        let a = Array2::<f64>::zeros((0, 3));
        assert!(small_svd(a.view()).is_err());
    }
}
