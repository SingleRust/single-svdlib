//! Result and scalar types shared by every algorithm in the crate.

use ndarray::{Array1, Array2};
use single_utilities::traits::FloatOpsTS;

/// Scalar types this crate can decompose.
///
/// Implemented for `f32` and `f64`. The supertraits are what the sparse kernels and
/// the ndarray glue need; nothing here leaks a linear-algebra backend, because the
/// small dense factorizations are always performed in `f64` and cast back.
pub trait SvdFloat:
    FloatOpsTS + ndarray::ScalarOperand + sprs::MulAcc + std::ops::DivAssign + 'static
{
    /// Machine epsilon.
    fn eps() -> Self;
    /// `eps^(3/4)`, the accuracy floor LAS2 clamps `kappa` to.
    fn eps34() -> Self;
    /// Widen to `f64` for the small dense factorizations.
    fn to_f64(self) -> f64;
    /// Narrow back from `f64`.
    fn from_f64_val(v: f64) -> Self;
    /// Equality within one ulp-ish epsilon.
    fn close(a: Self, b: Self) -> bool {
        num_traits::Float::abs(b - a) < Self::eps()
    }
}

impl SvdFloat for f32 {
    #[inline]
    fn eps() -> Self {
        f32::EPSILON
    }
    #[inline]
    fn eps34() -> Self {
        // Constant-folded rather than powf'd on every call; `eps34_constants_match_computed`
        // pins these to `EPSILON.powf(0.75)`.
        const V: f32 = 6.4155306e-6;
        V
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64_val(v: f64) -> Self {
        v as f32
    }
}

impl SvdFloat for f64 {
    #[inline]
    fn eps() -> Self {
        f64::EPSILON
    }
    #[inline]
    fn eps34() -> Self {
        const V: f64 = 1.8189894035458565e-12;
        V
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64_val(v: f64) -> Self {
        v
    }
}

/// Which algorithm produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Single-vector Lanczos, the SVDLIBC LAS2 port.
    Las2,
    /// Restarted Lanczos bidiagonalization.
    Irlba,
    /// Randomized range finder with power iterations.
    Randomized,
    /// Randomized block Krylov.
    BlockKrylov,
}

/// Algorithm-specific counters. The fields common to every method live on
/// [`Diagnostics`] itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Detail<T> {
    Lanczos {
        iterations: usize,
        lanczos_steps: usize,
        ritz_values_stabilized: usize,
        end_interval: [T; 2],
        kappa: T,
    },
    Irlba {
        restarts: usize,
        converged: bool,
        tolerance: T,
        /// Largest residual `||A v - s u||` over the returned triplets.
        max_residual: T,
    },
    Randomized {
        oversamples: usize,
        power_iterations: usize,
        block_size: usize,
    },
}

/// What the computation did, for logging and for deciding whether to trust a result.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics<T> {
    pub algorithm: Algorithm,
    /// Non-zeros in the input.
    pub non_zero: usize,
    /// Dimensions requested by the caller.
    pub dimensions: usize,
    /// Dimensions actually returned and considered significant.
    pub significant_values: usize,
    /// Whether the algorithm worked on the transpose internally.
    pub transposed: bool,
    pub random_seed: u64,
    /// Sparse matrix-vector products performed, counting a block product against `k`
    /// dense columns as `k`. Comparable across algorithms, so it is the honest way to
    /// price one method against another.
    pub matvecs: usize,
    pub detail: Detail<T>,
}

/// A singular value decomposition.
///
/// # Orientation
///
/// `A ≈ u · diag(s) · vt`, matching the `numpy.linalg.svd` / `scipy` convention:
///
/// - `u` is `m × d` — left singular vectors are **columns**
/// - `s` is `d`, descending
/// - `vt` is `d × n` — right singular vectors are **rows**
///
/// In 1.x this was inconsistent: the Lanczos path returned `u` transposed (`d × m`)
/// while the randomized path returned it as `m × d`, so [`SvdRec::recompose`] only
/// worked for square inputs. Both paths now follow the convention above.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdRec<T> {
    /// Number of singular triplets returned.
    pub d: usize,
    /// Left singular vectors, `m × d`.
    pub u: Array2<T>,
    /// Singular values, length `d`, descending.
    pub s: Array1<T>,
    /// Transposed right singular vectors, `d × n`.
    pub vt: Array2<T>,
    pub diagnostics: Diagnostics<T>,
}

impl<T: SvdFloat> SvdRec<T> {
    /// Rebuild the dense approximation `u · diag(s) · vt`.
    ///
    /// Allocates an `m × n` dense matrix — only reasonable for small inputs or for
    /// checking reconstruction error in tests.
    pub fn recompose(&self) -> Array2<T> {
        let scaled = &self.u * &self.s.view().insert_axis(ndarray::Axis(0));
        scaled.dot(&self.vt)
    }

    /// Whether an iterative method reached its tolerance.
    ///
    /// Always `true` for the randomized methods: they perform a fixed amount of work and
    /// complete by construction, and their accuracy is governed by the sketch size and
    /// power iterations rather than by a convergence test. For [`Algorithm::Irlba`] this
    /// is the real thing — `false` means the restart budget ran out and the triplets are
    /// a best effort.
    ///
    /// [`crate::irlba`] refuses to return an unconverged result by default, so this is a
    /// belt-and-braces check for callers who opted out of that.
    pub fn converged(&self) -> bool {
        match self.diagnostics.detail {
            Detail::Irlba { converged, .. } => converged,
            Detail::Lanczos { .. } | Detail::Randomized { .. } => true,
        }
    }

    /// The largest residual `‖A·vᵢ − σᵢ·uᵢ‖` over the returned triplets, when the
    /// algorithm tracks one.
    ///
    /// Compare against `s[0]` to judge it: a residual of `1e-9 · σ_max` is excellent, one
    /// of `0.1 · σ_max` means the answer is not usable.
    pub fn max_residual(&self) -> Option<T> {
        match self.diagnostics.detail {
            Detail::Irlba { max_residual, .. } => Some(max_residual),
            _ => None,
        }
    }

    /// Number of rows of the original matrix.
    pub fn nrows(&self) -> usize {
        self.u.nrows()
    }

    /// Number of columns of the original matrix.
    pub fn ncols(&self) -> usize {
        self.vt.ncols()
    }

    /// Truncate to the leading `k` triplets in place.
    pub fn truncate(&mut self, k: usize) {
        let k = k.min(self.d);
        if k == self.d {
            return;
        }
        self.u = self.u.slice(ndarray::s![.., ..k]).to_owned();
        self.s = self.s.slice(ndarray::s![..k]).to_owned();
        self.vt = self.vt.slice(ndarray::s![..k, ..]).to_owned();
        self.d = k;
        self.diagnostics.significant_values = k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps34_constants_match_computed() {
        // The hardcoded constants must equal the expression they replaced.
        approx::assert_relative_eq!(f32::eps34(), f32::EPSILON.powf(0.75), max_relative = 1e-6);
        approx::assert_relative_eq!(f64::eps34(), f64::EPSILON.powf(0.75), max_relative = 1e-12);
    }

    #[test]
    fn recompose_is_orientation_correct_for_non_square() {
        // A 3x2 rank-1 matrix: u (3x1), s (1), vt (1x2).
        let u = ndarray::arr2(&[[1.0f64], [2.0], [3.0]]);
        let s = ndarray::arr1(&[2.0f64]);
        let vt = ndarray::arr2(&[[1.0f64, 10.0]]);
        let rec = SvdRec {
            d: 1,
            u,
            s,
            vt,
            diagnostics: Diagnostics {
                algorithm: Algorithm::Las2,
                non_zero: 6,
                dimensions: 1,
                significant_values: 1,
                transposed: false,
                random_seed: 0,
                matvecs: 0,
                detail: Detail::Lanczos {
                    iterations: 0,
                    lanczos_steps: 0,
                    ritz_values_stabilized: 0,
                    end_interval: [0.0, 0.0],
                    kappa: 0.0,
                },
            },
        };
        let a = rec.recompose();
        assert_eq!(a.dim(), (3, 2));
        // row i = u[i] * s * vt
        assert_eq!(a[[0, 0]], 2.0);
        assert_eq!(a[[0, 1]], 20.0);
        assert_eq!(a[[2, 0]], 6.0);
        assert_eq!(a[[2, 1]], 60.0);
    }
}
