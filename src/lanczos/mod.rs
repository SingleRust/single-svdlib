//! Single-vector Lanczos with selective reorthogonalization — a port of LAS2 from
//! Doug Rohde's SVDLIBC.
//!
//! # ⚠ This module is deprecated and numerically unreliable
//!
//! LAS2 as implemented here does **not** agree with a dense LAPACK reference on any
//! matrix class tested. The largest singular value comes back with 18%–100% relative
//! error, including on `diag(n, n-1, ..., 1)` at full requested rank. The defect is
//! inherited from published 1.x, not introduced by the sprs port — running
//! `single-svdlib 1.0.9` on identical fixtures reproduces the same wrong values.
//!
//! Two causes are known:
//!
//! 1. **Fixed.** `imtqlb` hoisted its shift origin out of the iteration loop, so every
//!    eigenvalue after the first used a stale shift. EISPACK `IMTQL1` assigns
//!    `p = d(l)` inside the loop. This is what produced the "imtqlb had some
//!    convergence issues" warnings 1.x printed on nearly every input before continuing
//!    with corrupted Ritz values.
//! 2. **Open.** `ritvec` reads `s[k*js + i]` — row `k` — while `imtql2` stores
//!    eigenvectors as columns. Transposing roughly halves the residual error but does
//!    not eliminate it, so at least one further defect remains.
//!
//! Use [`crate::irlba`] instead: restarted Lanczos bidiagonalization, validated against
//! LAPACK, with a Krylov basis bounded by the requested rank rather than growing to
//! `min(rows, cols)`.
//!
//! The module is retained so 2.0 does not silently drop the API, and so the repair has
//! a home. The accuracy tests are present but `#[ignore]`d, and
//! `report_accuracy_vs_lapack` prints the current error profile.

// Numeric kernels index several arrays in step from one loop variable, and
// offset arithmetic is load-bearing; iterator rewrites obscure which array an
// index belongs to.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_checked_ops)]

use crate::error::{Result, SvdLibError};
use crate::matrix::SparseMat;
use crate::types::{Algorithm, Detail, Diagnostics, SvdFloat, SvdRec};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{rng, Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use std::cell::Cell;
use std::mem;

const MAXLL: usize = 2;
const MAX_QL_ITERATIONS: usize = 100;

/// Default end interval: eigenvalues inside it are considered unwanted.
pub const DEFAULT_END_INTERVAL: [f64; 2] = [-1.0e-30, 1.0e-30];
/// Default relative accuracy for accepting a Ritz value as an eigenvalue.
pub const DEFAULT_KAPPA: f64 = 1.0e-6;

/// SVD at full dimensionality with default tolerances.
#[deprecated(
    since = "2.0.0",
    note = "LAS2 is numerically unreliable (18%-100% error vs LAPACK); use `single_svdlib::irlba` instead. See the module docs."
)]
pub fn svd<T: SvdFloat, M: SparseMat<T>>(a: &M) -> Result<SvdRec<T>> {
    #[allow(deprecated)]
    svd_dim_seed(a, 0, 0)
}

/// SVD at the requested dimensionality with default tolerances.
///
/// `dimensions == 0` means `min(rows, cols)`.
#[deprecated(
    since = "2.0.0",
    note = "LAS2 is numerically unreliable (18%-100% error vs LAPACK); use `single_svdlib::irlba` instead. See the module docs."
)]
pub fn svd_dim<T: SvdFloat, M: SparseMat<T>>(a: &M, dimensions: usize) -> Result<SvdRec<T>> {
    #[allow(deprecated)]
    svd_dim_seed(a, dimensions, 0)
}

/// SVD at the requested dimensionality with a fixed seed.
///
/// `random_seed == 0` draws a seed from the OS.
#[deprecated(
    since = "2.0.0",
    note = "LAS2 is numerically unreliable (18%-100% error vs LAPACK); use `single_svdlib::irlba` instead. See the module docs."
)]
pub fn svd_dim_seed<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    dimensions: usize,
    random_seed: u64,
) -> Result<SvdRec<T>> {
    #[allow(deprecated)]
    svd_las2(
        a,
        dimensions,
        0,
        &[
            T::from_f64_val(DEFAULT_END_INTERVAL[0]),
            T::from_f64_val(DEFAULT_END_INTERVAL[1]),
        ],
        T::from_f64_val(DEFAULT_KAPPA),
        random_seed,
    )
}

/// Compute a singular value decomposition with full control.
///
/// - `dimensions`: upper limit on singular triplets, `0` for `min(rows, cols)`
/// - `iterations`: upper limit on Lanczos steps, `0` for `min(rows, cols)`; clamped
///   into `[dimensions, min(rows, cols)]`
/// - `end_interval`: interval bracketing unwanted (near-zero) eigenvalues
/// - `kappa`: relative accuracy for accepting Ritz values, floored at `eps^(3/4)`
/// - `random_seed`: `0` draws from the OS
///
/// Singular values come back in descending order, with `u` as `m × d` and `vt` as
/// `d × n`.
#[deprecated(
    since = "2.0.0",
    note = "LAS2 is numerically unreliable (18%-100% error vs LAPACK); use `single_svdlib::irlba` instead. See the module docs."
)]
pub fn svd_las2<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    dimensions: usize,
    iterations: usize,
    end_interval: &[T; 2],
    kappa: T,
    random_seed: u64,
) -> Result<SvdRec<T>> {
    let random_seed = if random_seed > 0 {
        random_seed
    } else {
        rng().next_u64()
    };

    let min_dim = a.rows().min(a.cols());
    if min_dim < 2 {
        return Err(SvdLibError::invalid(format!(
            "svd_las2 needs both dimensions >= 2, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }

    let dimensions = match dimensions {
        n if n == 0 || n > min_dim => min_dim,
        n => n,
    };
    let iterations = match iterations {
        n if n == 0 || n > min_dim => min_dim,
        n if n < dimensions => dimensions,
        n => n,
    };
    if dimensions < 2 {
        return Err(SvdLibError::invalid(format!(
            "svd_las2: insufficient dimensions: {dimensions}"
        )));
    }

    // Working on the transpose keeps the Lanczos vectors over the smaller dimension.
    let transposed = (a.cols() as f64) >= (a.rows() as f64) * 1.2;
    let nrows = if transposed { a.cols() } else { a.rows() };
    let ncols = if transposed { a.rows() } else { a.cols() };

    let mut wrk = WorkSpace::new(nrows, ncols, transposed, iterations);
    let mut store = Store::new(ncols);
    let tuning = Tuning::for_matrix(a.nnz(), a.rows(), a.cols());

    let mut neig = 0;
    let steps = lanso(
        a,
        dimensions,
        iterations,
        end_interval,
        &mut wrk,
        &mut neig,
        &mut store,
        random_seed,
        &tuning,
    )?;

    let kappa = Float::max(Float::abs(kappa), T::eps34());
    let mut raw = ritvec(
        a, dimensions, kappa, &mut wrk, steps, neig, &mut store, &tuning,
    )?;

    if transposed {
        mem::swap(&mut raw.ut, &mut raw.vt);
    }

    let d = raw.d;
    // `ut` is stored d x m row-major; the public contract is u as m x d.
    let u = Array2::from_shape_vec((d, raw.ut.cols), raw.ut.value)?
        .t()
        .to_owned();
    let s = Array1::from_vec(raw.s);
    let vt = Array2::from_shape_vec((d, raw.vt.cols), raw.vt.value)?;

    let mut rec = SvdRec {
        d,
        u,
        s,
        vt,
        // LAS2 has no centering mode, so this is the plain Frobenius norm.
        total_squared_norm: T::from_f64_val(crate::matrix::total_squared_norm(a, None)),
        diagnostics: Diagnostics {
            algorithm: Algorithm::Las2,
            non_zero: a.nnz(),
            dimensions,
            significant_values: raw.nsig,
            transposed,
            random_seed,
            matvecs: wrk.matvecs.get(),
            detail: Detail::Lanczos {
                iterations,
                lanczos_steps: steps + 1,
                ritz_values_stabilized: neig,
                end_interval: *end_interval,
                kappa,
            },
        },
    };
    sort_descending(&mut rec);
    Ok(rec)
}

/// Reorder a decomposition so singular values descend, permuting `u` and `vt` with
/// them. LAS2 produces them in ascending Ritz-value order internally.
fn sort_descending<T: SvdFloat>(rec: &mut SvdRec<T>) {
    let d = rec.d;
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| {
        rec.s[j]
            .partial_cmp(&rec.s[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if order.iter().enumerate().all(|(i, &o)| i == o) {
        return;
    }
    let s = Array1::from_iter(order.iter().map(|&i| rec.s[i]));
    let u = rec.u.select(ndarray::Axis(1), &order);
    let vt = rec.vt.select(ndarray::Axis(0), &order);
    rec.s = s;
    rec.u = u;
    rec.vt = vt;
}

/// Sparsity-derived tolerances and iteration caps.
///
/// SVDLIBC used fixed values; very sparse operands need looser tolerances and more QL
/// sweeps to converge, so these scale with fill.
struct Tuning<T> {
    /// Tolerance floor used in place of raw machine epsilon.
    eps: T,
    /// Iteration cap for the tridiagonal QL kernels.
    ql_iterations: usize,
    /// Extra Lanczos steps granted per restart on very sparse inputs.
    extra_steps: usize,
    /// Multiplier applied to `kappa` when deciding significance.
    kappa_scale: T,
}

impl<T: SvdFloat> Tuning<T> {
    fn for_matrix(nnz: usize, rows: usize, cols: usize) -> Self {
        let denom = (rows as f64) * (cols as f64);
        let sparsity = if denom > 0.0 {
            1.0 - (nnz as f64 / denom)
        } else {
            0.0
        };
        let eps = T::eps();
        let (eps_scale, ql_iterations, extra_steps, kappa_scale) = if sparsity > 0.999 {
            (100.0, 500, 5, 10.0)
        } else if sparsity > 0.99 {
            (100.0, 300, 5, 10.0)
        } else if sparsity > 0.9 {
            (10.0, 200, 0, 1.0)
        } else {
            (1.0, MAX_QL_ITERATIONS, 0, 1.0)
        };
        Self {
            eps: eps * T::from_f64_val(eps_scale),
            ql_iterations,
            extra_steps,
            kappa_scale: T::from_f64_val(kappa_scale),
        }
    }
}

/// Retained Lanczos vectors.
///
/// `storq` holds the Lanczos basis (offset by [`MAXLL`]); `storp` holds the first
/// [`MAXLL`] vectors used for the initial reorthogonalization.
struct Store<T> {
    n: usize,
    vecs: Vec<Vec<T>>,
}

impl<T: SvdFloat> Store<T> {
    fn new(n: usize) -> Self {
        Self { n, vecs: vec![] }
    }
    fn storq(&mut self, idx: usize, v: &[T]) {
        while idx + MAXLL >= self.vecs.len() {
            self.vecs.push(vec![T::zero(); self.n]);
        }
        self.vecs[idx + MAXLL].copy_from_slice(v);
    }
    fn storp(&mut self, idx: usize, v: &[T]) {
        while idx >= self.vecs.len() {
            self.vecs.push(vec![T::zero(); self.n]);
        }
        self.vecs[idx].copy_from_slice(v);
    }
    fn retrq(&self, idx: usize) -> &[T] {
        &self.vecs[idx + MAXLL]
    }
    fn retrp(&self, idx: usize) -> &[T] {
        &self.vecs[idx]
    }
}

struct WorkSpace<T> {
    nrows: usize,
    ncols: usize,
    transposed: bool,
    w0: Vec<T>,
    w1: Vec<T>,
    w2: Vec<T>,
    w3: Vec<T>,
    w4: Vec<T>,
    w5: Vec<T>,
    /// Diagonal of the tridiagonal matrix T.
    alf: Vec<T>,
    /// Orthogonality estimate at step j.
    eta: Vec<T>,
    /// Orthogonality estimate at step j-1.
    oldeta: Vec<T>,
    /// Off-diagonal of T.
    bet: Vec<T>,
    /// Error bounds.
    bnd: Vec<T>,
    /// Ritz values.
    ritz: Vec<T>,
    temp: Vec<T>,
    /// Sparse products issued, for diagnostics. The Lanczos recurrence is serial, so a
    /// `Cell` suffices — the parallelism lives inside each product.
    matvecs: Cell<usize>,
    /// Set when a QL sweep hit its iteration cap and fell back to best estimates.
    ql_degraded: Cell<bool>,
}

impl<T: SvdFloat> WorkSpace<T> {
    fn new(nrows: usize, ncols: usize, transposed: bool, iterations: usize) -> Self {
        Self {
            nrows,
            ncols,
            transposed,
            w0: vec![T::zero(); ncols],
            w1: vec![T::zero(); ncols],
            w2: vec![T::zero(); ncols],
            w3: vec![T::zero(); ncols],
            w4: vec![T::zero(); ncols],
            w5: vec![T::zero(); ncols],
            alf: vec![T::zero(); iterations],
            eta: vec![T::zero(); iterations],
            oldeta: vec![T::zero(); iterations],
            bet: vec![T::zero(); 1 + iterations],
            ritz: vec![T::zero(); 1 + iterations],
            bnd: vec![<T as num_traits::Bounded>::max_value(); 1 + iterations],
            temp: vec![T::zero(); nrows],
            matvecs: Cell::new(0),
            ql_degraded: Cell::new(false),
        }
    }
}

/// Row-major dense matrix; rows are consecutive.
struct DMat<T> {
    cols: usize,
    value: Vec<T>,
}

struct RawRec<T> {
    d: usize,
    nsig: usize,
    ut: DMat<T>,
    s: Vec<T>,
    vt: DMat<T>,
}

#[inline]
fn close<T: SvdFloat>(a: T, b: T) -> bool {
    T::close(a, b)
}

/// Sort `keys` ascending, applying the same permutation to `vals`.
///
/// Replaces SVDLIBC's insertion sort, which was quadratic in the Lanczos step count.
/// A stable sort keeps the tie ordering the original relied on.
fn sort_pair<T: SvdFloat>(n: usize, keys: &mut [T], vals: &mut [T]) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        keys[i]
            .partial_cmp(&keys[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sk: Vec<T> = order.iter().map(|&i| keys[i]).collect();
    let sv: Vec<T> = order.iter().map(|&i| vals[i]).collect();
    keys[..n].copy_from_slice(&sk);
    vals[..n].copy_from_slice(&sv);
}

/// `y = Aᵀ(Ax)`, using `temp` as the intermediate.
fn svd_opb<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    x: &[T],
    y: &mut [T],
    temp: &mut [T],
    transposed: bool,
    matvecs: &Cell<usize>,
) {
    a.mul_vec(x, temp, transposed);
    a.mul_vec(temp, y, !transposed);
    matvecs.set(matvecs.get() + 2);
}

fn daxpy<T: SvdFloat>(da: T, x: &[T], y: &mut [T]) {
    if x.len() < 1024 {
        for (yv, &xv) in y.iter_mut().zip(x.iter()) {
            *yv += da * xv;
        }
    } else {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(yv, &xv)| *yv += da * xv);
    }
}

fn ddot<T: SvdFloat>(x: &[T], y: &[T]) -> T {
    if x.len() < 1024 {
        x.iter().zip(y).map(|(&a, &b)| a * b).sum()
    } else {
        x.par_iter().zip(y.par_iter()).map(|(&a, &b)| a * b).sum()
    }
}

fn norm<T: SvdFloat>(x: &[T]) -> T {
    ddot(x, x).sqrt()
}

fn datx<T: SvdFloat>(d: T, x: &[T], y: &mut [T]) {
    for (yv, &xv) in y.iter_mut().zip(x.iter()) {
        *yv = d * xv;
    }
}

fn dscal<T: SvdFloat>(d: T, x: &mut [T]) {
    if x.len() < 1024 {
        for v in x.iter_mut() {
            *v *= d;
        }
    } else {
        x.par_iter_mut().for_each(|v| *v *= d);
    }
}

/// Copy `n` elements of `x` into `y` starting at `offset`, reversing their order.
fn dcopy_rev<T: SvdFloat>(n: usize, offset: usize, x: &[T], y: &mut [T]) {
    if n > 0 {
        let start = n - 1;
        for i in 0..n {
            y[offset + start - i] = x[offset + i];
        }
    }
}

/// Index of the element with the largest magnitude.
fn idamax<T: SvdFloat>(n: usize, x: &[T]) -> usize {
    debug_assert!(n > 0);
    let mut imax = 0;
    for i in 1..n {
        if Float::abs(x[i]) > Float::abs(x[imax]) {
            imax = i;
        }
    }
    imax
}

/// `|a|` if `b >= 0`, else `-|a|`.
fn fsign<T: SvdFloat>(a: T, b: T) -> T {
    if (a >= T::zero()) == (b >= T::zero()) {
        a
    } else {
        -a
    }
}

/// `sqrt(a² + b²)` without intermediate overflow.
fn pythag<T: SvdFloat>(a: T, b: T) -> T {
    let n = Float::max(Float::abs(a), Float::abs(b));
    if n <= T::zero() {
        return T::zero();
    }
    let four = T::from_f64_val(4.0);
    let two = T::from_f64_val(2.0);
    let mut p = n;
    let mut r = Float::powi(Float::min(Float::abs(a), Float::abs(b)) / p, 2);
    let mut t = four + r;
    // The convergence test is `t == 4`, which a NaN never satisfies — an unbounded loop
    // here would hang the process. The iteration converges quadratically, so a handful
    // of steps is ample and the cap only ever fires on a poisoned input.
    let mut guard = 0usize;
    while !close(t, four) && guard < 64 {
        guard += 1;
        let s = r / t;
        let u = T::one() + two * s;
        p *= u;
        r = Float::powi(s / u, 2);
        t = four + r;
    }
    p
}

/// Implicit QL for the eigenvalues of a symmetric tridiagonal matrix, tracking the
/// first components of the eigenvectors in `bnd`.
///
/// On hitting the iteration cap this widens the affected error bounds and continues
/// rather than failing, matching 1.x behaviour, and reports it through `degraded`.
fn imtqlb<T: SvdFloat>(
    n: usize,
    d: &mut [T],
    e: &mut [T],
    bnd: &mut [T],
    max_iter: usize,
    degraded: &Cell<bool>,
) {
    if n == 1 {
        return;
    }
    let size_factor = T::from_f64_val((n as f64).sqrt());
    bnd[0] = T::one();
    let last = n - 1;
    for i in 1..=last {
        bnd[i] = T::zero();
        e[i - 1] = e[i];
    }
    e[last] = T::zero();

    let mut i = 0;
    for l in 0..=last {
        let mut iteration = 0;

        while iteration <= max_iter {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = Float::abs(d[m]) + Float::abs(d[m + 1]);
                let tol =
                    T::eps() * T::from_f64_val(100.0) * Float::max(test, T::one()) * size_factor;
                if Float::abs(e[m]) <= tol {
                    break;
                }
                m += 1;
            }

            // The shift origin and the tracked eigenvector component must be re-read
            // from the *current* d and bnd on every sweep — EISPACK IMTQL1 assigns
            // `p = d(l)` at label 120, inside the iteration loop. 1.x hoisted both out
            // of the loop, so after the first sweep every subsequent eigenvalue was
            // computed from a stale shift. That is what produced the "imtqlb had some
            // convergence issues" warnings and the garbage Ritz values behind them.
            let mut p = d[l];
            let mut f = bnd[l];

            if m == l {
                // Insert this eigenvalue into the already-ordered prefix.
                let mut exchange = true;
                if l > 0 {
                    i = l;
                    while i >= 1 && exchange {
                        if p < d[i - 1] {
                            d[i] = d[i - 1];
                            bnd[i] = bnd[i - 1];
                            i -= 1;
                        } else {
                            exchange = false;
                        }
                    }
                }
                if exchange {
                    i = 0;
                }
                d[i] = p;
                bnd[i] = f;
                break;
            }

            if iteration == max_iter {
                degraded.set(true);
                for b in bnd.iter_mut().take(m + 1).skip(l) {
                    *b = Float::max(*b, T::from_f64_val(0.1));
                }
                e[l] = T::zero();
                break;
            }
            iteration += 1;

            let two = T::from_f64_val(2.0);
            let mut g = (d[l + 1] - p) / (two * e[l]);
            let mut r = pythag(g, T::one());
            g = d[m] - p + e[l] / (g + fsign(r, g));
            let mut s = T::one();
            let mut c = T::one();
            p = T::zero();

            debug_assert!(m > 0);
            i = m - 1;
            let mut underflow = false;
            while !underflow && i >= l {
                f = s * e[i];
                let b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;

                if r < T::eps() * T::from_f64_val(1000.0) * (Float::abs(f) + Float::abs(g)) {
                    underflow = true;
                    break;
                }
                if Float::abs(r) < T::eps() * T::from_f64_val(100.0) {
                    r = T::eps() * T::from_f64_val(100.0) * fsign(T::one(), r);
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + two * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                f = bnd[i + 1];
                bnd[i + 1] = s * bnd[i] + c * f;
                bnd[i] = c * bnd[i] - s * f;
                if i == 0 {
                    break;
                }
                i -= 1;
            }
            if underflow {
                d[i + 1] -= p;
            } else {
                d[l] -= p;
                e[l] = g;
            }
            e[m] = T::zero();
        }
    }
}

/// Implicit QL for eigenvalues *and* eigenvectors of a symmetric tridiagonal matrix.
fn imtql2<T: SvdFloat>(
    nm: usize,
    n: usize,
    d: &mut [T],
    e: &mut [T],
    z: &mut [T],
    max_iter: usize,
) -> Result<()> {
    if n == 1 {
        return Ok(());
    }
    let two = T::from_f64_val(2.0);
    let last = n - 1;
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[last] = T::zero();

    let nnm = n * nm;
    for l in 0..n {
        let mut iteration = 0;
        while iteration <= max_iter {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = Float::abs(d[m]) + Float::abs(d[m + 1]);
                if close(test, test + Float::abs(e[m])) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iteration == max_iter {
                return Err(SvdLibError::NoConvergence {
                    stage: "imtql2",
                    iterations: max_iter,
                });
            }
            iteration += 1;

            let mut g = (d[l + 1] - d[l]) / (two * e[l]);
            let mut r = pythag(g, T::one());
            g = d[m] - d[l] + e[l] / (g + fsign(r, g));
            let mut s = T::one();
            let mut c = T::one();
            let mut p = T::zero();

            debug_assert!(m > 0);
            let mut i = m - 1;
            let mut underflow = false;
            while !underflow && i >= l {
                let mut f = s * e[i];
                let b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;
                if close(r, T::zero()) {
                    underflow = true;
                } else {
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + two * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    for k in (0..nnm).step_by(n) {
                        let index = k + i;
                        f = z[index + 1];
                        z[index + 1] = s * z[index] + c * f;
                        z[index] = c * z[index] - s * f;
                    }
                    if i == 0 {
                        break;
                    }
                    i -= 1;
                }
            }
            if underflow {
                d[i + 1] -= p;
            } else {
                d[l] -= p;
                e[l] = g;
            }
            e[m] = T::zero();
        }
    }

    // Order eigenvalues ascending, carrying the eigenvectors along.
    for l in 1..n {
        let i = l - 1;
        let mut k = i;
        let mut p = d[i];
        for (j, item) in d.iter().enumerate().take(n).skip(l) {
            if *item < p {
                k = j;
                p = *item;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in (0..nnm).step_by(n) {
                z.swap(j + i, j + k);
            }
        }
    }
    Ok(())
}

/// Produce a starting vector in the range of `AᵀA`, orthogonal to the basis so far.
fn startv<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    wrk: &mut WorkSpace<T>,
    step: usize,
    store: &Store<T>,
    random_seed: u64,
) -> Result<T> {
    let mut rnm2 = ddot(&wrk.w0, &wrk.w0);
    for id in 0..3 {
        if id > 0 || step > 0 || close(rnm2, T::zero()) {
            let mut bytes = [0u8; 32];
            for (i, b) in random_seed.to_le_bytes().iter().enumerate() {
                bytes[i] = *b;
            }
            let mut seeded = StdRng::from_seed(bytes);
            for val in wrk.w0.iter_mut() {
                *val = T::from_f64_val(seeded.random_range(-1.0..1.0));
            }
        }
        wrk.w3.copy_from_slice(&wrk.w0);
        svd_opb(
            a,
            &wrk.w3,
            &mut wrk.w0,
            &mut wrk.temp,
            wrk.transposed,
            &wrk.matvecs,
        );
        wrk.w3.copy_from_slice(&wrk.w0);
        rnm2 = ddot(&wrk.w3, &wrk.w3);
        if rnm2 > T::zero() {
            break;
        }
    }

    if rnm2 <= T::zero() {
        return Err(SvdLibError::failed(
            "startv",
            format!("could not find a starting vector in range (rnm2 = {rnm2:?})"),
        ));
    }

    if step > 0 {
        for i in 0..step {
            let v = store.retrq(i);
            daxpy(-ddot(&wrk.w3, v), v, &mut wrk.w0);
        }
        // Keep q[step] orthogonal to q[step-1].
        let t = -ddot(&wrk.w4, &wrk.w0);
        let w2 = std::mem::take(&mut wrk.w2);
        daxpy(t, &w2, &mut wrk.w0);
        wrk.w2 = w2;
        wrk.w3.copy_from_slice(&wrk.w0);
        rnm2 = match ddot(&wrk.w3, &wrk.w3) {
            dot if dot <= T::eps() * rnm2 => T::zero(),
            dot => dot,
        };
    }
    Ok(rnm2.sqrt())
}

/// The first Lanczos step; returns `(rnm, tol)`.
fn stpone<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    wrk: &mut WorkSpace<T>,
    store: &Store<T>,
    random_seed: u64,
) -> Result<(T, T)> {
    let mut rnm = startv(a, wrk, 0, store, random_seed)?;
    if close(rnm, T::zero()) {
        return Err(SvdLibError::failed(
            "stpone",
            "starting vector has zero norm",
        ));
    }

    datx(Float::recip(rnm), &wrk.w0, &mut wrk.w1);
    dscal(Float::recip(rnm), &mut wrk.w3);

    svd_opb(
        a,
        &wrk.w3,
        &mut wrk.w0,
        &mut wrk.temp,
        wrk.transposed,
        &wrk.matvecs,
    );
    wrk.alf[0] = ddot(&wrk.w0, &wrk.w3);
    let alf0 = wrk.alf[0];
    let w1 = std::mem::take(&mut wrk.w1);
    daxpy(-alf0, &w1, &mut wrk.w0);
    let t = ddot(&wrk.w0, &wrk.w3);
    wrk.alf[0] += t;
    daxpy(-t, &w1, &mut wrk.w0);
    wrk.w1 = w1;
    wrk.w4.copy_from_slice(&wrk.w0);
    rnm = norm(&wrk.w4);
    let anorm = rnm + Float::abs(wrk.alf[0]);
    Ok((rnm, T::eps().sqrt() * anorm))
}

#[allow(clippy::too_many_arguments)]
fn lanczos_step<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    wrk: &mut WorkSpace<T>,
    first: usize,
    last: usize,
    ll: &mut usize,
    enough: &mut bool,
    rnm: &mut T,
    tol: &mut T,
    store: &mut Store<T>,
) -> Result<usize> {
    let eps1 = T::eps() * T::from_f64_val(wrk.ncols as f64).sqrt();
    let mut j = first;
    let four = T::from_f64_val(4.0);

    while j < last {
        mem::swap(&mut wrk.w1, &mut wrk.w2);
        mem::swap(&mut wrk.w3, &mut wrk.w4);

        store.storq(j - 1, &wrk.w2);
        if j - 1 < MAXLL {
            store.storp(j - 1, &wrk.w4);
        }
        wrk.bet[j] = *rnm;

        // Restart if an invariant subspace turned up.
        if close(*rnm, T::zero()) {
            *rnm = startv(a, wrk, j, store, 0)?;
            if close(*rnm, T::zero()) {
                *enough = true;
            }
        }
        if *enough {
            mem::swap(&mut wrk.w1, &mut wrk.w2);
            break;
        }

        datx(Float::recip(*rnm), &wrk.w0, &mut wrk.w1);
        dscal(Float::recip(*rnm), &mut wrk.w3);
        svd_opb(
            a,
            &wrk.w3,
            &mut wrk.w0,
            &mut wrk.temp,
            wrk.transposed,
            &wrk.matvecs,
        );
        let rnm_v = *rnm;
        let w2 = std::mem::take(&mut wrk.w2);
        daxpy(-rnm_v, &w2, &mut wrk.w0);
        wrk.w2 = w2;
        wrk.alf[j] = ddot(&wrk.w0, &wrk.w3);
        let alfj = wrk.alf[j];
        let w1 = std::mem::take(&mut wrk.w1);
        daxpy(-alfj, &w1, &mut wrk.w0);
        wrk.w1 = w1;

        // Reorthogonalize against the first few Lanczos vectors.
        if j <= MAXLL && Float::abs(wrk.alf[j - 1]) > four * Float::abs(wrk.alf[j]) {
            *ll = j;
        }
        for i in 0..(j - 1).min(*ll) {
            let t = ddot(store.retrp(i), &wrk.w0);
            daxpy(-t, store.retrq(i), &mut wrk.w0);
            wrk.eta[i] = eps1;
            wrk.oldeta[i] = eps1;
        }

        // Extended local reorthogonalization.
        let t = ddot(&wrk.w0, &wrk.w4);
        let w2 = std::mem::take(&mut wrk.w2);
        daxpy(-t, &w2, &mut wrk.w0);
        wrk.w2 = w2;
        if wrk.bet[j] > T::zero() {
            wrk.bet[j] += t;
        }
        let t = ddot(&wrk.w0, &wrk.w3);
        let w1 = std::mem::take(&mut wrk.w1);
        daxpy(-t, &w1, &mut wrk.w0);
        wrk.w1 = w1;
        wrk.alf[j] += t;
        wrk.w4.copy_from_slice(&wrk.w0);
        *rnm = norm(&wrk.w4);
        let anorm = wrk.bet[j] + Float::abs(wrk.alf[j]) + *rnm;
        *tol = T::eps().sqrt() * anorm;

        ortbnd(wrk, j, *rnm, eps1);
        purge(wrk.ncols, *ll, wrk, j, rnm, *tol, store);
        if *rnm <= *tol {
            *rnm = T::zero();
        }
        j += 1;
    }
    Ok(j)
}

/// Restore orthogonality once the estimates say it has been lost.
fn purge<T: SvdFloat>(
    n: usize,
    ll: usize,
    wrk: &mut WorkSpace<T>,
    step: usize,
    rnm: &mut T,
    tol: T,
    store: &Store<T>,
) {
    if step < ll + 2 {
        return;
    }
    let reps = T::eps().sqrt();
    let eps1 = T::eps() * T::from_f64_val(n as f64).sqrt();

    let k = idamax(step - (ll + 1), &wrk.eta) + ll;
    if Float::abs(wrk.eta[k]) > reps {
        let reps1 = eps1 / reps;
        let mut iteration = 0;
        let mut flag = true;
        while iteration < 2 && flag {
            if *rnm > tol {
                let mut tq = T::zero();
                let mut tr = T::zero();
                for i in ll..step {
                    let v = store.retrq(i);
                    let t = ddot(v, &wrk.w3);
                    tq += Float::abs(t);
                    daxpy(-t, v, &mut wrk.w1);
                    let t = ddot(v, &wrk.w4);
                    tr += Float::abs(t);
                    daxpy(-t, v, &mut wrk.w0);
                }
                wrk.w3.copy_from_slice(&wrk.w1);
                let t = ddot(&wrk.w0, &wrk.w3);
                tr += Float::abs(t);
                let w1 = std::mem::take(&mut wrk.w1);
                daxpy(-t, &w1, &mut wrk.w0);
                wrk.w1 = w1;
                wrk.w4.copy_from_slice(&wrk.w0);
                *rnm = norm(&wrk.w4);
                if tq <= reps1 && tr <= *rnm * reps1 {
                    flag = false;
                }
            }
            iteration += 1;
        }
        for i in ll..=step {
            wrk.eta[i] = eps1;
            wrk.oldeta[i] = eps1;
        }
    }
}

/// Update the running estimates of basis orthogonality.
fn ortbnd<T: SvdFloat>(wrk: &mut WorkSpace<T>, step: usize, rnm: T, eps1: T) {
    if step < 1 {
        return;
    }
    if !close(rnm, T::zero()) && step > 1 {
        wrk.oldeta[0] = (wrk.bet[1] * wrk.eta[1] + (wrk.alf[0] - wrk.alf[step]) * wrk.eta[0]
            - wrk.bet[step] * wrk.oldeta[0])
            / rnm
            + eps1;
        if step > 2 {
            for i in 1..=step - 2 {
                wrk.oldeta[i] = (wrk.bet[i + 1] * wrk.eta[i + 1]
                    + (wrk.alf[i] - wrk.alf[step]) * wrk.eta[i]
                    + wrk.bet[i] * wrk.eta[i - 1]
                    - wrk.bet[step] * wrk.oldeta[i])
                    / rnm
                    + eps1;
            }
        }
    }
    wrk.oldeta[step - 1] = eps1;
    mem::swap(&mut wrk.oldeta, &mut wrk.eta);
    wrk.eta[step] = eps1;
}

/// Tighten error bounds and count how many Ritz values have stabilized.
fn error_bound<T: SvdFloat>(
    enough: &mut bool,
    endl: T,
    endr: T,
    ritz: &mut [T],
    bnd: &mut [T],
    step: usize,
    tol: T,
) -> usize {
    debug_assert!(step > 0);
    let mid = idamax(step + 1, bnd);
    let sixteen = T::from_f64_val(16.0);

    // Fold bounds together for Ritz values that are nearly coincident.
    let mut i = ((step + 1) + (step - 1)) / 2;
    while i > mid + 1 {
        if Float::abs(ritz[i - 1] - ritz[i]) < T::eps34() * Float::abs(ritz[i])
            && bnd[i] > tol
            && bnd[i - 1] > tol
        {
            bnd[i - 1] = (Float::powi(bnd[i], 2) + Float::powi(bnd[i - 1], 2)).sqrt();
            bnd[i] = T::zero();
        }
        i -= 1;
    }
    let mut i = ((step + 1) - (step - 1)) / 2;
    while i + 1 < mid {
        if Float::abs(ritz[i + 1] - ritz[i]) < T::eps34() * Float::abs(ritz[i])
            && bnd[i] > tol
            && bnd[i + 1] > tol
        {
            bnd[i + 1] = (Float::powi(bnd[i], 2) + Float::powi(bnd[i + 1], 2)).sqrt();
            bnd[i] = T::zero();
        }
        i += 1;
    }

    let mut neig = 0;
    let mut gapl = ritz[step] - ritz[0];
    for i in 0..=step {
        let mut gap = gapl;
        if i < step {
            gapl = ritz[i + 1] - ritz[i];
        }
        gap = Float::min(gap, gapl);
        if gap > bnd[i] {
            bnd[i] *= bnd[i] / gap;
        }
        if bnd[i] <= sixteen * T::eps() * Float::abs(ritz[i]) {
            neig += 1;
            if !*enough {
                *enough = endl < ritz[i] && ritz[i] < endr;
            }
        }
    }
    neig
}

/// Recover singular triplets from the converged Lanczos basis.
#[allow(clippy::too_many_arguments)]
fn ritvec<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    dimensions: usize,
    kappa: T,
    wrk: &mut WorkSpace<T>,
    steps: usize,
    neig: usize,
    store: &mut Store<T>,
    tuning: &Tuning<T>,
) -> Result<RawRec<T>> {
    let js = steps + 1;
    let jsq = js * js;
    let adaptive_eps = tuning.eps;

    let mut s = vec![T::zero(); jsq];
    for i in (0..jsq).step_by(js + 1) {
        s[i] = T::one();
    }

    let mut eigenvalues = vec![T::zero(); wrk.ncols.max(js)];
    dcopy_rev(js, 0, &wrk.alf, &mut eigenvalues);
    dcopy_rev(steps, 1, &wrk.bet, &mut wrk.w5);

    // On return `eigenvalues` is ascending and `s` holds the matching eigenvectors.
    imtql2(
        js,
        js,
        &mut eigenvalues,
        &mut wrk.w5,
        &mut s,
        tuning.ql_iterations,
    )?;

    let max_eigenvalue = eigenvalues
        .iter()
        .take(js)
        .fold(T::zero(), |mx, &v| Float::max(mx, Float::abs(v)));
    let adaptive_kappa = kappa * tuning.kappa_scale;

    let store_vectors: Vec<&[T]> = (0..js).map(|i| store.retrq(i)).collect();

    let significant: Vec<usize> = (0..js)
        .filter(|&k| {
            let bound =
                adaptive_kappa * Float::max(Float::abs(wrk.ritz[k]), max_eigenvalue * adaptive_eps);
            wrk.bnd[k] <= bound && k + 1 > js - neig
        })
        .collect();
    let nsig = significant.len();

    let d = dimensions.min(nsig);
    if d == 0 {
        return Err(SvdLibError::failed(
            "ritvec",
            "no singular values met the significance threshold; \
             try more iterations or a larger kappa",
        ));
    }

    // `imtql2` and `lanso` both order Ritz values ascending, so the *largest* `d` are
    // the tail of `significant`. 1.x took the leading `d` instead, which silently
    // returned the smallest converged triplets whenever more converged than were
    // requested — on `diag(40..1)` that reported the 10th-largest singular value as
    // the largest. Keep the tail, then restore ascending order within it.
    let keep: Vec<usize> = significant[nsig - d..].to_vec();

    let mut vt_vectors: Vec<(usize, Vec<T>)> = keep
        .into_par_iter()
        .map(|k| {
            let mut vec = vec![T::zero(); wrk.ncols];
            for (i, sv) in store_vectors.iter().enumerate().take(js) {
                let coeff = s[k * js + i];
                if Float::abs(coeff) > adaptive_eps {
                    for (dst, &src) in vec.iter_mut().zip(sv.iter()).take(wrk.ncols) {
                        *dst += coeff * src;
                    }
                }
            }
            (k, vec)
        })
        .collect();
    vt_vectors.sort_by_key(|(k, _)| *k);

    let mut vt = DMat {
        cols: wrk.ncols,
        value: vec![T::zero(); wrk.ncols * d],
    };
    for (i, (_, vec)) in vt_vectors.into_iter().enumerate() {
        let off = i * vt.cols;
        vt.value[off..off + vt.cols].copy_from_slice(&vec);
    }

    let mut ut = DMat {
        cols: wrk.nrows,
        value: vec![T::zero(); wrk.nrows * d],
    };
    let mut sv = vec![T::zero(); d];

    // Each triplet needs A·v and Aᵀ(A·v); the products are serial because they share
    // `wrk.temp`, but each one is internally parallel.
    for i in 0..d {
        let off = i * vt.cols;
        let v = &vt.value[off..off + vt.cols];
        let mut abv = vec![T::zero(); vt.cols];
        let mut av = vec![T::zero(); wrk.nrows];

        svd_opb(a, v, &mut abv, &mut wrk.temp, wrk.transposed, &wrk.matvecs);
        a.mul_vec(v, &mut av, wrk.transposed);
        wrk.matvecs.set(wrk.matvecs.get() + 1);

        let t = ddot(v, &abv);
        let sval = Float::max(t, T::zero()).sqrt();
        sv[i] = sval;

        let scale = T::one() / Float::max(sval, adaptive_eps);
        dscal(scale, &mut av);
        let uoff = i * ut.cols;
        ut.value[uoff..uoff + ut.cols].copy_from_slice(&av);
    }

    Ok(RawRec {
        d,
        nsig,
        ut,
        s: sv,
        vt,
    })
}

/// The outer restart loop: run Lanczos steps until enough Ritz values stabilize.
#[allow(clippy::too_many_arguments)]
fn lanso<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    dim: usize,
    iterations: usize,
    end_interval: &[T; 2],
    wrk: &mut WorkSpace<T>,
    neig: &mut usize,
    store: &mut Store<T>,
    random_seed: u64,
    tuning: &Tuning<T>,
) -> Result<usize> {
    let adaptive_eps = tuning.eps;
    let (endl, endr) = (end_interval[0], end_interval[1]);

    let (mut rnm, mut tol) = stpone(a, wrk, store, random_seed)?;

    let eps1 = adaptive_eps * T::from_f64_val(wrk.ncols as f64).sqrt();
    wrk.eta[0] = eps1;
    wrk.oldeta[0] = eps1;
    let mut ll = 0;
    let mut first = 1;
    let mut last = iterations.min(dim.max(8) + dim);
    let mut enough = false;
    let mut j = 0;
    let mut intro = 0;

    while !enough {
        if rnm <= tol {
            rnm = T::zero();
        }

        let steps = lanczos_step(
            a,
            wrk,
            first,
            last,
            &mut ll,
            &mut enough,
            &mut rnm,
            &mut tol,
            store,
        )?;
        j = if enough { steps - 1 } else { last - 1 };

        first = j + 1;
        wrk.bet[first] = rnm;

        // Analyze T one unreduced block at a time.
        let mut l = 0;
        for _ in 0..j {
            if l > j {
                break;
            }
            let mut i = l;
            while i <= j {
                if Float::abs(wrk.bet[i + 1]) <= adaptive_eps {
                    break;
                }
                i += 1;
            }
            i = i.min(j);

            let sz = i - l;
            dcopy_rev(sz + 1, l, &wrk.alf, &mut wrk.ritz);
            dcopy_rev(sz, l + 1, &wrk.bet, &mut wrk.w5);

            imtqlb(
                sz + 1,
                &mut wrk.ritz[l..],
                &mut wrk.w5[l..],
                &mut wrk.bnd[l..],
                tuning.ql_iterations,
                &wrk.ql_degraded,
            );

            for m in l..=i {
                wrk.bnd[m] = rnm * Float::abs(wrk.bnd[m]);
            }
            l = i + 1;
        }

        sort_pair(j + 1, &mut wrk.ritz, &mut wrk.bnd);
        *neig = error_bound(&mut enough, endl, endr, &mut wrk.ritz, &mut wrk.bnd, j, tol);

        if *neig < dim {
            if *neig == 0 {
                last = first + 9;
                intro = first;
            } else {
                last =
                    first + 3.max(1 + ((j - intro) * (dim - *neig)) / *neig) + tuning.extra_steps;
            }
            last = last.min(iterations);
        } else {
            enough = true;
        }
        enough = enough || first >= iterations;
    }
    store.storq(j, &wrk.w1);
    Ok(j)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::matrix::SvdMat;
    use crate::testing::{dense_of, gen_lowrank, gen_sparse, reference_singular_values};
    use sprs::TriMatI;

    /// `diag(n, n-1, ..., 1)` — singular values are known exactly, so this is the
    /// least forgiving accuracy probe available.
    fn diagonal(n: usize) -> SvdMat<f64> {
        let mut t = TriMatI::<f64, u32>::new((n, n));
        for i in 0..n {
            t.add_triplet(i, i, (n - i) as f64);
        }
        t.to_csr::<u64>()
    }

    // ---------------------------------------------------------------------------
    // Structural properties. These hold today and guard the port.
    // ---------------------------------------------------------------------------

    #[test]
    fn singular_values_descend() {
        let a = gen_sparse(200, 120, 0.05, 3);
        let svd = svd_dim_seed(&a, 20, 42).unwrap();
        for w in svd.s.to_vec().windows(2) {
            assert!(w[0] >= w[1], "not descending: {:?}", svd.s);
        }
    }

    /// `u` must be `m x d` and `vt` `d x n` for every input shape, including the
    /// internally-transposed case. 1.x returned `u` as `d x m` from this path while
    /// the randomized path returned `m x d`, so `recompose` only worked when square.
    #[test]
    fn orientation_is_consistent_for_wide_and_tall() {
        for (r, c) in [(200usize, 60usize), (60, 200)] {
            let a = gen_sparse(r, c, 0.1, 11);
            let svd = svd_dim_seed(&a, 10, 42).unwrap();
            assert_eq!(svd.u.nrows(), r, "u rows for {r}x{c}");
            assert_eq!(svd.u.ncols(), svd.d, "u cols for {r}x{c}");
            assert_eq!(svd.vt.nrows(), svd.d, "vt rows for {r}x{c}");
            assert_eq!(svd.vt.ncols(), c, "vt cols for {r}x{c}");
        }
    }

    #[test]
    fn csc_input_matches_csr() {
        let a = gen_sparse(150, 90, 0.08, 5);
        let csc = a.to_other_storage();
        let from_csr = svd_dim_seed(&a, 12, 42).unwrap();
        let from_csc = svd_dim_seed(&csc, 12, 42).unwrap();
        for (x, y) in from_csr.s.iter().zip(from_csc.s.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-10);
        }
    }

    #[test]
    fn rejects_degenerate_shapes() {
        let a = gen_sparse(1, 10, 1.0, 1);
        assert!(matches!(
            svd_dim_seed(&a, 0, 42),
            Err(SvdLibError::InvalidArgument(_))
        ));
    }

    #[test]
    fn diagnostics_count_matvecs() {
        let a = gen_sparse(100, 60, 0.1, 13);
        let svd = svd_dim_seed(&a, 8, 42).unwrap();
        assert!(svd.diagnostics.matvecs > 0);
        assert_eq!(svd.diagnostics.algorithm, Algorithm::Las2);
    }

    /// The `imtqlb` shift-origin fix, pinned directly.
    ///
    /// `imtqlb` (eigenvalues only) and `imtql2` (eigenvalues and vectors) run the same
    /// implicit-QL recurrence on the same tridiagonal matrix, so their eigenvalues must
    /// agree. Before the fix they diverged wildly — on `diag(40..1)` `imtqlb` returned
    /// `39.90, 7.11, 0.019, ...` against `imtql2`'s correct `39.89, 38.96, 37.86, ...`.
    #[test]
    fn imtqlb_agrees_with_imtql2_on_the_same_tridiagonal() {
        // A tridiagonal with well-separated eigenvalues.
        let n = 24;
        let d0: Vec<f64> = (0..n).map(|i| 2.0 + i as f64).collect();
        let e0: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * (i as f64)).collect();

        let mut d_b = d0.clone();
        let mut e_b = e0.clone();
        let mut bnd = vec![0.0f64; n];
        let degraded = Cell::new(false);
        imtqlb(
            n,
            &mut d_b,
            &mut e_b,
            &mut bnd,
            MAX_QL_ITERATIONS,
            &degraded,
        );
        assert!(!degraded.get(), "imtqlb reported degraded convergence");

        let mut d_2 = d0.clone();
        let mut e_2 = e0.clone();
        let mut z = vec![0.0f64; n * n];
        for i in (0..n * n).step_by(n + 1) {
            z[i] = 1.0;
        }
        imtql2(n, n, &mut d_2, &mut e_2, &mut z, MAX_QL_ITERATIONS).unwrap();

        // Not bit-identical: `imtqlb` deflates on a size-scaled tolerance while
        // `imtql2` uses the tighter `test + |e| == test`, so it stops marginally
        // earlier. A few ulps of spread is expected; the pre-fix divergence was
        // orders of magnitude.
        for i in 0..n {
            approx::assert_relative_eq!(d_b[i], d_2[i], max_relative = 1e-5);
        }
    }

    // ---------------------------------------------------------------------------
    // Accuracy against a dense LAPACK reference.
    //
    // These are `#[ignore]`d because LAS2 does not currently pass them — the failure
    // is inherited from published 1.0.9, not introduced by the sprs port (verified by
    // running 1.0.9 on identical fixtures). Two defects are identified so far:
    //
    //   1. `imtqlb` hoisted the shift origin out of its iteration loop — FIXED.
    //   2. `ritvec` reads `s[k*js + i]` (row `k`) while `imtql2` stores eigenvectors
    //      as columns; transposing roughly halves the error but does not close it,
    //      so at least one further defect remains.
    //
    // Un-ignore these once LAS2 is repaired, or delete them with the module if LAS2
    // is retired in favour of `crate::irlba`.
    // ---------------------------------------------------------------------------

    fn assert_matches_lapack(name: &str, a: &SvdMat<f64>, dims: usize, tol: f64) {
        let want = reference_singular_values(&dense_of(a));
        let svd = svd_dim_seed(a, dims, 42).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (i, &g) in svd.s.iter().enumerate() {
            let rel = (g - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(
                rel < tol,
                "{name}: singular value {i}: got {g:.9e}, LAPACK {:.9e} (rel {rel:.3e})",
                want[i]
            );
        }
    }

    #[test]
    #[ignore = "LAS2 accuracy defect inherited from 1.0.9; see module comment"]
    fn exact_on_diagonal_matrix() {
        assert_matches_lapack("diagonal_40", &diagonal(40), 10, 1e-8);
    }

    #[test]
    #[ignore = "LAS2 accuracy defect inherited from 1.0.9; see module comment"]
    fn agrees_with_dense_reference_lowrank() {
        assert_matches_lapack("lowrank_80x50_r8", &gen_lowrank(80, 50, 8, 21), 8, 1e-6);
    }

    #[test]
    #[ignore = "LAS2 accuracy defect inherited from 1.0.9; see module comment"]
    fn agrees_with_dense_reference_sparse() {
        assert_matches_lapack("sparse_500x40", &gen_sparse(500, 40, 0.10, 7), 10, 1e-6);
    }

    #[test]
    #[ignore = "LAS2 accuracy defect inherited from 1.0.9; see module comment"]
    fn recompose_round_trips() {
        let a = gen_lowrank(40, 25, 25, 99);
        let dense = dense_of(&a);
        let svd = svd_dim_seed(&a, 25, 42).unwrap();
        let rec = svd.recompose();
        let err: f64 = (&rec - &dense).iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale: f64 = dense.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err / scale < 1e-8,
            "relative reconstruction error {}",
            err / scale
        );
    }

    /// Scope report: prints LAS2's error against LAPACK across matrix classes.
    /// Not an assertion — a diagnostic for whoever picks up the repair.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn report_accuracy_vs_lapack() {
        let cases: Vec<(&str, SvdMat<f64>, usize)> = vec![
            ("diagonal_40", diagonal(40), 10),
            ("diagonal_40_full", diagonal(40), 40),
            ("lowrank_80x50_r8", gen_lowrank(80, 50, 8, 21), 8),
            ("lowrank_200x80_r10", gen_lowrank(200, 80, 10, 555), 15),
            ("sparse_500x40_d10", gen_sparse(500, 40, 0.10, 7), 10),
            ("sparse_200x120_d05", gen_sparse(200, 120, 0.05, 3), 20),
        ];
        for (name, a, dims) in cases {
            let want = reference_singular_values(&dense_of(&a));
            match svd_dim_seed(&a, dims, 42) {
                Ok(svd) => {
                    let got = svd.s.to_vec();
                    let n = got.len().min(want.len());
                    let worst = (0..n)
                        .map(|i| (got[i] - want[i]).abs() / want[i].abs().max(1e-30))
                        .fold(0.0f64, f64::max);
                    println!(
                        "{name:<24} dims={dims:<3} d={:<3} top_rel={:>9.2e} worst_rel={worst:>9.2e}",
                        svd.d,
                        (got[0] - want[0]).abs() / want[0].abs()
                    );
                }
                Err(e) => println!("{name:<24} dims={dims:<3} ERROR {e}"),
            }
        }
    }
}
