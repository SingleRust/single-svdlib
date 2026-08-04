//! Tall-skinny QR.
//!
//! For an `m × n` matrix with `m ≫ n` — the shape every randomized range finder
//! produces — a conventional Householder QR is a sequence of `n` passes over all `m`
//! rows, each pass dependent on the last. TSQR (Demmel, Grigori, Hoemmen & Langou)
//! instead splits the rows into `P` independent panels:
//!
//! 1. each panel gets its own local QR, in parallel, yielding an `n × n` factor `Rᵢ`;
//! 2. the `Rᵢ` are stacked into a `Pn × n` matrix and reduced by one small QR, giving
//!    the final `R` and a `Pn × n` factor `Q_s`;
//! 3. each panel's `Qᵢ` is multiplied by its `n × n` block of `Q_s`, in parallel.
//!
//! Work is `O(mn²/P)` and the only auxiliary storage is `O(Pn²)` — independent of `m`.
//! The alternative in 1.x was `nalgebra`'s dense QR on the full `m × l` matrix, which
//! is serial in `m` and materialises its own `m × l` factor.

// Numeric kernels index several arrays in step from one loop variable, and
// offset arithmetic is load-bearing; iterator rewrites obscure which array an
// index belongs to.
#![allow(clippy::needless_range_loop)]

use crate::error::{Result, SvdLibError};
use crate::types::SvdFloat;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Axis};
use num_traits::Float;
use rayon::prelude::*;

/// Below this many rows, panel splitting costs more than it saves.
const MIN_ROWS_FOR_PANELS: usize = 2048;

/// Build a Householder reflector `H = I - τ v vᵀ` with `v[0] == 1` such that
/// `H x = β e₁`. Returns `(β, τ)` and overwrites `x[1..]` with `v[1..]`.
///
/// Mirrors LAPACK `dlarfg`, including the sign choice that avoids cancellation.
fn housegen<T: SvdFloat>(x: &mut [T]) -> (T, T) {
    let n = x.len();
    if n == 0 {
        return (T::zero(), T::zero());
    }
    let alpha = x[0];
    if n == 1 {
        return (alpha, T::zero());
    }
    // ||x[1..]||
    let tail_norm = {
        let mut acc = T::zero();
        for &v in &x[1..] {
            acc += v * v;
        }
        acc.sqrt()
    };
    if tail_norm.is_zero() {
        return (alpha, T::zero());
    }
    let norm = Float::hypot(alpha, tail_norm);
    // Choose the sign opposite alpha so `alpha - beta` cannot cancel.
    let beta = if alpha >= T::zero() { -norm } else { norm };
    let tau = (beta - alpha) / beta;
    let scale = T::one() / (alpha - beta);
    for v in &mut x[1..] {
        *v *= scale;
    }
    (beta, tau)
}

/// Apply `H = I - τ v vᵀ` on the left of `c`, where `v[0] == 1` and `v[1..] == v_tail`.
fn apply_reflector<T: SvdFloat>(tau: T, v_tail: &[T], mut c: ArrayViewMut2<T>) {
    if tau.is_zero() {
        return;
    }
    let (p, q) = (c.nrows(), c.ncols());
    debug_assert_eq!(v_tail.len() + 1, p);
    // w = cᵀ v
    let mut w = vec![T::zero(); q];
    for (j, wj) in w.iter_mut().enumerate() {
        let mut acc = c[[0, j]];
        for (i, &vi) in v_tail.iter().enumerate() {
            acc += vi * c[[i + 1, j]];
        }
        *wj = acc;
    }
    // c -= tau v wᵀ
    for (j, &wj) in w.iter().enumerate() {
        let f = tau * wj;
        if f.is_zero() {
            continue;
        }
        c[[0, j]] -= f;
        for (i, &vi) in v_tail.iter().enumerate() {
            c[[i + 1, j]] -= f * vi;
        }
    }
}

/// Householder QR of `a` (`r × n`, `r >= n`), in place.
///
/// On return the strict lower triangle of `a` holds the reflector tails and the upper
/// triangle holds `R`. `tau` receives the `n` reflector scalars.
fn qr_in_place<T: SvdFloat>(mut a: ArrayViewMut2<T>, tau: &mut [T]) {
    let (r, n) = a.dim();
    let k = n.min(r);
    for j in 0..k {
        // Generate the reflector from the column below and including the diagonal.
        let mut col: Vec<T> = a.slice(s![j.., j]).to_vec();
        let (beta, t) = housegen(&mut col);
        tau[j] = t;
        a[[j, j]] = beta;
        for (i, &v) in col[1..].iter().enumerate() {
            a[[j + 1 + i, j]] = v;
        }
        if j + 1 < n {
            let v_tail: Vec<T> = col[1..].to_vec();
            apply_reflector(t, &v_tail, a.slice_mut(s![j.., j + 1..]));
        }
    }
}

/// Extract the `n × n` upper-triangular `R` from a factored panel.
fn extract_r<T: SvdFloat>(a: &ArrayView2<T>, n: usize) -> Array2<T> {
    let mut r = Array2::<T>::zeros((n, n));
    for i in 0..n.min(a.nrows()) {
        for j in i..n {
            r[[i, j]] = a[[i, j]];
        }
    }
    r
}

/// Overwrite a factored panel with its thin `Q` (`r × n`), from the reflectors and
/// `tau` produced by [`qr_in_place`]. Mirrors LAPACK `dorgqr`.
fn form_q_in_place<T: SvdFloat>(mut a: ArrayViewMut2<T>, tau: &[T]) {
    let (r, n) = a.dim();
    let k = n.min(r);
    // Stash the reflector tails before overwriting with the identity.
    let mut vs: Vec<Vec<T>> = Vec::with_capacity(k);
    for j in 0..k {
        vs.push(a.slice(s![j + 1.., j]).to_vec());
    }
    a.fill(T::zero());
    for j in 0..n.min(r) {
        a[[j, j]] = T::one();
    }
    // Apply H_0 H_1 ... H_{k-1} in reverse.
    for j in (0..k).rev() {
        apply_reflector(tau[j], &vs[j], a.slice_mut(s![j.., ..]));
    }
}

/// `panel = panel · block`, where `block` is `n × n`. One `n`-element row temp.
fn mul_panel_by_block<T: SvdFloat>(mut panel: ArrayViewMut2<T>, block: &Array2<T>) {
    let n = panel.ncols();
    debug_assert_eq!(block.dim(), (n, n));
    let mut tmp = vec![T::zero(); n];
    for mut row in panel.rows_mut() {
        for (j, t) in tmp.iter_mut().enumerate() {
            let mut acc = T::zero();
            for i in 0..n {
                acc += row[i] * block[[i, j]];
            }
            *t = acc;
        }
        for (dst, &src) in row.iter_mut().zip(tmp.iter()) {
            *dst = src;
        }
    }
}

/// Thin QR of a tall matrix.
///
/// Overwrites `a` (`m × n`, `m >= n`) with an orthonormal `Q` and returns the `n × n`
/// upper-triangular `R`, so that the original `a == Q · R`.
///
/// # Errors
/// If `a` is wider than it is tall.
pub fn tsqr<T: SvdFloat>(a: &mut Array2<T>) -> Result<Array2<T>> {
    let (m, n) = a.dim();
    if n > m {
        return Err(SvdLibError::shape(format!(
            "tsqr needs at least as many rows as columns, got {m}x{n}"
        )));
    }
    if n == 0 || m == 0 {
        return Ok(Array2::zeros((n, n)));
    }

    // How many panels can we cut while keeping each at least `n` rows deep? A panel
    // shallower than `n` has a rank-deficient local R and buys nothing.
    let threads = rayon::current_num_threads().max(1);
    let panels = threads.min(m / n.max(1)).max(1);

    if panels == 1 || m < MIN_ROWS_FOR_PANELS {
        let mut tau = vec![T::zero(); n];
        qr_in_place(a.view_mut(), &mut tau);
        let r = extract_r(&a.view(), n);
        form_q_in_place(a.view_mut(), &tau);
        return Ok(r);
    }

    // Panel boundaries, each at least `n` rows.
    let base = m / panels;
    let rem = m % panels;
    let mut bounds = Vec::with_capacity(panels + 1);
    bounds.push(0usize);
    for p in 0..panels {
        let take = base + usize::from(p < rem);
        bounds.push(bounds[p] + take);
    }
    debug_assert_eq!(*bounds.last().unwrap(), m);

    // Stage 1: local QR per panel, in parallel. Each returns its own R and leaves the
    // panel holding its local thin Q.
    let mut panel_views: Vec<ArrayViewMut2<T>> = Vec::with_capacity(panels);
    {
        let mut rest = a.view_mut();
        for p in 0..panels {
            let take = bounds[p + 1] - bounds[p];
            let (head, tail) = rest.split_at(Axis(0), take);
            panel_views.push(head);
            rest = tail;
        }
    }

    // Factored in place through the view: no per-panel copy, so the only auxiliary
    // storage is the stacked R factors.
    let locals: Vec<Array2<T>> = panel_views
        .par_iter_mut()
        .map(|panel| {
            let mut tau = vec![T::zero(); n];
            qr_in_place(panel.view_mut(), &mut tau);
            let r = extract_r(&panel.view(), n);
            form_q_in_place(panel.view_mut(), &tau);
            r
        })
        .collect();

    // Stage 2: one small QR of the stacked R factors.
    let mut stacked = Array2::<T>::zeros((panels * n, n));
    for (p, r) in locals.iter().enumerate() {
        stacked.slice_mut(s![p * n..(p + 1) * n, ..]).assign(r);
    }
    let mut tau = vec![T::zero(); n];
    qr_in_place(stacked.view_mut(), &mut tau);
    let r_final = extract_r(&stacked.view(), n);
    form_q_in_place(stacked.view_mut(), &tau);

    // Stage 3: fold each block of Q_s back into its panel, in parallel.
    let blocks: Vec<Array2<T>> = (0..panels)
        .map(|p| stacked.slice(s![p * n..(p + 1) * n, ..]).to_owned())
        .collect();
    panel_views
        .par_iter_mut()
        .zip(blocks.par_iter())
        .for_each(|(panel, block)| mul_panel_by_block(panel.view_mut(), block));

    Ok(r_final)
}

/// Orthonormalise the columns of `a` in place, discarding `R`.
///
/// This is the operation a randomized range finder actually wants from a QR step.
pub fn orthonormalize<T: SvdFloat>(a: &mut Array2<T>) -> Result<()> {
    tsqr(a).map(|_| ())
}

/// `‖AᵀA − I‖_F`, the departure from orthonormality of `a`'s columns.
pub fn orthogonality_error<T: SvdFloat>(a: &ArrayView2<T>) -> T {
    let n = a.ncols();
    let mut acc = T::zero();
    for i in 0..n {
        for j in 0..n {
            let mut dot = T::zero();
            for k in 0..a.nrows() {
                dot += a[[k, i]] * a[[k, j]];
            }
            let target = if i == j { T::one() } else { T::zero() };
            let d = dot - target;
            acc += d * d;
        }
    }
    acc.sqrt()
}

/// Row count above which [`tsqr`] engages panel splitting; exposed for tests that need
/// to exercise both paths.
#[doc(hidden)]
pub const PANEL_THRESHOLD: usize = MIN_ROWS_FOR_PANELS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Lcg;

    fn random_tall(m: usize, n: usize, seed: u64) -> Array2<f64> {
        let mut rng = Lcg::new(seed);
        Array2::from_shape_fn((m, n), |_| rng.signed())
    }

    fn frob(a: &Array2<f64>) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn check_qr(m: usize, n: usize, seed: u64) {
        let original = random_tall(m, n, seed);
        let mut q = original.clone();
        let r = tsqr(&mut q).unwrap();

        assert_eq!(q.dim(), (m, n), "Q shape");
        assert_eq!(r.dim(), (n, n), "R shape");

        // R upper triangular.
        for i in 1..n {
            for j in 0..i {
                assert!(
                    r[[i, j]].abs() < 1e-12,
                    "R not upper triangular at ({i},{j}): {}",
                    r[[i, j]]
                );
            }
        }

        // Q orthonormal.
        let orth = orthogonality_error(&q.view());
        assert!(orth < 1e-10, "||Q^T Q - I||_F = {orth:.3e} for {m}x{n}");

        // A == Q R.
        let recon = q.dot(&r);
        let err = frob(&(&recon - &original)) / frob(&original).max(1e-30);
        assert!(err < 1e-10, "||A - QR||/||A|| = {err:.3e} for {m}x{n}");
    }

    #[test]
    fn serial_path_shapes() {
        for (m, n) in [(8usize, 3usize), (64, 8), (100, 1), (50, 50), (257, 17)] {
            check_qr(m, n, 42 + m as u64);
        }
    }

    /// Above `PANEL_THRESHOLD` the multi-panel reduction runs; it must agree with the
    /// serial path to the same accuracy.
    #[test]
    fn panel_path_matches_serial_accuracy() {
        for (m, n) in [(4096usize, 16usize), (5000, 32), (8192, 8)] {
            check_qr(m, n, 7 + m as u64);
        }
    }

    /// The panelled and serial paths must produce the same factorization, up to the
    /// column sign freedom that Householder QR leaves.
    #[test]
    fn panel_and_serial_agree_on_r_magnitude() {
        let (m, n) = (4096, 12);
        let original = random_tall(m, n, 99);

        let mut q_panel = original.clone();
        let r_panel = tsqr(&mut q_panel).unwrap();

        // Force the serial path by running inside a single-thread pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (r_serial, q_serial) = pool.install(|| {
            let mut q = original.clone();
            let r = tsqr(&mut q).unwrap();
            (r, q)
        });

        for i in 0..n {
            for j in i..n {
                approx::assert_relative_eq!(
                    r_panel[[i, j]].abs(),
                    r_serial[[i, j]].abs(),
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
        assert!(orthogonality_error(&q_serial.view()) < 1e-10);
    }

    #[test]
    fn rejects_wide_matrices() {
        let mut a = random_tall(4, 9, 1);
        assert!(matches!(tsqr(&mut a), Err(SvdLibError::ShapeMismatch(_))));
    }

    /// A rank-deficient operand still yields an orthonormal Q; the deficiency shows up
    /// as (near-)zero diagonal entries of R, not as a loss of orthogonality.
    #[test]
    fn handles_rank_deficient_input() {
        let (m, n) = (200, 6);
        let mut a = random_tall(m, n, 3);
        // Make column 4 a copy of column 1.
        let c1 = a.column(1).to_owned();
        a.column_mut(4).assign(&c1);
        let mut q = a.clone();
        let r = tsqr(&mut q).unwrap();
        let recon = q.dot(&r);
        let err = frob(&(&recon - &a)) / frob(&a);
        assert!(err < 1e-10, "||A - QR||/||A|| = {err:.3e}");
        assert!(
            r[[4, 4]].abs() < 1e-10,
            "expected a deficient pivot, got {}",
            r[[4, 4]]
        );
    }

    #[test]
    fn f32_works() {
        let mut rng = Lcg::new(5);
        let original = Array2::from_shape_fn((512, 8), |_| rng.signed() as f32);
        let mut q = original.clone();
        let r = tsqr(&mut q).unwrap();
        let orth = orthogonality_error(&q.view());
        assert!(orth < 1e-3, "f32 ||Q^T Q - I||_F = {orth:.3e}");
        let recon = q.dot(&r);
        let num: f32 = (&recon - &original)
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        let den: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(num / den < 1e-4, "f32 reconstruction {:.3e}", num / den);
    }
}
