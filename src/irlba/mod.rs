//! Thick-restarted Lanczos bidiagonalization.
//!
//! Golub–Kahan–Lanczos bidiagonalization with augmented thick restarts, following
//! Baglama & Reichel (2005). This is the algorithm behind R's `irlba` and, in spirit,
//! `scipy.sparse.linalg.svds`.
//!
//! # Why this rather than [`crate::lanczos`]
//!
//! LAS2 keeps every Lanczos vector it generates, so its basis grows with the iteration
//! count — unbounded in practice, since `iterations` defaults to `min(rows, cols)`.
//! Here the basis is fixed at `work` vectors (`rank + 7` by default) no matter how many
//! restarts are needed, so peak memory is known before the solve starts:
//!
//! ```text
//! (work + 1) · cols + work · rows   scalars
//! ```
//!
//! For 200k × 30k at rank 50 that is about 95 MiB and it does not grow.
//!
//! # Method
//!
//! Each cycle extends the factorization
//!
//! ```text
//! A·V  = U·B
//! Aᵀ·U = V·Bᵀ + β·v_next·eᵀ
//! ```
//!
//! to `work` columns, where `B` is small and bidiagonal, then takes the SVD of `B`.
//! Its singular values are the Ritz estimates and `|β · P[work-1, i]|` is the residual
//! for triplet `i`. Unconverged cycles restart from the `rank` best Ritz vectors plus
//! the residual direction, which preserves the factorization's structure — the restart
//! costs one extra column in `B` rather than throwing the subspace away.

use crate::dense::{small_svd, svd_flip};
use crate::error::{Result, SvdLibError};
use crate::matrix::{SparseMat, SparseMatDense};
use crate::types::{Algorithm, Detail, Diagnostics, SvdFloat, SvdRec};
use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{rng, Rng, RngExt, SeedableRng};

/// Default relative residual tolerance.
pub const DEFAULT_TOL: f64 = 1e-10;
/// Default extra basis vectors beyond the requested rank.
pub const DEFAULT_EXTRA_WORK: usize = 7;
/// Default cap on restart cycles.
pub const DEFAULT_MAX_RESTARTS: usize = 1000;

/// Configuration for [`svd_with`].
#[derive(Debug, Clone)]
pub struct IrlbaConfig {
    /// Number of singular triplets wanted.
    pub rank: usize,
    /// Basis size. Must exceed `rank`; defaults to `rank + 7`, clamped to
    /// `min(rows, cols)`.
    ///
    /// # This is the knob that matters on tall matrices
    ///
    /// Each restart re-orthogonalises against the whole basis, so a step costs
    /// `O(work · (rows + cols))` — on a matrix with a million rows that dominates the
    /// sparse products by a wide margin. A larger `work` makes each step dearer but
    /// converges in far fewer restarts, and the second effect wins comfortably.
    ///
    /// Measured on 400k × 30k restricted to 2238 columns, 50 components
    /// (`cargo run --release --example tune`):
    ///
    /// | `work` | time | matvecs | accuracy |
    /// |---|---|---|---|
    /// | `rank + 7` (default) | 30.2 s | 1515 | 6.7e-17 |
    /// | `rank + 30` | **13.7 s** | 701 | 1.9e-14 |
    /// | `rank + 50` | 18.6 s | 601 | 2.0e-14 |
    /// | `rank + 100` | 21.4 s | 501 | 2.2e-14 |
    ///
    /// So `rank + 30` was **2.2× faster** at the same accuracy. The cost is basis
    /// memory, which is linear in `work`: `work · rows` scalars for the left basis, or
    /// 600 MiB rather than 456 MiB at a million rows. The default stays conservative
    /// because memory is the reason to choose this crate; raise it when you have the
    /// headroom.
    ///
    /// Run the `tune` example on your own shape rather than trusting these numbers —
    /// the optimum moves with the aspect ratio and the spectrum.
    pub work: Option<usize>,
    /// Relative residual tolerance: triplet `i` is accepted once its residual falls
    /// below `tol · σ_max`.
    pub tol: f64,
    /// Cap on restart cycles before giving up.
    pub max_restarts: usize,
    /// Fixed seed for the starting vector; `None` draws from the OS.
    pub seed: Option<u64>,
    /// Subtract column means without materialising the centered matrix — i.e. PCA
    /// rather than plain SVD.
    pub mean_center: bool,
    /// Refuse to return triplets that did not reach [`tol`](Self::tol). Defaults to
    /// `true`.
    ///
    /// Exhausting the restart budget means the answer is a best effort of unknown
    /// quality. Returning it as `Ok` puts the burden on every caller to remember to
    /// inspect the diagnostics, and a pipeline that forgets gets a silently degraded
    /// decomposition feeding whatever comes next. Failing loudly is the safer default;
    /// set this to `false` if a best effort is genuinely what you want, then check
    /// [`SvdRec::converged`].
    pub require_convergence: bool,
}

impl IrlbaConfig {
    /// Configuration for `rank` triplets, everything else defaulted.
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            work: None,
            tol: DEFAULT_TOL,
            max_restarts: DEFAULT_MAX_RESTARTS,
            seed: None,
            mean_center: false,
            require_convergence: true,
        }
    }
    /// Set the basis size.
    pub fn work(mut self, work: usize) -> Self {
        self.work = Some(work);
        self
    }
    /// Set the relative residual tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the restart cap.
    pub fn max_restarts(mut self, n: usize) -> Self {
        self.max_restarts = n;
        self
    }
    /// Fix the seed, making the result reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Enable implicit mean centering.
    pub fn mean_center(mut self, yes: bool) -> Self {
        self.mean_center = yes;
        self
    }
    /// Accept a best-effort result instead of failing when the restart budget runs out.
    pub fn allow_unconverged(mut self) -> Self {
        self.require_convergence = false;
        self
    }
}

/// `rank` largest singular triplets, defaults throughout.
pub fn svd<T: SvdFloat, M: SparseMat<T>>(a: &M, rank: usize) -> Result<SvdRec<T>> {
    svd_with(a, &IrlbaConfig::new(rank), None)
}

/// `rank` largest singular triplets with a fixed seed.
pub fn svd_seed<T: SvdFloat, M: SparseMat<T>>(a: &M, rank: usize, seed: u64) -> Result<SvdRec<T>> {
    svd_with(a, &IrlbaConfig::new(rank).seed(seed), None)
}

/// PCA: `rank` largest singular triplets of the implicitly mean-centered matrix.
///
/// Requires [`SparseMatDense`] only to obtain the column means; the solve itself uses
/// matrix-vector products throughout.
pub fn svd_centered<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    rank: usize,
    seed: Option<u64>,
) -> Result<SvdRec<T>> {
    let means = a.col_means();
    let mut cfg = IrlbaConfig::new(rank).mean_center(true);
    cfg.seed = seed;
    svd_with(a, &cfg, Some(means))
}

/// An operand with optional implicit mean centering applied around its products.
///
/// Centering a sparse matrix would destroy its sparsity, so the shift is folded into
/// each product as the rank-1 term it is.
struct Op<'a, T, M> {
    a: &'a M,
    /// Column means, when centering.
    means: Option<&'a [T]>,
    _p: std::marker::PhantomData<T>,
}

impl<'a, T: SvdFloat, M: SparseMat<T>> Op<'a, T, M> {
    fn rows(&self) -> usize {
        self.a.rows()
    }
    fn cols(&self) -> usize {
        self.a.cols()
    }

    /// `y = A·x` (`trans == false`) or `y = Aᵀ·x` (`trans == true`), centered if
    /// configured.
    fn mul(&self, x: &[T], y: &mut [T], trans: bool) {
        self.a.mul_vec(x, y, trans);
        let Some(m) = self.means else { return };
        if !trans {
            // (A − 1·mᵀ)x = A·x − 1·(m·x)
            let c: T = m.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            for yi in y.iter_mut() {
                *yi -= c;
            }
        } else {
            // (A − 1·mᵀ)ᵀx = Aᵀ·x − m·(1ᵀx)
            let sum: T = x.iter().copied().sum();
            for (yi, &mi) in y.iter_mut().zip(m.iter()) {
                *yi -= mi * sum;
            }
        }
    }
}

/// Two-pass classical Gram–Schmidt against the first `count` rows of `basis`.
///
/// One pass leaves `O(κ·eps)` non-orthogonality; twice is enough to reach machine
/// precision (Kahan–Parlett), and expressing it as BLAS-2 products keeps it far cheaper
/// than the `count` separate axpy pairs the equivalent loop would issue.
///
/// `coeffs` is a caller-owned scratch buffer of at least `count` elements, and the
/// correction is accumulated straight into `w` via `general_mat_vec_mul`. Allocating
/// either of those here would mean two heap allocations per Lanczos step, one of them
/// the full length of `w`.
/// Returns the *total* coefficient removed along each basis vector, accumulated over
/// both passes.
///
/// Callers must record these. In an undisturbed Krylov recurrence they are numerical
/// drift, around zero, and discarding them is harmless. After a breakdown restart they
/// are not: the injected random direction has genuine components along every previous
/// vector, and dropping them silently breaks `A·V = U·B`, so `B`'s spectrum stops being
/// `A`'s. That surfaced as a *skipped* singular value — ten real triplets returned, but
/// the ninth-largest missing — with every residual small enough to claim convergence.
fn reorthogonalize<T: SvdFloat>(
    w: &mut Array1<T>,
    basis: &ArrayView2<T>,
    count: usize,
    coeffs: &mut Array1<T>,
) {
    if count == 0 {
        return;
    }
    let b = basis.slice(s![..count, ..]);
    let mut total = Array1::<T>::zeros(count);
    {
        let mut c = coeffs.slice_mut(s![..count]);
        for _ in 0..2 {
            // c = b · w
            ndarray::linalg::general_mat_vec_mul(T::one(), &b, w, T::zero(), &mut c);
            // w = w - bᵀ · c
            ndarray::linalg::general_mat_vec_mul(-T::one(), &b.t(), &c, T::one(), w);
            total += &c;
        }
    }
    coeffs.slice_mut(s![..count]).assign(&total);
}

/// Draw a unit vector orthogonal to the first `count` rows of `basis`.
///
/// Returns `false` when the space is exhausted — no direction remains.
///
/// The quality floor matters. A random draw that happens to land mostly inside the
/// existing span leaves a tiny residual, and normalising that residual scales the
/// Gram-Schmidt error up by its reciprocal. Accepting any non-zero residual let a
/// basis vector be orthogonal to only ~1e-8, which propagated into the returned
/// singular vectors as `||UᵀU - I|| = 2e-7`. Re-drawing costs nothing here and keeps
/// the amplification at O(1).
fn random_orthogonal<T: SvdFloat>(
    out: &mut Array1<T>,
    basis: &ArrayView2<T>,
    count: usize,
    coeffs: &mut Array1<T>,
    rng_state: &mut StdRng,
) -> bool {
    let floor = T::from_f64_val(0.1);
    for _ in 0..4 {
        random_unit(out, rng_state);
        // The coefficients here describe the *random* vector, not `A·v`, so unlike the
        // main path they must not be recorded in `B`.
        reorthogonalize(out, basis, count, coeffs);
        let n = norm(out);
        if n > floor && num_traits::Float::is_finite(n) {
            *out /= n;
            return true;
        }
    }
    false
}

fn norm<T: SvdFloat>(v: &Array1<T>) -> T {
    v.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Fill `v` with a deterministic unit random vector.
fn random_unit<T: SvdFloat>(v: &mut Array1<T>, rng_state: &mut StdRng) {
    for x in v.iter_mut() {
        *x = T::from_f64_val(rng_state.random_range(-1.0..1.0));
    }
    let n = norm(v);
    if n > T::zero() {
        *v /= n;
    } else {
        v.fill(T::zero());
        v[0] = T::one();
    }
}

/// Compute a decomposition with explicit configuration.
///
/// `means`, when given, must have length `a.cols()` and is only consulted if
/// `cfg.mean_center` is set. [`svd_centered`] is the convenient entry point.
pub fn svd_with<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    cfg: &IrlbaConfig,
    means: Option<Array1<T>>,
) -> Result<SvdRec<T>> {
    let (rows, cols) = (a.rows(), a.cols());
    let min_dim = rows.min(cols);

    if cfg.rank == 0 {
        return Err(SvdLibError::invalid("irlba: rank must be at least 1"));
    }
    if cfg.rank > min_dim {
        return Err(SvdLibError::invalid(format!(
            "irlba: rank {} exceeds min(rows, cols) = {min_dim}",
            cfg.rank
        )));
    }
    if cfg.mean_center {
        match &means {
            Some(m) if m.len() == cols => {}
            Some(m) => {
                return Err(SvdLibError::shape(format!(
                    "irlba: means has length {} but the matrix has {cols} columns",
                    m.len()
                )))
            }
            None => {
                return Err(SvdLibError::invalid(
                    "irlba: mean_center is set but no means were supplied; \
                     use `svd_centered`",
                ))
            }
        }
    }

    let k = cfg.rank;
    let seed = cfg.seed.unwrap_or_else(|| rng().next_u64());

    // A matrix with a dimension of 1 has exactly one singular triplet and no Krylov
    // subspace to build. Solve it in closed form rather than rejecting it: masking down
    // to a single column is a perfectly ordinary thing to do.
    if min_dim == 1 {
        return trivial_rank_one(a, &means, cfg, seed);
    }

    let work = cfg
        .work
        .unwrap_or(k + DEFAULT_EXTRA_WORK)
        .clamp(k + 1, min_dim.max(k + 1))
        .min(min_dim);
    if work <= k {
        return Err(SvdLibError::invalid(format!(
            "irlba: rank {k} needs a basis of at least {} vectors but the matrix only \
             admits {min_dim}; request at most {} triplets, or use \
             `single_svdlib::randomized`, which supports the full rank",
            k + 1,
            min_dim - 1
        )));
    }

    let means_slice = if cfg.mean_center {
        means.as_ref().map(|m| m.as_slice().unwrap())
    } else {
        None
    };
    let op = Op {
        a,
        means: means_slice,
        _p: std::marker::PhantomData,
    };

    let mut state = Solve::new(&op, work, k, cfg.tol, seed);
    let outcome = state.run(cfg.max_restarts)?;

    // Assemble the requested triplets.
    let Solve { v, u, .. } = state;
    let SolveOutcome {
        p,
        q,
        sigma,
        restarts,
        converged,
        max_residual,
        matvecs,
    } = outcome;

    // u_out[r, i] = Σ_j P[j, i] · U[j, r]        (rows × k)
    // vt_out[i, c] = Σ_j Q[j, i] · V[j, c]       (k × cols)
    let pk = p.slice(s![.., ..k]);
    let qk = q.slice(s![.., ..k]);
    let mut u_out = pk
        .t()
        .dot(&u.slice(s![..work, ..]))
        .reversed_axes()
        .to_owned();
    let mut vt_out = qk.t().dot(&v.slice(s![..work, ..])).to_owned();
    let s_out = sigma.slice(s![..k]).to_owned();

    if cfg.require_convergence && !converged {
        return Err(SvdLibError::failed(
            "irlba",
            format!(
                "did not converge in {restarts} restarts: largest residual is {:.3e} \
                 against a threshold of {:.3e} (tol {:.1e} x sigma_max). Raise \
                 `max_restarts` or `work`, loosen `tol`, or call `allow_unconverged` to \
                 accept a best effort.",
                max_residual.to_f64(),
                cfg.tol * sigma[0].to_f64(),
                cfg.tol,
            ),
        ));
    }

    // Pin the per-triplet sign so repeat runs agree.
    svd_flip(&mut u_out, &mut vt_out);

    Ok(SvdRec {
        d: k,
        u: u_out,
        s: s_out,
        vt: vt_out,
        total_squared_norm: T::from_f64_val(crate::matrix::total_squared_norm(
            a,
            means.as_ref().map(|m| m.view()),
        )),
        diagnostics: Diagnostics {
            algorithm: Algorithm::Irlba,
            non_zero: a.nnz(),
            dimensions: k,
            significant_values: k,
            transposed: false,
            random_seed: seed,
            matvecs,
            detail: Detail::Irlba {
                restarts,
                converged,
                tolerance: T::from_f64_val(cfg.tol),
                max_residual,
            },
        },
    })
}

/// The `min(rows, cols) == 1` case, in closed form.
///
/// Such a matrix is a single row or column, so it has exactly one singular value —
/// the vector's norm — with the unit vector on the short side and the normalised
/// vector on the long side.
fn trivial_rank_one<T: SvdFloat, M: SparseMat<T>>(
    a: &M,
    means: &Option<Array1<T>>,
    cfg: &IrlbaConfig,
    seed: u64,
) -> Result<SvdRec<T>> {
    let (rows, cols) = (a.rows(), a.cols());
    let op = Op {
        a,
        means: if cfg.mean_center {
            means.as_ref().map(|m| m.as_slice().unwrap())
        } else {
            None
        },
        _p: std::marker::PhantomData,
    };

    // Materialise the single row (or column) by probing with a unit vector.
    let (long, short, trans) = if rows == 1 {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };
    let mut probe = vec![T::zero(); short];
    probe[0] = T::one();
    let mut vec = vec![T::zero(); long];
    op.mul(&probe, &mut vec, trans);

    let sigma = vec.iter().map(|&x| x * x).sum::<T>().sqrt();
    if !num_traits::Float::is_finite(sigma) {
        return Err(SvdLibError::failed("irlba", "the operand is not finite"));
    }

    let (u, vt) = if sigma > T::zero() {
        let unit: Vec<T> = vec.iter().map(|&x| x / sigma).collect();
        if rows == 1 {
            // 1 x cols: u = [1], vt = row / sigma
            (
                Array2::from_shape_vec((1, 1), vec![T::one()])?,
                Array2::from_shape_vec((1, cols), unit)?,
            )
        } else {
            // rows x 1: u = col / sigma, vt = [1]
            (
                Array2::from_shape_vec((rows, 1), unit)?,
                Array2::from_shape_vec((1, 1), vec![T::one()])?,
            )
        }
    } else {
        // The zero matrix: any unit vectors will do.
        let mut u = Array2::<T>::zeros((rows, 1));
        let mut vt = Array2::<T>::zeros((1, cols));
        u[[0, 0]] = T::one();
        vt[[0, 0]] = T::one();
        (u, vt)
    };

    Ok(SvdRec {
        d: 1,
        u,
        s: Array1::from_vec(vec![sigma]),
        vt,
        total_squared_norm: T::from_f64_val(crate::matrix::total_squared_norm(
            a,
            if cfg.mean_center {
                means.as_ref().map(|m| m.view())
            } else {
                None
            },
        )),
        diagnostics: Diagnostics {
            algorithm: Algorithm::Irlba,
            non_zero: a.nnz(),
            dimensions: 1,
            significant_values: 1,
            transposed: false,
            random_seed: seed,
            matvecs: 1,
            detail: Detail::Irlba {
                restarts: 0,
                converged: true,
                tolerance: T::from_f64_val(cfg.tol),
                max_residual: T::zero(),
            },
        },
    })
}

struct SolveOutcome<T> {
    /// Left singular vectors of `B`, `work × work`.
    p: Array2<T>,
    /// Right singular vectors of `B` (as columns), `work × work`.
    q: Array2<T>,
    sigma: Array1<T>,
    restarts: usize,
    converged: bool,
    max_residual: T,
    matvecs: usize,
}

/// The bidiagonalization state. Vectors are stored as **rows** so each is contiguous
/// and can be handed to [`SparseMat::mul_vec`] without a copy.
struct Solve<'a, T, M> {
    op: &'a Op<'a, T, M>,
    work: usize,
    k: usize,
    tol: f64,
    /// `(work + 1) × cols`
    v: Array2<T>,
    /// `work × rows`
    u: Array2<T>,
    /// `work × work`, bidiagonal plus the restart coupling column.
    b: Array2<T>,
    rng: StdRng,
    matvecs: usize,
    /// Running estimate of `||A||`, taken as the largest recurrence coefficient seen.
    /// The breakdown test has to be relative to this: a rank-deficient operand yields a
    /// coefficient around `1e-17` rather than exactly zero, and dividing by it amplifies
    /// rounding noise to O(1) garbage and then to NaN.
    anorm: T,
}

impl<'a, T: SvdFloat, M: SparseMat<T>> Solve<'a, T, M> {
    fn new(op: &'a Op<'a, T, M>, work: usize, k: usize, tol: f64, seed: u64) -> Self {
        Self {
            op,
            work,
            k,
            tol,
            v: Array2::zeros((work + 1, op.cols())),
            u: Array2::zeros((work, op.rows())),
            b: Array2::zeros((work, work)),
            rng: StdRng::seed_from_u64(seed),
            matvecs: 0,
            anorm: T::zero(),
        }
    }

    /// Below this, a recurrence coefficient is treated as zero and the subspace as
    /// invariant. Scaled by the operator norm so it means the same thing whatever the
    /// matrix's magnitude; when nothing has been seen yet (`anorm == 0`, e.g. an
    /// all-zero matrix) it degenerates to an exact-zero test, which is correct.
    fn breakdown_threshold(&self) -> T {
        let dim = T::from_f64_val((self.op.rows().max(self.op.cols()) as f64).sqrt());
        self.anorm * T::eps() * dim
    }

    /// Extend the factorization from column `start` to `work`.
    ///
    /// `coupling` is the restart's `ρ` vector when `start > 0`: at the first extended
    /// column the new left vector must be orthogonalised against all `k` retained
    /// left Ritz vectors, not just its immediate predecessor.
    ///
    /// Returns `(β, v_next)` — the trailing residual norm and direction.
    fn extend(&mut self, start: usize, coupling: Option<&Array1<T>>) -> Result<(T, Array1<T>)> {
        let (rows, cols) = (self.op.rows(), self.op.cols());
        let mut w = Array1::<T>::zeros(rows);
        let mut z = Array1::<T>::zeros(cols);
        // Reused by every reorthogonalization in this sweep.
        let mut coeffs = Array1::<T>::zeros(self.work + 1);

        for j in start..self.work {
            // w = A·v_j, minus the coupling to the already-built left vectors.
            {
                let vj = self.v.row(j).to_owned();
                self.op
                    .mul(vj.as_slice().unwrap(), w.as_slice_mut().unwrap(), false);
                self.matvecs += 1;
            }
            if j == start && start > 0 {
                let rho = coupling.expect("restart requires a coupling vector");
                // w -= Σ_{i<k} ρ_i · u_i
                let uk = self.u.slice(s![..self.k, ..]);
                w -= &uk.t().dot(rho);
            } else if j > 0 {
                let beta_prev = self.b[[j - 1, j]];
                let uprev = self.u.row(j - 1);
                w.scaled_add(-beta_prev, &uprev);
            }

            {
                let ub = self.u.view();
                reorthogonalize(&mut w, &ub, j, &mut coeffs);
            }
            // Record what reorthogonalization removed. `A·v_j = Σ_i B[i,j]·u_i` only
            // holds if these land in B; see `reorthogonalize`.
            for i in 0..j {
                self.b[[i, j]] += coeffs[i];
            }
            let alpha = norm(&w);
            // A non-finite norm means the operand (or the iterate) is poisoned. Bail
            // immediately: continuing would spend the whole restart budget producing
            // NaN and then report a residual that means nothing.
            if !num_traits::Float::is_finite(alpha) {
                return Err(SvdLibError::failed(
                    "irlba",
                    "the left Krylov vector became non-finite; the matrix most likely \
                     contains NaN or infinity",
                ));
            }
            // `alpha` is the true recurrence coefficient even when the subspace has
            // gone invariant, in which case it is (numerically) zero and a random
            // direction carries the basis forward. Recording the *random* vector's norm
            // here instead would invent a singular value out of nothing — on an
            // all-zero matrix that reported 2.9.
            let alpha_kept = if alpha <= self.breakdown_threshold() {
                let ub = self.u.view();
                if !random_orthogonal(&mut w, &ub, j, &mut coeffs, &mut self.rng) {
                    // Same completion case as on the right, reached when `work` meets
                    // `rows`: no direction remains orthogonal to those already held.
                    return Ok((T::zero(), Array1::zeros(cols)));
                }
                T::zero()
            } else {
                self.anorm = Float::max(self.anorm, alpha);
                w /= alpha;
                alpha
            };
            self.u.row_mut(j).assign(&w);
            self.b[[j, j]] = alpha_kept;

            // z = Aᵀ·u_j − α·v_j
            self.op
                .mul(w.as_slice().unwrap(), z.as_slice_mut().unwrap(), true);
            self.matvecs += 1;
            {
                let vj = self.v.row(j);
                z.scaled_add(-alpha_kept, &vj);
            }
            {
                let vb = self.v.view();
                reorthogonalize(&mut z, &vb, j + 1, &mut coeffs);
            }
            let beta = norm(&z);
            if !num_traits::Float::is_finite(beta) {
                return Err(SvdLibError::failed(
                    "irlba",
                    "the right Krylov vector became non-finite; the matrix most likely \
                     contains NaN or infinity",
                ));
            }
            let (beta_kept, zn) = if beta <= self.breakdown_threshold() {
                let vb = self.v.view();
                if !random_orthogonal(&mut z, &vb, j + 1, &mut coeffs, &mut self.rng) {
                    // No direction left that is orthogonal to the `j + 1` already held:
                    // the basis spans the whole space. That is *completion*, not
                    // failure — `A` has been fully captured, the residual is exactly
                    // zero, and the untouched columns of `B` are correctly zero. It
                    // happens whenever `work` reaches `cols`, which 6% of unseeded runs
                    // on a 4x3 operand did.
                    return Ok((T::zero(), Array1::zeros(cols)));
                }
                (T::zero(), z.clone())
            } else {
                self.anorm = Float::max(self.anorm, beta);
                (beta, &z / beta)
            };
            self.v.row_mut(j + 1).assign(&zn);
            if j + 1 < self.work {
                self.b[[j, j + 1]] = beta_kept;
            } else {
                return Ok((beta_kept, zn));
            }
        }
        unreachable!("extend always terminates at the final column")
    }

    fn run(&mut self, max_restarts: usize) -> Result<SolveOutcome<T>> {
        // Starting vector, drawn from the *row space* rather than from all of R^cols.
        //
        // A random vector generally has a component in `null(A)`. The Krylov space then
        // spends one of its `work` dimensions carrying that component, which is
        // orthogonal to everything `A` can reach, so only `work - 1` row-space
        // directions get explored. When `work` is close to `min(rows, cols)` that costs
        // a real singular value: on an 11x13 operand with a 2-dimensional null space it
        // returned ten genuine triplets while silently skipping the ninth-largest, and
        // reported convergence, because every triplet it *did* return was accurate.
        //
        // `Aᵀ·r` lies in the row space by construction, so one extra product removes the
        // whole failure mode.
        {
            let mut probe = Array1::<T>::zeros(self.op.rows());
            random_unit(&mut probe, &mut self.rng);
            let mut v0 = Array1::<T>::zeros(self.op.cols());
            self.op
                .mul(probe.as_slice().unwrap(), v0.as_slice_mut().unwrap(), true);
            self.matvecs += 1;

            let n = norm(&v0);
            if n > T::zero() && num_traits::Float::is_finite(n) {
                v0 /= n;
            } else {
                // `A` is (numerically) zero, so the row space is empty and any unit
                // vector will do.
                random_unit(&mut v0, &mut self.rng);
            }
            self.v.row_mut(0).assign(&v0);
        }

        let mut start = 0usize;
        let mut coupling: Option<Array1<T>> = None;

        for restart in 0..=max_restarts {
            let (beta, v_next) = self.extend(start, coupling.as_ref())?;

            let svd = small_svd(self.b.view())?;
            let sigma = svd.s;
            let p = svd.u; // work × work
            let q = svd.vt.reversed_axes().as_standard_layout().to_owned(); // work × work

            // Residual for triplet i is |β · P[work-1, i]|.
            let smax = Float::max(sigma[0], T::eps());
            let thresh = T::from_f64_val(self.tol) * smax;
            let mut max_resid = T::zero();
            for i in 0..self.k {
                let r = Float::abs(beta * p[[self.work - 1, i]]);
                if r > max_resid {
                    max_resid = r;
                }
            }

            if max_resid <= thresh || restart == max_restarts {
                return Ok(SolveOutcome {
                    p,
                    q,
                    sigma,
                    restarts: restart,
                    converged: max_resid <= thresh,
                    max_residual: max_resid,
                    matvecs: self.matvecs,
                });
            }

            // Thick restart: retain the k best Ritz pairs plus the residual direction.
            //
            // A·(V·qᵢ) = σᵢ·(U·pᵢ) and Aᵀ·(U·pᵢ) = σᵢ·(V·qᵢ) + ρᵢ·v_next, so the
            // restarted B is diag(σ) with ρ as its final column — the subspace is
            // carried over rather than discarded.
            let vk = q
                .slice(s![.., ..self.k])
                .t()
                .dot(&self.v.slice(s![..self.work, ..]));
            let uk = p
                .slice(s![.., ..self.k])
                .t()
                .dot(&self.u.slice(s![..self.work, ..]));

            let mut rho = Array1::<T>::zeros(self.k);
            for i in 0..self.k {
                rho[i] = beta * p[[self.work - 1, i]];
            }

            self.v.slice_mut(s![..self.k, ..]).assign(&vk);
            self.u.slice_mut(s![..self.k, ..]).assign(&uk);
            self.v.row_mut(self.k).assign(&v_next);

            self.b.fill(T::zero());
            for i in 0..self.k {
                self.b[[i, i]] = sigma[i];
                self.b[[i, self.k]] = rho[i];
            }

            start = self.k;
            coupling = Some(rho);
        }
        unreachable!("the restart loop returns on its final iteration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::SvdMat;
    use crate::testing::{dense_of, gen_lowrank, gen_sparse, reference_singular_values, Lcg};
    use ndarray::Axis;
    use sprs::TriMatI;

    fn diagonal(n: usize) -> SvdMat<f64> {
        let mut t = TriMatI::<f64, u32>::new((n, n));
        for i in 0..n {
            t.add_triplet(i, i, (n - i) as f64);
        }
        t.to_csr::<u64>()
    }

    fn dense_random(r: usize, c: usize, seed: u64) -> SvdMat<f64> {
        let mut rng = Lcg::new(seed);
        let mut t = TriMatI::<f64, u32>::new((r, c));
        for i in 0..r {
            for j in 0..c {
                t.add_triplet(i, j, rng.signed());
            }
        }
        t.to_csr::<u64>()
    }

    /// The acceptance criterion: agreement with a dense LAPACK reference.
    fn assert_matches_lapack(name: &str, a: &SvdMat<f64>, rank: usize, tol: f64) -> SvdRec<f64> {
        let want = reference_singular_values(&dense_of(a));
        let got = svd_seed(a, rank, 42).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.d, rank, "{name}: rank");
        for (i, &g) in got.s.iter().enumerate() {
            let rel = (g - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(
                rel < tol,
                "{name}: singular value {i}: irlba {g:.12e} vs LAPACK {:.12e} (rel {rel:.3e})",
                want[i]
            );
        }
        got
    }

    #[test]
    fn exact_on_diagonal_matrix() {
        // Singular values are exactly 40, 39, 38, ... — the case LAS2 gets 20% wrong.
        let a = diagonal(40);
        let got = assert_matches_lapack("diagonal_40", &a, 10, 1e-10);
        approx::assert_relative_eq!(got.s[0], 40.0, max_relative = 1e-10);
        approx::assert_relative_eq!(got.s[9], 31.0, max_relative = 1e-10);
    }

    #[test]
    fn matches_lapack_on_dense_random() {
        assert_matches_lapack("dense_random_60x40", &dense_random(60, 40, 3), 10, 1e-9);
    }

    #[test]
    fn matches_lapack_on_lowrank() {
        assert_matches_lapack("lowrank_80x50_r8", &gen_lowrank(80, 50, 8, 21), 8, 1e-9);
        assert_matches_lapack(
            "lowrank_200x80_r10",
            &gen_lowrank(200, 80, 10, 555),
            15,
            1e-8,
        );
    }

    #[test]
    fn matches_lapack_on_sparse() {
        assert_matches_lapack("sparse_500x40", &gen_sparse(500, 40, 0.10, 7), 10, 1e-9);
        assert_matches_lapack("sparse_200x120", &gen_sparse(200, 120, 0.05, 3), 20, 1e-9);
        assert_matches_lapack(
            "sparse_100x100",
            &gen_sparse(100, 100, 0.0098, 42),
            20,
            1e-8,
        );
    }

    /// Wide inputs must work as well as tall ones.
    #[test]
    fn matches_lapack_on_wide() {
        assert_matches_lapack("wide_50x400", &gen_sparse(50, 400, 0.05, 1234), 10, 1e-9);
    }

    #[test]
    fn orientation_and_reconstruction() {
        for (r, c) in [(200usize, 60usize), (60, 200)] {
            let a = gen_sparse(r, c, 0.1, 11);
            let rank = 10;
            let svd = svd_seed(&a, rank, 42).unwrap();
            assert_eq!(svd.u.dim(), (r, rank), "u shape for {r}x{c}");
            assert_eq!(svd.vt.dim(), (rank, c), "vt shape for {r}x{c}");

            // Rank-`rank` truncation error must match the reference tail exactly:
            // ||A - A_k||_F = sqrt(Σ_{i>k} σ_i²).
            let dense = dense_of(&a);
            let refs = reference_singular_values(&dense);
            let tail: f64 = refs[rank..].iter().map(|v| v * v).sum::<f64>().sqrt();
            let err: f64 = (&svd.recompose() - &dense)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            approx::assert_relative_eq!(err, tail, max_relative = 1e-6);
        }
    }

    /// Singular vectors must be orthonormal and satisfy `A·vᵢ = σᵢ·uᵢ`.
    #[test]
    fn singular_vectors_are_orthonormal_and_consistent() {
        let a = gen_sparse(300, 120, 0.06, 17);
        let rank = 12;
        let svd = svd_seed(&a, rank, 42).unwrap();

        let orth_u = crate::dense::orthogonality_error(&svd.u.view());
        assert!(orth_u < 1e-9, "||UᵀU - I|| = {orth_u:.3e}");
        let vt_t = svd.vt.t().to_owned();
        let orth_v = crate::dense::orthogonality_error(&vt_t.view());
        assert!(orth_v < 1e-9, "||VᵀV - I|| = {orth_v:.3e}");

        // A·vᵢ − σᵢ·uᵢ ≈ 0
        for i in 0..rank {
            let vi: Vec<f64> = svd.vt.row(i).to_vec();
            let mut av = vec![0.0; a.rows()];
            SparseMat::mul_vec(&a, &vi, &mut av, false);
            let resid: f64 = av
                .iter()
                .zip(svd.u.column(i).iter())
                .map(|(&x, &ui)| {
                    let d = x - svd.s[i] * ui;
                    d * d
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                resid / svd.s[0] < 1e-8,
                "triplet {i}: ||A v - s u|| / s_max = {:.3e}",
                resid / svd.s[0]
            );
        }
    }

    #[test]
    fn csr_and_csc_agree() {
        let a = gen_sparse(150, 90, 0.08, 5);
        let csc = a.to_other_storage();
        let x = svd_seed(&a, 12, 42).unwrap();
        let y = svd_seed(&csc, 12, 42).unwrap();
        for (p, q) in x.s.iter().zip(y.s.iter()) {
            approx::assert_relative_eq!(p, q, max_relative = 1e-10);
        }
    }

    /// Mean centering must match an explicitly centered dense reference — this is the
    /// PCA path, and 1.x computed the correction wrongly.
    #[test]
    fn mean_centering_matches_dense_pca() {
        let a = gen_lowrank(120, 40, 6, 31);
        let dense = dense_of(&a);
        let means = dense.mean_axis(Axis(0)).unwrap();
        let centered = &dense - &means.view().insert_axis(Axis(0));
        let want = reference_singular_values(&centered);

        let got = svd_centered(&a, 6, Some(42)).unwrap();
        for (i, &g) in got.s.iter().enumerate() {
            let rel = (g - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(
                rel < 1e-8,
                "centered singular value {i}: {g:.9e} vs {:.9e} (rel {rel:.3e})",
                want[i]
            );
        }
    }

    #[test]
    fn f32_matches_reference_at_f32_precision() {
        let a64 = gen_lowrank(100, 50, 6, 77);
        let want = reference_singular_values(&dense_of(&a64));
        // Same matrix at f32.
        let mut t = TriMatI::<f32, u32>::new((100, 50));
        for (v, (i, j)) in a64.iter() {
            t.add_triplet(i as usize, j as usize, *v as f32);
        }
        let a32: SvdMat<f32> = t.to_csr::<u64>();
        let got = svd_seed(&a32, 6, 42).unwrap();
        for (i, &g) in got.s.iter().enumerate() {
            let rel = ((g as f64) - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(rel < 1e-4, "f32 singular value {i}: rel {rel:.3e}");
        }
    }

    #[test]
    fn reports_convergence_and_bounded_restarts() {
        let a = gen_sparse(200, 100, 0.05, 9);
        let svd = svd_seed(&a, 10, 42).unwrap();
        assert_eq!(svd.diagnostics.algorithm, Algorithm::Irlba);
        match svd.diagnostics.detail {
            Detail::Irlba {
                converged,
                restarts,
                max_residual,
                ..
            } => {
                assert!(converged, "expected convergence");
                assert!(restarts < 50, "unexpectedly many restarts: {restarts}");
                assert!(max_residual >= 0.0);
            }
            ref other => panic!("wrong detail variant: {other:?}"),
        }
        assert!(svd.diagnostics.matvecs > 0);
    }

    /// A tighter tolerance must not produce a worse answer.
    #[test]
    fn tolerance_is_monotone() {
        let a = gen_lowrank(150, 60, 8, 44);
        let want = reference_singular_values(&dense_of(&a));
        let mut prev = f64::INFINITY;
        for tol in [1e-4, 1e-8, 1e-12] {
            let cfg = IrlbaConfig::new(8).seed(42).tol(tol);
            let got = svd_with(&a, &cfg, None).unwrap();
            let err = (0..8)
                .map(|i| (got.s[i] - want[i]).abs() / want[i])
                .fold(0.0f64, f64::max);
            assert!(
                err <= prev * 10.0 + 1e-12,
                "tol {tol:.0e} gave error {err:.3e}, worse than the looser tolerance's {prev:.3e}"
            );
            prev = err.max(1e-16);
        }
    }

    #[test]
    fn rejects_bad_configuration() {
        let a = gen_sparse(50, 30, 0.2, 1);
        assert!(matches!(svd(&a, 0), Err(SvdLibError::InvalidArgument(_))));
        assert!(matches!(svd(&a, 31), Err(SvdLibError::InvalidArgument(_))));
        // mean_center without means.
        let cfg = IrlbaConfig::new(5).mean_center(true);
        assert!(matches!(
            svd_with(&a, &cfg, None),
            Err(SvdLibError::InvalidArgument(_))
        ));
        // means of the wrong length.
        let cfg = IrlbaConfig::new(5).mean_center(true);
        assert!(matches!(
            svd_with(&a, &cfg, Some(Array1::zeros(7))),
            Err(SvdLibError::ShapeMismatch(_))
        ));
    }

    /// The same seed must reproduce bit-identical output, including vector signs.
    #[test]
    fn is_reproducible_given_a_seed() {
        let a = gen_sparse(120, 70, 0.1, 23);
        let x = svd_seed(&a, 8, 1234).unwrap();
        let y = svd_seed(&a, 8, 1234).unwrap();
        assert_eq!(x.s, y.s);
        assert_eq!(x.u, y.u);
        assert_eq!(x.vt, y.vt);
    }

    /// Full rank on a small matrix: every singular value, exactly.
    #[test]
    fn full_rank_request() {
        let a = gen_lowrank(30, 20, 20, 88);
        assert_matches_lapack("full_rank_30x20", &a, 19, 1e-8);
    }
}
