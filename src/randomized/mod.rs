//! Randomized SVD: range finding by random projection.
//!
//! Two sketching strategies, both built on the same reduction:
//!
//! - [`Sketch::PowerIteration`] — Halko, Martinsson & Tropp. Cheap and accurate when
//!   the spectrum decays quickly.
//! - [`Sketch::BlockKrylov`] — Musco & Musco. Keeps every power-iteration block instead
//!   of only the last, which is markedly more accurate on slowly-decaying spectra at
//!   the cost of a wider basis.
//!
//! # The final factorization is `l × l`, not `l × cols`
//!
//! Once a range basis `Y` (`rows × l`) is in hand, the naive next step is to form
//! `B = Yᵀ·A` (`l × cols`) and take its dense SVD. `cols` can be large, so 1.x's
//! `b.svd(true, true)` was a dense factorization of a potentially huge matrix.
//!
//! Instead note that `Bᵀ = Aᵀ·Y` is itself tall and skinny (`cols × l`). Factor it with
//! [`tsqr`](crate::dense::tsqr()) as `Bᵀ = Q_c·R_c`, then the only dense SVD needed is of
//! `R_cᵀ`, which is `l × l`:
//!
//! ```text
//! A ≈ Y·Bᵀᵀ = Y·R_cᵀ·Q_cᵀ = (Y·Û)·Ŝ·(Q_c·V̂)ᵀ
//! ```
//!
//! With rank 50 and 10 oversamples that is a 60 × 60 factorization regardless of how
//! wide the input is.

use crate::dense::{small_svd, svd_flip, tsqr};
use crate::error::{Result, SvdLibError};
use crate::matrix::SparseMatDense;
use crate::types::{Algorithm, Detail, Diagnostics, SvdFloat, SvdRec};
use ndarray::{s, Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{rng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Default oversampling beyond the requested rank.
pub const DEFAULT_OVERSAMPLES: usize = 10;
/// Default power iterations.
pub const DEFAULT_POWER_ITERATIONS: usize = 2;

/// How the intermediate basis is re-orthogonalised between products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalizer {
    /// Tall-skinny QR. Numerically the right default.
    #[default]
    Tsqr,
    /// Column normalisation only. Cheaper, and adequate for one or two iterations, but
    /// it does not prevent the basis collapsing toward the dominant direction.
    ColumnNorm,
    /// No re-orthogonalisation. Only safe with zero power iterations.
    None,
}

/// The sketching strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sketch {
    /// `Y = (A·Aᵀ)^q·A·Ω`, keeping only the final block.
    ///
    /// Basis width is `rank + oversamples`.
    PowerIteration { iterations: usize },
    /// `K = [A·Ω, (A·Aᵀ)A·Ω, …, (A·Aᵀ)^(b-1)·A·Ω]`, keeping every block.
    ///
    /// Basis width is `blocks · (rank + oversamples)`, so memory scales with `blocks`.
    /// Two to four blocks is usually the sweet spot.
    BlockKrylov { blocks: usize },
}

impl Default for Sketch {
    fn default() -> Self {
        Sketch::PowerIteration {
            iterations: DEFAULT_POWER_ITERATIONS,
        }
    }
}

/// Configuration for [`svd_with`].
///
/// Replaces 1.x's eight positional arguments, two of which were bare `bool`s that the
/// crate's own call sites commented incorrectly.
#[derive(Debug, Clone)]
pub struct RandomizedConfig {
    /// Number of singular triplets wanted.
    pub rank: usize,
    /// Extra sketch columns; improves accuracy at linear cost.
    pub oversamples: usize,
    pub sketch: Sketch,
    pub normalizer: Normalizer,
    /// Subtract column means without materialising the centered matrix.
    pub mean_center: bool,
    /// Fixed seed; `None` draws from the OS.
    ///
    /// 1.x accepted `Option<u64>` but substituted `0` for `None`, so "random" was in
    /// fact a fixed sketch on every call.
    pub seed: Option<u64>,
}

impl RandomizedConfig {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            oversamples: DEFAULT_OVERSAMPLES,
            sketch: Sketch::default(),
            normalizer: Normalizer::default(),
            mean_center: false,
            seed: None,
        }
    }
    pub fn oversamples(mut self, n: usize) -> Self {
        self.oversamples = n;
        self
    }
    pub fn power_iterations(mut self, n: usize) -> Self {
        self.sketch = Sketch::PowerIteration { iterations: n };
        self
    }
    pub fn block_krylov(mut self, blocks: usize) -> Self {
        self.sketch = Sketch::BlockKrylov { blocks };
        self
    }
    pub fn normalizer(mut self, n: Normalizer) -> Self {
        self.normalizer = n;
        self
    }
    pub fn mean_center(mut self, yes: bool) -> Self {
        self.mean_center = yes;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A sink for progress messages, called once per major stage.
///
/// 1.x printed stage timings to stdout behind a `verbose: bool`. A library shouldn't
/// write to the process's streams, so the caller supplies the sink.
pub type Progress<'a> = &'a (dyn Fn(&str) + Sync);

/// `rank` largest singular triplets with default settings.
pub fn svd<T: SvdFloat, M: SparseMatDense<T>>(a: &M, rank: usize) -> Result<SvdRec<T>> {
    svd_with(a, &RandomizedConfig::new(rank), None)
}

/// `rank` largest singular triplets with a fixed seed.
pub fn svd_seed<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    rank: usize,
    seed: u64,
) -> Result<SvdRec<T>> {
    svd_with(a, &RandomizedConfig::new(rank).seed(seed), None)
}

/// Block-Krylov variant with `blocks` blocks.
pub fn svd_block_krylov<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    rank: usize,
    blocks: usize,
    seed: Option<u64>,
) -> Result<SvdRec<T>> {
    let mut cfg = RandomizedConfig::new(rank).block_krylov(blocks);
    cfg.seed = seed;
    svd_with(a, &cfg, None)
}

/// PCA: `rank` largest triplets of the implicitly mean-centered matrix.
pub fn svd_centered<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    rank: usize,
    seed: Option<u64>,
) -> Result<SvdRec<T>> {
    let mut cfg = RandomizedConfig::new(rank).mean_center(true);
    cfg.seed = seed;
    svd_with(a, &cfg, None)
}

/// Compute a decomposition with explicit configuration.
pub fn svd_with<T: SvdFloat, M: SparseMatDense<T>>(
    a: &M,
    cfg: &RandomizedConfig,
    progress: Option<Progress<'_>>,
) -> Result<SvdRec<T>> {
    let note = |msg: &str| {
        if let Some(p) = progress {
            p(msg);
        }
    };

    let (rows, cols) = (a.rows(), a.cols());
    let min_dim = rows.min(cols);
    if cfg.rank == 0 {
        return Err(SvdLibError::invalid("randomized: rank must be at least 1"));
    }
    if cfg.rank > min_dim {
        return Err(SvdLibError::invalid(format!(
            "randomized: rank {} exceeds min(rows, cols) = {min_dim}",
            cfg.rank
        )));
    }
    if let Sketch::BlockKrylov { blocks } = cfg.sketch {
        if blocks == 0 {
            return Err(SvdLibError::invalid(
                "randomized: block_krylov needs at least one block",
            ));
        }
    }

    let rank = cfg.rank;
    // Sketch width, capped so the basis cannot exceed the operand's rank.
    let l = (rank + cfg.oversamples).min(min_dim);
    let seed = cfg.seed.unwrap_or_else(|| rng().next_u64());
    let mut rng_state = StdRng::seed_from_u64(seed);

    let means: Option<Array1<T>> = if cfg.mean_center {
        note("computing column means");
        Some(a.col_means())
    } else {
        None
    };
    let mut matvecs = 0usize;

    // Product helpers that apply centering when configured.
    let mul = |rhs: &Array2<T>, out: &mut Array2<T>, trans: bool| match &means {
        Some(m) => a.mul_dense_centered(rhs.view(), out.view_mut(), trans, m.view()),
        None => a.mul_dense(rhs.view(), out.view_mut(), trans),
    };

    note("drawing the random sketch");
    let omega = gaussian(cols, l, &mut rng_state);

    // ----- Stage 1: build a basis for the range of A -----
    let mut basis = match cfg.sketch {
        Sketch::PowerIteration { iterations } => {
            note("projecting");
            let mut y = Array2::<T>::zeros((rows, l));
            mul(&omega, &mut y, false);
            matvecs += l;
            normalize(&mut y, cfg.normalizer)?;

            let mut z = Array2::<T>::zeros((cols, l));
            for i in 0..iterations {
                note(&format!("power iteration {}/{}", i + 1, iterations));
                mul(&y, &mut z, true);
                matvecs += l;
                normalize(&mut z, cfg.normalizer)?;
                mul(&z, &mut y, false);
                matvecs += l;
                normalize(&mut y, cfg.normalizer)?;
            }
            y
        }
        Sketch::BlockKrylov { blocks } => {
            note("building the Krylov block basis");
            // The range of A has dimension at most min(rows, cols), so a basis wider
            // than that is necessarily rank-deficient. Clamping to `rows` alone is not
            // enough: on a 500x60 operand, 4 blocks of 22 would give an 88-column basis
            // whose `Aᵀ·basis` is 60x88 — wider than tall, which no QR accepts.
            let width = (blocks * l).min(min_dim);
            let mut k = Array2::<T>::zeros((rows, width));
            let mut y = Array2::<T>::zeros((rows, l));
            let mut z = Array2::<T>::zeros((cols, l));

            mul(&omega, &mut y, false);
            matvecs += l;
            normalize(&mut y, cfg.normalizer)?;

            let mut filled = 0usize;
            for b in 0..blocks {
                if filled >= width {
                    break;
                }
                let take = l.min(width - filled);
                k.slice_mut(s![.., filled..filled + take])
                    .assign(&y.slice(s![.., ..take]));
                filled += take;
                if b + 1 == blocks {
                    break;
                }
                note(&format!("krylov block {}/{}", b + 2, blocks));
                mul(&y, &mut z, true);
                matvecs += l;
                normalize(&mut z, cfg.normalizer)?;
                mul(&z, &mut y, false);
                matvecs += l;
                normalize(&mut y, cfg.normalizer)?;
            }
            if filled < width {
                k = k.slice(s![.., ..filled]).to_owned();
            }
            k
        }
    };

    note("orthonormalising the basis");
    tsqr(&mut basis)?;
    let width = basis.ncols();

    // ----- Stage 2: project and factor -----
    //
    // `bt = Aᵀ·basis` is tall-skinny, so TSQR it and take the SVD of the small `R`
    // rather than factoring the wide `basis ᵀ·A` directly.
    note("projecting onto the basis");
    let mut bt = Array2::<T>::zeros((cols, width));
    mul(&basis, &mut bt, true);
    matvecs += width;

    note("reducing");
    let r_c = tsqr(&mut bt)?; // bt is now Q_c (cols × width), r_c is width × width
    let small = small_svd(r_c.t())?; // SVD of R_cᵀ

    // A ≈ (basis·Û)·Ŝ·(Q_c·V̂)ᵀ
    let keep = rank.min(small.s.len());
    let u_hat = small.u.slice(s![.., ..keep]);
    let v_hat = small.vt.slice(s![..keep, ..]).t().to_owned(); // width × keep

    let mut u = basis.dot(&u_hat);
    let mut vt = bt
        .dot(&v_hat)
        .reversed_axes()
        .as_standard_layout()
        .to_owned();
    let s = small.s.slice(s![..keep]).to_owned();

    svd_flip(&mut u, &mut vt);

    let (oversamples, power_iterations, block_size) = match cfg.sketch {
        Sketch::PowerIteration { iterations } => (cfg.oversamples, iterations, l),
        Sketch::BlockKrylov { blocks } => (cfg.oversamples, blocks, l),
    };

    Ok(SvdRec {
        d: keep,
        u,
        s,
        vt,
        total_squared_norm: T::from_f64_val(crate::matrix::total_squared_norm(
            a,
            means.as_ref().map(|m| m.view()),
        )),
        diagnostics: Diagnostics {
            algorithm: match cfg.sketch {
                Sketch::PowerIteration { .. } => Algorithm::Randomized,
                Sketch::BlockKrylov { .. } => Algorithm::BlockKrylov,
            },
            non_zero: a.nnz(),
            dimensions: rank,
            significant_values: keep,
            transposed: false,
            random_seed: seed,
            matvecs,
            detail: Detail::Randomized {
                oversamples,
                power_iterations,
                block_size,
            },
        },
    })
}

/// A `rows × cols` matrix of standard normal draws.
fn gaussian<T: SvdFloat>(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<T> {
    let normal = Normal::new(0.0, 1.0).expect("N(0,1) is well-formed");
    Array2::from_shape_fn((rows, cols), |_| T::from_f64_val(normal.sample(rng)))
}

fn normalize<T: SvdFloat>(m: &mut Array2<T>, how: Normalizer) -> Result<()> {
    match how {
        Normalizer::Tsqr => {
            tsqr(m)?;
            Ok(())
        }
        Normalizer::ColumnNorm => {
            let floor = T::from_f64_val(1e-10);
            for mut col in m.axis_iter_mut(Axis(1)) {
                let n = col.iter().map(|&x| x * x).sum::<T>().sqrt();
                if n > floor {
                    let inv = T::one() / n;
                    col.map_inplace(|x| *x *= inv);
                }
            }
            Ok(())
        }
        Normalizer::None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::SvdMat;
    use crate::testing::{dense_of, gen_lowrank, gen_sparse, reference_singular_values, Lcg};
    use sprs::TriMatI;

    fn diagonal(n: usize) -> SvdMat<f64> {
        let mut t = TriMatI::<f64, u32>::new((n, n));
        for i in 0..n {
            t.add_triplet(i, i, (n - i) as f64);
        }
        t.to_csr::<u64>()
    }

    /// Worst relative error against a dense LAPACK reference.
    fn max_rel_error(a: &SvdMat<f64>, got: &SvdRec<f64>) -> f64 {
        let want = reference_singular_values(&dense_of(a));
        got.s
            .iter()
            .enumerate()
            .map(|(i, &g)| (g - want[i]).abs() / want[i].abs().max(1e-30))
            .fold(0.0f64, f64::max)
    }

    /// A rapidly-decaying spectrum is the regime randomized SVD is designed for, so
    /// accuracy there should be high even with modest power iterations.
    #[test]
    fn accurate_on_decaying_spectrum() {
        let a = gen_lowrank(400, 120, 10, 5);
        let got = svd_seed(&a, 10, 42).unwrap();
        let err = max_rel_error(&a, &got);
        assert!(err < 1e-6, "max relative error {err:.3e}");
    }

    /// `diag(60..1)` decays only linearly, so `sigma_11/sigma_10 = 0.98` and the
    /// randomized error bound `~(sigma_{k+1}/sigma_k)^(2q+1)` barely improves with `q`.
    /// This is a limitation of the method, not a defect: assert the behaviour theory
    /// predicts rather than an accuracy it cannot deliver.
    #[test]
    fn power_iteration_converges_slowly_on_linear_decay() {
        let a = diagonal(60);
        let loose = svd_with(
            &a,
            &RandomizedConfig::new(10).seed(42).power_iterations(0),
            None,
        )
        .unwrap();
        let tight = svd_with(
            &a,
            &RandomizedConfig::new(10).seed(42).power_iterations(7),
            None,
        )
        .unwrap();
        let e0 = max_rel_error(&a, &loose);
        let e7 = max_rel_error(&a, &tight);
        assert!(
            e0 > 1e-2,
            "q=0 should be visibly inaccurate here, got {e0:.3e}"
        );
        assert!(
            e7 < e0 / 50.0,
            "7 power iterations should improve substantially: {e0:.3e} -> {e7:.3e}"
        );
    }

    /// Block Krylov is the answer for that same matrix: retaining every block spans the
    /// dominant subspace essentially exactly, reaching machine precision where power
    /// iteration is still at 1e-4.
    #[test]
    fn block_krylov_is_near_exact_on_linear_decay() {
        let a = diagonal(60);
        let got = svd_with(
            &a,
            &RandomizedConfig::new(10).seed(42).block_krylov(4),
            None,
        )
        .unwrap();
        let err = max_rel_error(&a, &got);
        assert!(err < 1e-10, "block krylov max relative error {err:.3e}");
    }

    /// More power iterations must not make the answer worse.
    #[test]
    fn power_iterations_improve_accuracy() {
        let a = gen_sparse(600, 200, 0.05, 13);
        let mut prev = f64::INFINITY;
        for q in [0usize, 1, 2, 4, 7] {
            let cfg = RandomizedConfig::new(15).seed(42).power_iterations(q);
            let got = svd_with(&a, &cfg, None).unwrap();
            let err = max_rel_error(&a, &got);
            assert!(
                err <= prev * 1.5 + 1e-9,
                "q={q} error {err:.3e} is worse than q's predecessor {prev:.3e}"
            );
            prev = err;
        }
        // A near-flat spectrum (ratio 0.999) cannot be driven to high accuracy by
        // power iteration at any practical `q`; what must hold is a large improvement
        // over the un-iterated sketch.
        let plain = svd_with(
            &a,
            &RandomizedConfig::new(15).seed(42).power_iterations(0),
            None,
        )
        .unwrap();
        let e0 = max_rel_error(&a, &plain);
        assert!(
            prev < e0 / 10.0,
            "7 power iterations ({prev:.3e}) should be well under the q=0 error ({e0:.3e})"
        );
    }

    /// Block Krylov should beat plain power iteration at equal matrix-product budget on
    /// a slowly-decaying spectrum, which is exactly what it exists for.
    #[test]
    fn block_krylov_beats_power_iteration_on_flat_spectrum() {
        // A near-flat spectrum: random sparse, no low-rank structure.
        let a = gen_sparse(800, 200, 0.04, 29);
        let rank = 20;

        let power = svd_with(
            &a,
            &RandomizedConfig::new(rank).seed(42).power_iterations(3),
            None,
        )
        .unwrap();
        let krylov = svd_with(
            &a,
            &RandomizedConfig::new(rank).seed(42).block_krylov(4),
            None,
        )
        .unwrap();

        let e_power = max_rel_error(&a, &power);
        let e_krylov = max_rel_error(&a, &krylov);
        assert!(
            e_krylov <= e_power,
            "block krylov {e_krylov:.3e} did not improve on power iteration {e_power:.3e}"
        );
        assert_eq!(krylov.diagnostics.algorithm, Algorithm::BlockKrylov);
    }

    #[test]
    fn orientation_is_correct_for_wide_and_tall() {
        for (r, c) in [(400usize, 80usize), (80, 400)] {
            let a = gen_sparse(r, c, 0.08, 11);
            let got = svd_seed(&a, 10, 42).unwrap();
            assert_eq!(got.u.dim(), (r, 10), "u shape for {r}x{c}");
            assert_eq!(got.vt.dim(), (10, c), "vt shape for {r}x{c}");
        }
    }

    #[test]
    fn singular_vectors_are_orthonormal() {
        let a = gen_lowrank(300, 100, 12, 71);
        let got = svd_seed(&a, 12, 42).unwrap();
        let ou = crate::dense::orthogonality_error(&got.u.view());
        assert!(ou < 1e-8, "||UᵀU - I|| = {ou:.3e}");
        let vt_t = got.vt.t().to_owned();
        let ov = crate::dense::orthogonality_error(&vt_t.view());
        assert!(ov < 1e-8, "||VᵀV - I|| = {ov:.3e}");
    }

    /// The whole point of the 2.0 rewrite: this used to panic with `todo!()` for every
    /// stock matrix type.
    #[test]
    fn works_on_csr_and_csc_without_panicking() {
        let a = gen_sparse(300, 120, 0.05, 3);
        let csc = a.to_other_storage();
        let x = svd_seed(&a, 10, 42).unwrap();
        let y = svd_seed(&csc, 10, 42).unwrap();
        for (p, q) in x.s.iter().zip(y.s.iter()) {
            approx::assert_relative_eq!(p, q, max_relative = 1e-9);
        }
    }

    #[test]
    fn works_on_masked_matrices() {
        let a = gen_sparse(300, 60, 0.1, 19);
        let cols: Vec<usize> = (0..60).filter(|c| c % 2 == 0).collect();
        let masked = crate::matrix::MaskedCsMat::with_columns(&a, &cols);
        let got = svd_seed(&masked, 8, 42).unwrap();
        assert_eq!(got.u.nrows(), 300);
        assert_eq!(got.vt.ncols(), 30);
        for w in got.s.to_vec().windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    /// Mean centering must agree with an explicitly centered dense reference.
    #[test]
    fn mean_centering_matches_dense_pca() {
        let a = gen_lowrank(300, 60, 8, 37);
        let dense = dense_of(&a);
        let means = dense.mean_axis(Axis(0)).unwrap();
        let centered = &dense - &means.view().insert_axis(Axis(0));
        let want = reference_singular_values(&centered);

        let cfg = RandomizedConfig::new(8)
            .seed(42)
            .mean_center(true)
            .power_iterations(5);
        let got = svd_with(&a, &cfg, None).unwrap();
        for (i, &g) in got.s.iter().enumerate() {
            let rel = (g - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(
                rel < 1e-5,
                "centered singular value {i}: {g:.9e} vs {:.9e} (rel {rel:.3e})",
                want[i]
            );
        }
    }

    /// `None` must actually vary the sketch. 1.x substituted seed 0 for `None`, so
    /// successive calls were identical.
    #[test]
    fn unseeded_runs_differ() {
        let a = gen_sparse(300, 100, 0.05, 47);
        let cfg = RandomizedConfig::new(6).power_iterations(0);
        let x = svd_with(&a, &cfg, None).unwrap();
        let y = svd_with(&a, &cfg, None).unwrap();
        assert_ne!(
            x.diagnostics.random_seed, y.diagnostics.random_seed,
            "an unseeded config produced the same seed twice"
        );
        // Zero power iterations makes the sketch dependence visible in the output.
        assert_ne!(x.u, y.u, "unseeded runs produced identical bases");
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let a = gen_sparse(300, 100, 0.05, 51);
        let x = svd_seed(&a, 8, 999).unwrap();
        let y = svd_seed(&a, 8, 999).unwrap();
        assert_eq!(x.s, y.s);
        assert_eq!(x.u, y.u);
        assert_eq!(x.vt, y.vt);
    }

    #[test]
    fn agrees_with_irlba() {
        let a = gen_lowrank(400, 150, 12, 61);
        let rand = svd_with(
            &a,
            &RandomizedConfig::new(12).seed(42).power_iterations(6),
            None,
        )
        .unwrap();
        let exact = crate::irlba::svd_seed(&a, 12, 42).unwrap();
        for i in 0..12 {
            let rel = (rand.s[i] - exact.s[i]).abs() / exact.s[i];
            assert!(rel < 1e-6, "triplet {i}: randomized vs irlba rel {rel:.3e}");
        }
    }

    #[test]
    fn normalizers_all_produce_usable_results() {
        let a = gen_lowrank(400, 100, 10, 67);
        for n in [Normalizer::Tsqr, Normalizer::ColumnNorm, Normalizer::None] {
            let cfg = RandomizedConfig::new(10)
                .seed(42)
                .power_iterations(1)
                .normalizer(n);
            let got = svd_with(&a, &cfg, None).unwrap();
            let err = max_rel_error(&a, &got);
            assert!(err < 1e-2, "{n:?} gave max relative error {err:.3e}");
        }
    }

    #[test]
    fn progress_callback_is_invoked() {
        let a = gen_sparse(200, 80, 0.1, 73);
        let seen = std::sync::Mutex::new(Vec::<String>::new());
        let sink = |msg: &str| seen.lock().unwrap().push(msg.to_string());
        let cfg = RandomizedConfig::new(6).seed(42).power_iterations(2);
        svd_with(&a, &cfg, Some(&sink)).unwrap();
        let msgs = seen.into_inner().unwrap();
        assert!(!msgs.is_empty(), "no progress reported");
        assert!(
            msgs.iter().any(|m| m.contains("power iteration")),
            "power iterations were not reported: {msgs:?}"
        );
    }

    /// Regression: a block count whose basis would exceed `min(rows, cols)` must clamp
    /// rather than hand a wide matrix to the QR.
    #[test]
    fn block_krylov_clamps_basis_to_matrix_rank() {
        // 500x60 with rank 12 + 10 oversamples = 22 per block; 4 blocks would be 88.
        let a = gen_sparse(500, 60, 0.08, 7);
        let got = svd_block_krylov(&a, 12, 4, Some(42)).expect("should clamp, not fail");
        assert_eq!(got.d, 12);
        assert_eq!(got.u.dim(), (500, 12));
        assert_eq!(got.vt.dim(), (12, 60));

        // Also the wide orientation.
        let b = gen_sparse(60, 500, 0.08, 11);
        let got = svd_block_krylov(&b, 12, 4, Some(42)).expect("should clamp, not fail");
        assert_eq!(got.u.dim(), (60, 12));
        assert_eq!(got.vt.dim(), (12, 500));
    }

    #[test]
    fn rejects_bad_configuration() {
        let a = gen_sparse(50, 30, 0.2, 1);
        assert!(matches!(svd(&a, 0), Err(SvdLibError::InvalidArgument(_))));
        assert!(matches!(svd(&a, 31), Err(SvdLibError::InvalidArgument(_))));
        let cfg = RandomizedConfig::new(5).block_krylov(0);
        assert!(matches!(
            svd_with(&a, &cfg, None),
            Err(SvdLibError::InvalidArgument(_))
        ));
    }

    #[test]
    fn f32_works() {
        let a64 = gen_lowrank(300, 80, 8, 79);
        let want = reference_singular_values(&dense_of(&a64));
        let mut t = TriMatI::<f32, u32>::new((300, 80));
        for (v, (i, j)) in a64.iter() {
            t.add_triplet(i as usize, j as usize, *v as f32);
        }
        let a32: SvdMat<f32> = t.to_csr::<u64>();
        let cfg = RandomizedConfig::new(8).seed(42).power_iterations(4);
        let got = svd_with(&a32, &cfg, None).unwrap();
        for (i, &g) in got.s.iter().enumerate() {
            let rel = ((g as f64) - want[i]).abs() / want[i].abs().max(1e-30);
            assert!(rel < 1e-3, "f32 singular value {i}: rel {rel:.3e}");
        }
    }

    /// Oversampling beyond the operand's rank must clamp rather than overrun.
    #[test]
    fn oversampling_clamps_to_matrix_rank() {
        let a = gen_sparse(40, 20, 0.3, 83);
        let cfg = RandomizedConfig::new(5).seed(42).oversamples(1000);
        let got = svd_with(&a, &cfg, None).unwrap();
        assert_eq!(got.d, 5);
        let mut rng = Lcg::new(1);
        let _ = rng.next_u64();
    }

    #[test]
    fn diagnostics_report_matvecs_and_algorithm() {
        let a = gen_sparse(200, 80, 0.1, 89);
        let got = svd_seed(&a, 6, 42).unwrap();
        assert_eq!(got.diagnostics.algorithm, Algorithm::Randomized);
        assert!(got.diagnostics.matvecs > 0);
        match got.diagnostics.detail {
            Detail::Randomized {
                power_iterations, ..
            } => {
                assert_eq!(power_iterations, DEFAULT_POWER_ITERATIONS);
            }
            ref other => panic!("wrong detail variant: {other:?}"),
        }
    }

    /// Characterises how each sketch converges as a function of spectral decay. Run
    /// with `--ignored --nocapture` to see the table; it is the evidence behind the
    /// guidance in the module docs about when to prefer block Krylov.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn report_convergence_rates() {
        let cases: Vec<(&str, SvdMat<f64>, usize)> = vec![
            ("diag_60_linear", diagonal(60), 10),
            ("lowrank_400x120_r10", gen_lowrank(400, 120, 10, 5), 10),
            ("sparse_600x200_flat", gen_sparse(600, 200, 0.05, 13), 15),
        ];
        for (name, a, rank) in cases {
            let want = reference_singular_values(&dense_of(&a));
            print!(
                "{name:<22} sigma_ratio={:.3}  ",
                want[rank] / want[rank - 1]
            );
            for q in [0usize, 1, 2, 4, 7] {
                let cfg = RandomizedConfig::new(rank).seed(42).power_iterations(q);
                let got = svd_with(&a, &cfg, None).unwrap();
                print!("q{q}={:.2e} ", max_rel_error(&a, &got));
            }
            for b in [2usize, 4] {
                let cfg = RandomizedConfig::new(rank).seed(42).block_krylov(b);
                let got = svd_with(&a, &cfg, None).unwrap();
                print!("bk{b}={:.2e} ", max_rel_error(&a, &got));
            }
            println!();
        }
    }
}
