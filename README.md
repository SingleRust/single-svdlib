# single-svdlib

[![Crate](https://img.shields.io/crates/v/single-svdlib.svg)](https://crates.io/crates/single-svdlib)
[![Documentation](https://docs.rs/single-svdlib/badge.svg)](https://docs.rs/single-svdlib)

Sparse SVD and PCA over [`sprs`](https://crates.io/crates/sprs) matrices. Built for
matrices too big to densify — you can mask columns and mean-center without copying or
touching the original.

```toml
[dependencies]
single-svdlib = "2.0"
```

## Quick start

```rust
use single_svdlib::{sprs::TriMatI, SvdMat};

let mut tri = TriMatI::<f64, u32>::new((4, 3));
tri.add_triplet(0, 0, 1.0);
tri.add_triplet(1, 1, 2.0);
tri.add_triplet(2, 2, 3.0);
tri.add_triplet(3, 0, 4.0);
let a: SvdMat<f64> = tri.to_csr::<u64>();

let svd = single_svdlib::svd(&a, 2)?;

// A ≈ u · diag(s) · vt
assert_eq!(svd.u.dim(),  (4, 2));  // left vectors are columns
assert_eq!(svd.vt.dim(), (2, 3));  // right vectors are rows
# Ok::<(), single_svdlib::SvdLibError>(())
```

## Picking a solver

| module | method | use when |
|---|---|---|
| `irlba` | thick-restarted Lanczos bidiagonalization | **default.** Accurate, memory bounded by the rank you ask for |
| `randomized` | range finder, power iteration or block Krylov | huge inputs where an approximation is fine |
| `lanczos` | LAS2 from SVDLIBC | **deprecated, don't use.** Behind the `las2` feature, see [below](#las2-is-deprecated) |

`single_svdlib::svd` calls `irlba`.

```rust
use single_svdlib::{irlba, randomized};
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((40, 20));
# for i in 0..40 { for j in 0..20 { t.add_triplet(i, j, ((i * 7 + j * 3) % 11) as f64); } }
# let a: SvdMat<f64> = t.to_csr::<u64>();

let exact = irlba::svd_seed(&a, 10, 42)?;            // reproducible
let pca = irlba::svd_centered(&a, 10, Some(42))?;    // mean-centered, no densifying
let approx = randomized::svd_seed(&a, 10, 42)?;      // faster, less accurate
let better = randomized::svd_block_krylov(&a, 10, 3, Some(42))?;  // slow spectra
# Ok::<(), single_svdlib::SvdLibError>(())
```

`IrlbaConfig` and `RandomizedConfig` give you the knobs:

```rust
use single_svdlib::randomized::{svd_with, Normalizer, RandomizedConfig};
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((40, 20));
# for i in 0..40 { for j in 0..20 { t.add_triplet(i, j, ((i * 5 + j) % 7) as f64); } }
# let a: SvdMat<f64> = t.to_csr::<u64>();

let cfg = RandomizedConfig::new(10)
    .oversamples(15)
    .power_iterations(4)
    .normalizer(Normalizer::Tsqr)
    .mean_center(true)
    .seed(42);

// Last argument is an optional progress callback, one call per stage.
let svd = svd_with(&a, &cfg, Some(&|stage: &str| eprintln!("{stage}")))?;
# Ok::<(), single_svdlib::SvdLibError>(())
```

## Memory

`SvdMat<T>` is `CsMatI<T, u32, u64>` — 32-bit column indices, 64-bit row pointers. That's
12 bytes per non-zero for `f64` instead of the 16 you get with `usize` everywhere, and
8 instead of 16 for `f32`. The wider pointer still handles more than `u32::MAX`
non-zeros. Need bigger indices? Name them: `SvdMat<f64, u64, u64>`.

`irlba` restarts, so it holds `work + 1` right vectors and `work` left vectors
(`work = rank + 7` by default) no matter how many restarts it takes. Peak is known up
front:

```text
(work + 1) · cols + work · rows   scalars
```

200 000 × 30 000, 4.9M non-zeros, rank 50, 10 cores:

| | time | peak RSS |
|---|---|---|
| `irlba` | 2.2 s | 514 MiB |
| `randomized`, 2 power iterations | 1.2 s | 654 MiB |
| `randomized`, block Krylov ×3 | 4.0 s | 1080 MiB |

The matrix is 57.8 MiB, against 76.6 MiB with `usize` indices. Reproduce with
`cargo run --release --example scale irlba`.

Block Krylov's basis is `blocks × (rank + oversamples)` wide, so its memory grows with
`blocks`. That's what its accuracy costs.

## Big matrices, column subsets, PCA

The thing this crate exists for:

```rust
use single_svdlib::{irlba, MaskedCsMat};
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((200, 60));
# for i in 0..200 { for j in 0..60 { t.add_triplet(i, j, ((i * 7 + j) % 13) as f64); } }
# let counts: SvdMat<f64> = t.to_csr::<u64>();
# let selected_genes: Vec<usize> = (0..60).step_by(3).collect();

// A view over the selected columns. No copy, `counts` untouched.
let view = MaskedCsMat::with_columns(&counts, &selected_genes);

let pca = irlba::svd_centered(&view, 10, Some(42))?;
// pca.u  — scores,   cells x components
// pca.vt — loadings, components x selected genes
# Ok::<(), single_svdlib::SvdLibError>(())
```

1 000 000 cells × 30 000 genes, 145M non-zeros, masked to 2238 genes, 50 components
(`cargo run --release --example pca_at_scale`):

| | |
|---|---|
| matrix, sparse | 1.63 GiB |
| same matrix dense | 223.52 GiB — **137× bigger**, never materialised |
| building the view | 5 ms, no copy |
| PCA on the view | 44 s, converged |
| peak RSS | ~3.9 GiB |
| `‖A_c·vᵢ − σᵢ·uᵢ‖ / σ_max` | 1.8e-15 |

Three things to know:

**A mask doesn't make products cheaper.** Every product still walks all the source's
non-zeros and checks each against the mask — only the output gets narrower. With a
restrictive mask and hundreds of products, extract once instead. It stays sparse:

```rust
# use single_svdlib::{irlba, MaskedCsMat, sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((200, 60));
# for i in 0..200 { for j in 0..60 { t.add_triplet(i, j, ((i * 7 + j) % 13) as f64); } }
# let counts: SvdMat<f64> = t.to_csr::<u64>();
# let selected_genes: Vec<usize> = (0..60).step_by(3).collect();
let sub = MaskedCsMat::with_columns(&counts, &selected_genes).to_sparse();
let pca = irlba::svd_centered(&sub, 10, Some(42))?;
# Ok::<(), single_svdlib::SvdLibError>(())
```

At 1M × 30k that copy took 146 ms and made the PCA 18% faster (44 s → 36 s).

**An unconverged result is an error, not a value.** `irlba` won't hand back triplets that
missed `tol` — too easy to skip the diagnostics and carry a bad decomposition downstream.
The error tells you the residual and what to change. If you really want a best effort,
call `.allow_unconverged()` and check `converged()` / `max_residual()` yourself.

**Raise `work` on tall matrices.** With many rows it's re-orthogonalization that
dominates, not the sparse products. `rank + 30` was 2.2× faster than the default at the
same accuracy. `cargo run --release --example tune` prints the comparison for your shape.

## Result

```rust,ignore
pub struct SvdRec<T> {
    pub d: usize,                 // triplets returned
    pub u: Array2<T>,             // m × d, left vectors are columns
    pub s: Array1<T>,             // d, descending
    pub vt: Array2<T>,            // d × n, right vectors are rows
    pub total_squared_norm: T,    // ‖A‖²_F of what was decomposed
    pub diagnostics: Diagnostics<T>,
}
```

`A ≈ u · diag(s) · vt`, same layout as `numpy.linalg.svd`. `Diagnostics::matvecs` counts
sparse products (a block product against `k` columns counts as `k`), which is comparable
across solvers. `Diagnostics::detail` has per-algorithm stuff like `irlba`'s restart count.

### Explained variance

```rust
# use single_svdlib::{irlba, sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((200, 60));
# for i in 0..200 { for j in 0..60 { t.add_triplet(i, j, ((i * 7 + j) % 13) as f64); } }
# let a: SvdMat<f64> = t.to_csr::<u64>();
let pca = irlba::svd_centered(&a, 10, Some(42))?;

let scores = pca.scores();                    // m × d, u · diag(s)
let var    = pca.explained_variance();        // sᵢ² / (m − 1), sklearn's convention
let ratio  = pca.explained_variance_ratio();  // fraction of total, sums to ≤ 1

assert!(ratio.iter().sum::<f64>() <= 1.0 + 1e-9);
# Ok::<(), single_svdlib::SvdLibError>(())
```

The ratio needs a denominator a truncated SVD can't give you — the tail you threw away —
so every solver makes one extra pass over the non-zeros to record `total_squared_norm`.
That's nothing next to the hundreds of products a solve does.

If the solver was centering, that norm is of the centered matrix, summed per entry so it
stays accurate on columns with a large offset. On a `MaskedCsMat` it covers only the
selected entries, so a masked PCA's ratios are relative to the submatrix you decomposed.

## Accuracy

The small reduced factors (IRLBA's `B`, randomized's `R`) decide the accuracy of the whole
result, so they're factored with a **one-sided Jacobi SVD** rather than bidiagonal QR.
Jacobi is accurate to the condition number *after* column scaling (Demmel & Veselić). On a
`5 × 2` operand with `κ ≈ 8·10⁶`, `nalgebra`'s Golub–Reinsch reconstructed to `1.3·10⁻⁹`;
Jacobi hits `< 1·10⁻¹⁵`. It's also why there's no linear-algebra backend in the dependency
tree — `sprs`, `ndarray`, `rayon`, `num-traits`, `rand` and `thiserror` is the lot.

Three test suites:

- **`tests/cross_algorithm.rs`** — every solver against a dense LAPACK-grade reference.
- **`tests/robustness.rs`** — degenerate and hostile inputs (all-zero, rank-deficient,
  `NaN`, `∞`, huge dynamic range, empty masks) must give a correct answer or a typed
  error. Never a panic, a hang, or a quietly wrong result.
- **`tests/properties.rs`** — invariants over shapes and distributions `proptest` picks:
  agreement with the reference, orthonormality, `A·vᵢ = σᵢ·uᵢ`, truncation error equal to
  the spectral tail, storage-order and index-width invariance, reproducibility, and the
  same set again in `f32`. Push harder with
  `PROPTEST_CASES=100000 cargo test --release --test properties`.

### Storage order

A compressed matrix can only be walked along its outer dimension, so the two product
directions differ. `A · D` on CSR writes output row `i` from sparse row `i` — threads own
disjoint rows, no scratch, no reduction. `Aᵀ · D` scatters, so threads collide and have to
accumulate. `transpose_view()` doesn't get you out of this; it relabels CSR as a CSC view
without changing what's traversable. What it does mean is that a CSC-stored matrix gets
the cheap kernel for `Aᵀ · D` for free. Store CSC if your workload is transpose-heavy.

For the scatter direction there's one accumulator per thread, and if
`threads × cols × k` would blow past `DEFAULT_SCRATCH_BUDGET` (64 MiB) the dense columns
are processed in blocks.

## Migrating from 1.x

2.0 is a clean break. Full list in [CHANGELOG.md](CHANGELOG.md).

| 1.x | 2.0 |
|---|---|
| `nalgebra_sparse::CsrMatrix<f64>` | `SvdMat<f64>` (`sprs::CsMatI<f64, u32, u64>`) |
| `lanczos::svd_dim_seed(&m, k, seed)` | `single_svdlib::svd_seed(&m, k, seed)` |
| `randomized::randomized_svd(&m, k, o, q, norm, center, seed, verbose)` | `randomized::svd_with(&m, &RandomizedConfig::new(k)…, progress)` |
| `MaskedCSRMatrix` | `MaskedCsMat` |
| `SMat` trait | `SparseMat` + `SparseMatDense` |
| `anyhow::Result` from `randomized` | `single_svdlib::Result` everywhere |
| `svd.u` orientation varied by solver | always `m × d` |

Several 1.x results were wrong, not just differently shaped — `randomized` panicked for
every built-in matrix type, mean centering computed the wrong quantity, and `seed: None`
meant seed 0. If you have numbers from 1.x, re-check them. Details in the changelog.

### LAS2 is deprecated

`lanczos` is still there so 2.0 doesn't silently drop the API, but it's `#[deprecated]`,
behind an off-by-default feature, and **you shouldn't use it**. Against a dense LAPACK
reference it gets the largest singular value 18–100% wrong on every matrix class tested —
including `diag(n, n-1, …, 1)`, where asking for the full rank reports `32` when the
answer is `40`.

```toml
# Only to keep a 1.x caller compiling while it migrates.
single-svdlib = { version = "2.0", features = ["las2"] }
```

This came from 1.x, not the sprs port — running `single-svdlib 1.0.9` from crates.io on
the same fixtures reproduces the same wrong values. One cause is fixed: `imtqlb` hoisted
its shift `p = d[l]` out of the iteration loop, where EISPACK's `IMTQL1` assigns it inside
(label 120), so every eigenvalue after the first used a stale shift. That's where 1.x's
`imtqlb had some convergence issues` warnings came from. At least one more remains:
`ritvec` reads `s[k*js + i]` — row `k` — while `imtql2` stores eigenvectors as columns.
Transposing roughly halves the error but doesn't close it.

Use `irlba`. It matches LAPACK to 1e-10 on the same fixtures, diagonal case included, and
its memory is bounded. `cargo test --release --all-features -- --ignored
lanczos::tests::report_accuracy_vs_lapack` prints the current error profile.

## Licence

BSD, same as the original SVDLIBC. See `SVDLIBC-LICENSE.txt`.

## Credits

- Original SVDLIBC by Doug Rohde
- Rust port of SVDLIBC by Dave Farnham
- Extensions and modifications by Ian F. Diks
