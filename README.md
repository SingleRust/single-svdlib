# single-svdlib

[![Crate](https://img.shields.io/crates/v/single-svdlib.svg)](https://crates.io/crates/single-svdlib)
[![Documentation](https://docs.rs/single-svdlib/badge.svg)](https://docs.rs/single-svdlib)

Sparse singular value decomposition in Rust, over [`sprs`](https://crates.io/crates/sprs)
matrices.

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

// The two largest singular triplets.
let svd = single_svdlib::svd(&a, 2)?;

// A ≈ u · diag(s) · vt
assert_eq!(svd.u.dim(),  (4, 2));  // left vectors are columns
assert_eq!(svd.vt.dim(), (2, 3));  // right vectors are rows
# Ok::<(), single_svdlib::SvdLibError>(())
```

## Choosing a solver

| module | method | use when |
|---|---|---|
| `irlba` | thick-restarted Lanczos bidiagonalization | **default.** Accurate; memory bounded by the requested rank |
| `randomized` | randomized range finder — power iteration or block Krylov | very large inputs where an approximation is acceptable |
| `lanczos` | LAS2 from SVDLIBC | **deprecated — numerically unreliable.** See [below](#las2-is-deprecated) |

`single_svdlib::svd` dispatches to `irlba`.

```rust
use single_svdlib::{irlba, randomized};
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((40, 20));
# for i in 0..40 { for j in 0..20 { t.add_triplet(i, j, ((i * 7 + j * 3) % 11) as f64); } }
# let a: SvdMat<f64> = t.to_csr::<u64>();

// Reproducible.
let exact = irlba::svd_seed(&a, 10, 42)?;

// PCA: mean-centered, without ever densifying the matrix.
let pca = irlba::svd_centered(&a, 10, Some(42))?;

// Approximate, for when the matrix is too large to iterate on.
let approx = randomized::svd_seed(&a, 10, 42)?;

// Block Krylov: much more accurate when the spectrum decays slowly.
let better = randomized::svd_block_krylov(&a, 10, 3, Some(42))?;
# Ok::<(), single_svdlib::SvdLibError>(())
```

Full control is available through `IrlbaConfig` and `RandomizedConfig`:

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

// The last argument is an optional progress sink, called once per stage.
let svd = svd_with(&a, &cfg, Some(&|stage: &str| eprintln!("{stage}")))?;
# Ok::<(), single_svdlib::SvdLibError>(())
```

## Memory

Two things keep the footprint down.

**Narrow indices.** `SvdMat<T>` is `CsMatI<T, u32, u64>`: 32-bit column indices with
64-bit row pointers. Against `usize` for both that is 12 bytes per non-zero instead of
16 for `f64` data, and 8 instead of 16 for `f32`. The split pointer width keeps matrices
with more than `u32::MAX` non-zeros representable. Widen by naming the parameters:
`SvdMat<f64, u64, u64>`.

**A bounded basis.** `irlba` restarts, so it holds exactly `work + 1` right vectors and
`work` left vectors (`work = rank + 7` by default) however many restarts convergence
takes. Peak is therefore known before the solve begins:

```text
(work + 1) · cols + work · rows   scalars
```

Measured on a 200 000 × 30 000 matrix with 4.9M non-zeros, rank 50, 10 cores:

| | time | peak RSS |
|---|---|---|
| `irlba` | 2.2 s | 514 MiB |
| `randomized`, 2 power iterations | 1.2 s | 654 MiB |
| `randomized`, block Krylov ×3 | 4.0 s | 1080 MiB |

The matrix itself is 57.8 MiB, against 76.6 MiB with `usize` indices. Reproduce with:

```bash
cargo run --release --example scale irlba
```

Block Krylov's basis is `blocks × (rank + oversamples)` columns wide, so its memory
scales with `blocks` — that is the price of its accuracy.

## Sparse × dense products

A compressed matrix can only be traversed along its outer dimension, which makes the two
product directions genuinely different:

- `A · D` on a CSR matrix writes output row `i` from sparse row `i`. Threads own disjoint
  rows, so it needs **no scratch and no reduction**.
- `Aᵀ · D` scatters, so threads collide and accumulation is unavoidable.

`transpose_view()` does not escape this — it relabels a CSR matrix as a CSC view of the
transpose without changing which dimension is traversable. What it does buy is that a
**CSC-stored** matrix gets the disjoint kernel for `Aᵀ · D` for free. If your workload is
transpose-heavy, store CSC.

For the scatter direction the accumulator count is the *thread* count, and if
`threads × cols × k` would still exceed `DEFAULT_SCRATCH_BUDGET` (64 MiB) the dense
columns are processed in blocks so the bound always holds.

## Accuracy

The reduced factors these methods produce — IRLBA's `B`, the randomized path's `R` —
inherit their conditioning from the operand, and their factorization decides the accuracy
of the whole result. They are therefore factored with a **one-sided Jacobi SVD**, which is
accurate to the condition number *after* column scaling (Demmel & Veselić), rather than
with the bidiagonal QR a linear-algebra backend provides. On a `5 × 2` operand with
`κ ≈ 8·10⁶`, `nalgebra`'s Golub–Reinsch reconstructed to only `1.3·10⁻⁹` relative;
Jacobi reaches `< 1·10⁻¹⁵` on the same input. That is also why the crate has no
linear-algebra backend dependency — `sprs`, `ndarray`, `rayon`, `num-traits`, `rand` and
`thiserror` are the whole tree.

Correctness is checked three ways:

- **Unit and integration tests** compare every solver against a dense LAPACK-grade
  reference on fixed fixtures.
- **Adversarial tests** (`tests/robustness.rs`) assert that degenerate and hostile
  operands — all-zero, rank-deficient, duplicated rows, `NaN`, `∞`, 12-orders-of-magnitude
  dynamic range, empty masks — produce either a correct answer or a typed error, never a
  panic, a hang, or a silently wrong result.
- **Property tests** (`tests/properties.rs`) check invariants over shapes, densities and
  value distributions that `proptest` chooses: agreement with the dense reference,
  orthonormality, `A·vᵢ = σᵢ·uᵢ`, truncation error equal to the spectral tail,
  storage-order and index-width invariance, and reproducibility. Run them harder with
  `PROPTEST_CASES=100000 cargo test --release --test properties`.

## Large matrices, column subsets, PCA

The workload this crate is built for: reduce a very large sparse matrix, restricted to a
subset of columns, without densifying or modifying it.

```rust
use single_svdlib::{irlba, MaskedCsMat};
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((200, 60));
# for i in 0..200 { for j in 0..60 { t.add_triplet(i, j, ((i * 7 + j) % 13) as f64); } }
# let counts: SvdMat<f64> = t.to_csr::<u64>();
# let selected_genes: Vec<usize> = (0..60).step_by(3).collect();

// A view over the selected columns. No copy; `counts` is untouched.
let view = MaskedCsMat::with_columns(&counts, &selected_genes);

// PCA of that submatrix: centered implicitly, never densified.
let pca = irlba::svd_centered(&view, 10, Some(42))?;
// pca.u  — scores,   cells x components
// pca.vt — loadings, components x selected genes
# Ok::<(), single_svdlib::SvdLibError>(())
```

Measured on **1 000 000 cells × 30 000 genes**, 145M non-zeros, masked to 2238 genes,
50 components (`cargo run --release --example pca_at_scale`):

| | |
|---|---|
| matrix, sparse | 1.63 GiB |
| the same matrix dense | 223.52 GiB — **137× larger**, never materialised |
| building the view | 5 ms, no copy |
| PCA on the view | 44 s, converged |
| peak RSS | ~3.9 GiB |
| `‖A_c·vᵢ − σᵢ·uᵢ‖ / σ_max` | 1.8e-15 |

### Two things worth knowing

**A column mask does not make products cheaper.** Every product still walks all the
source's non-zeros and tests each against the mask; only the output width shrinks. If the
mask is restrictive and you are running an iterative solver — hundreds of products —
extract the submatrix once instead. It stays sparse, and the copy is repaid immediately:

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

At 1M × 30k the extraction took 146 ms and made the PCA 18% faster (44 s → 36 s).

**An unconverged result is an error, not a return value.** `irlba` refuses to hand back
triplets that did not reach `tol`, because a pipeline that forgets to inspect the
diagnostics would otherwise carry a silently degraded decomposition into everything
downstream. The error says what the residual was and what to change. If a best effort is
genuinely what you want, call `.allow_unconverged()` and check
[`SvdRec::converged`] / [`SvdRec::max_residual`] yourself.

**Raise `work` on tall matrices.** Re-orthogonalisation, not the sparse products,
dominates when there are many rows. See [`IrlbaConfig::work`] — `rank + 30` was 2.2×
faster than the default at identical accuracy. `cargo run --release --example tune` prints
the comparison for your own shape.

## Column masking

Run a solver on a subset of columns without materialising the submatrix:

```rust
use single_svdlib::MaskedCsMat;
# use single_svdlib::{sprs::TriMatI, SvdMat};
# let mut t = TriMatI::<f64, u32>::new((60, 20));
# for i in 0..60 { for j in 0..20 { t.add_triplet(i, j, ((i + j * 3) % 5) as f64); } }
# let a: SvdMat<f64> = t.to_csr::<u64>();

let masked = MaskedCsMat::with_columns(&a, &[0, 2, 5, 7]);
let svd = single_svdlib::svd(&masked, 3)?;
assert_eq!(svd.vt.ncols(), 4); // one column per selected index
# Ok::<(), single_svdlib::SvdLibError>(())
```

## Result

```rust,ignore
pub struct SvdRec<T> {
    pub d: usize,          // number of triplets returned
    pub u: Array2<T>,      // m × d, left vectors are columns
    pub s: Array1<T>,      // d, descending
    pub vt: Array2<T>,     // d × n, right vectors are rows
    pub diagnostics: Diagnostics<T>,
}
```

`A ≈ u · diag(s) · vt`, matching `numpy.linalg.svd`. `Diagnostics` carries a `matvecs`
count — sparse products issued, with a block product against `k` dense columns counted as
`k` — which is comparable across solvers and is the honest way to price one against
another. `Diagnostics::detail` carries per-algorithm figures such as `irlba`'s restart
count and converged flag.

## Migrating from 1.x

2.0 is a clean break.

| 1.x | 2.0 |
|---|---|
| `nalgebra_sparse::CsrMatrix<f64>` | `SvdMat<f64>` (`sprs::CsMatI<f64, u32, u64>`) |
| `lanczos::svd_dim_seed(&m, k, seed)` | `single_svdlib::svd_seed(&m, k, seed)` |
| `randomized::randomized_svd(&m, k, o, q, norm, center, seed, verbose)` | `randomized::svd_with(&m, &RandomizedConfig::new(k)…, progress)` |
| `MaskedCSRMatrix` | `MaskedCsMat` |
| `SMat` trait | `SparseMat` + `SparseMatDense` |
| `anyhow::Result` from `randomized` | `single_svdlib::Result` everywhere |
| `svd.u` orientation varied by solver | always `m × d` |

Behavioural changes worth knowing about:

- **`u` orientation is now consistent.** 1.x returned `u` as `d × m` from the Lanczos
  path but `m × d` from the randomized path, so `recompose()` only worked on square
  inputs. Both are now `m × d`.
- **Singular values are always descending.**
- **`randomized` actually works.** In 1.x, four of `SMat`'s five methods were `todo!()`
  in every built-in implementation, so `randomized_svd` panicked for `CsrMatrix`,
  `CscMatrix` and `CooMatrix` alike — every documented usage. Nine of the crate's
  seventeen tests failed.
- **Mean centering is fixed.** 1.x computed `(Σⱼ mⱼ)·(Σᵢ D[i,c])` where the correction is
  `Σⱼ mⱼ·D[j,c]` — the product of the sums instead of the sum of the products.
- **An unseeded randomized run is now actually random.** 1.x mapped `seed: None` to
  seed `0`, so every "random" sketch was identical.
- **Masked matrices no longer mis-dispatch.** 1.x delegated to the *unmasked* matrix for
  any small input regardless of the mask, feeding a masked-width vector to a full-width
  product.

### LAS2 is deprecated

The `lanczos` module is retained so 2.0 does not silently drop the API, but it is
`#[deprecated]` and **should not be used**. Checked against a dense LAPACK reference, it
returns the largest singular value with 18%–100% relative error on every matrix class
tested — including `diag(n, n-1, …, 1)`, where asking for the full rank still reports
`32` when the answer is `40`.

This is inherited from published 1.x, not introduced by the sprs port; running
`single-svdlib 1.0.9` from crates.io on identical fixtures reproduces the same wrong
values. Two causes are known:

1. **Fixed.** `imtqlb` hoisted its shift origin `p = d[l]` out of the iteration loop,
   where EISPACK `IMTQL1` assigns it *inside* (label 120), so every eigenvalue after the
   first was computed from a stale shift. This is the source of the
   `imtqlb had some convergence issues` warnings 1.x printed on nearly every input before
   continuing with corrupted Ritz values.
2. **Open.** `ritvec` reads `s[k*js + i]` — row `k` — while `imtql2` stores eigenvectors
   as columns. Transposing roughly halves the residual error but does not close it, so at
   least one further defect remains.

Use `irlba` instead. It is validated against LAPACK to 1e-10 on the same fixtures,
including the diagonal case, and its memory is bounded.

Run `cargo test --release -- --ignored lanczos::tests::report_accuracy_vs_lapack` to see
the current error profile.

## Licence

BSD, as the original SVDLIBC. See `SVDLIBC-LICENSE.txt`.

## Credits

- Original SVDLIBC by Doug Rohde
- Rust port of SVDLIBC by Dave Farnham
- Extensions and modifications by Ian F. Diks
