# Changelog

## 2.0.0

A rewrite: different sparse backend, new solvers, and a handful of 1.x results that were
just wrong.

### Breaking

- `sprs` replaces `nalgebra-sparse`. `SvdMat<T, I = u32, Iptr = u64>` is `sprs::CsMatI`,
  generic over both index types — 12 bytes per non-zero for `f64` instead of 16.
  `nalgebra` is now dev-only.
- `u` is always `m × d` (numpy layout). 1.x returned `d × m` from Lanczos and `m × d`
  from randomized.
- `SMat` is split into `SparseMat` (shape, matvec) and `SparseMatDense` (block products,
  means, centering).
- `MaskedCSRMatrix` is now `MaskedCsMat`, and masks rows as well as columns.
- Solvers take config builders (`IrlbaConfig`, `RandomizedConfig`) instead of positional
  arguments.
- `randomized` returns `single_svdlib::Result`, not `anyhow::Result`.
- `svd` and `svd_dim` use IRLBA, not LAS2.
- LAS2 is deprecated and behind the `las2` feature, off by default.
- IRLBA errors instead of returning an unconverged result. `allow_unconverged()` opts out.
- MSRV 1.88.

### Added

- `irlba` — restarted Lanczos bidiagonalization. The default solver.
- `dense::tsqr` — tall-skinny QR.
- `dense::small_svd` — one-sided Jacobi, replacing a dependency that lost accuracy on
  badly scaled columns.
- Block Krylov sketching in `randomized`.
- `explained_variance`, `explained_variance_ratio`, `total_variance` and `scores` on
  `SvdRec`.
- `MaskedCsMat::to_sparse` — copy a view into an owned sparse submatrix.
- `SvdRec::converged` and `max_residual`.
- `Diagnostics::matvecs`.
- Property tests, an edge-case suite, and benchmarks.

### Fixed

All checked against the published 1.0.9.

- `randomized_svd` panicked for every built-in matrix type — four of five `SMat` methods
  were `todo!()`.
- Mean centering computed the product of the sums instead of the sum of the products.
- `seed: None` meant seed 0, so unseeded sketches were all identical.
- Masked matrices used the unmasked matrix for small inputs.
- NaN or infinity in the input could hang forever.
- All-zero matrices reported a non-zero singular value.
- Rank-deficient matrices could fail on an absolute breakdown test.
- `min(rows, cols) == 1` was rejected instead of solved directly.
- The centered Frobenius norm cancelled on columns with a large offset — 51% off at an
  `f32` offset of 1000. Now summed per entry.

### Known issues

LAS2 is still wrong: 18–100% error against LAPACK. One cause is fixed (`imtqlb` hoisted
its shift out of the loop), at least one remains in `ritvec`. Inherited from 1.x, not the
port. Its accuracy tests are `#[ignore]`d. Hence deprecated and gated.
