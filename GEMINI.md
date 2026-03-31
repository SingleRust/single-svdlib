# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build          # Build the library
cargo test           # Run all tests
cargo test <name>    # Run a specific test by name
cargo clippy         # Lint
cargo fmt            # Format code
cargo doc --open     # Generate and open documentation
```

## Architecture

`single-svdlib` is a Rust port of SVDLIBC's Lanczos-based SVD algorithm for sparse matrices, with an additional randomized SVD implementation.

### Core Abstractions (`src/utils.rs`)

**`SMat<T: Float>`** — the central trait abstracting sparse matrix operations. Implemented for `CsrMatrix`, `CscMatrix`, and `CooMatrix` from `nalgebra-sparse`. Key method: `svd_opa(x, y, transposed)` computes `y = A*x` or `y = Aᵀ*x` in-place.

**`SvdFloat`** — bounds `f32`/`f64` with library-specific epsilon methods (`eps()`, `eps34()`).

**`SvdRec<T>`** — the result type. Fields `u`, `s`, `vt` are `ndarray` `Array2`/`Array1`. Note: `u` and `vt` are stored **transposed** (rows = singular vectors). `d` is the computed rank.

**`SvdLibError`** — custom error enum for algorithm-stage failures.

### Algorithms

**Lanczos LAS2** (`src/lanczos/mod.rs`) — the main algorithm. Entry points:
- `svd_las2(matrix, dimensions, iterations, end, kappa, seed)` — full control
- `svd(matrix)`, `svd_dim(matrix, d)`, `svd_dim_seed(matrix, d, seed)` — convenience wrappers

Internal state is managed via `WorkSpace` and `Store` structs. Parallel thresholds: `PARALLEL_THRESHOLD_ROWS = 5000`, `PARALLEL_THRESHOLD_COLS = 1000` — Rayon kicks in above these.

**Randomized SVD** (`src/randomized/mod.rs`) — for very large/extremely sparse matrices. Entry: `randomized_svd(matrix, k, n_oversampling, n_power_iter, normalization, seed, center)`. Normalization options: QR, LU, or none.

**Masked SVD** (`src/lanczos/masked.rs`) — wraps a `CsrMatrix` with a column mask so Lanczos operates on a subspace without copying data. Use `MaskedCSRMatrix`.

### Module Layout

```
src/
├── lib.rs          # Public re-exports, integration tests
├── error.rs        # SvdLibError enum
├── utils.rs        # SMat, SvdFloat, SvdRec, Diagnostics traits/types
├── lanczos/
│   ├── mod.rs      # LAS2 Lanczos algorithm + convenience wrappers
│   └── masked.rs   # Column-masked CSR wrapper
└── randomized/
    └── mod.rs      # Randomized SVD
```

### Key Design Choices

- All compute functions return `Result<SvdRec<T>, SvdLibError>`.
- The `SMat` trait makes the algorithms matrix-format-agnostic; adding support for a new sparse format means implementing `SMat`.
- `SvdRec.u` and `SvdRec.vt` use `ndarray` (not `nalgebra`) for interop with the broader `single-rust` ecosystem.
- Diagnostics in `SvdRec.diagnostics` capture per-run metadata (iterations, convergence, Ritz values) useful for debugging numerical issues.
