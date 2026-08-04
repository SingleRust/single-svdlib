//! The primitives every solve is built from. A solve issues hundreds of these, so a
//! regression here is a regression everywhere.
//!
//! The two product directions are timed separately: `A·D` on CSR is write-disjoint and
//! needs no scratch, `Aᵀ·D` scatters and runs a different kernel. One combined number
//! would hide a regression in either.
//!
//! ```text
//! cargo bench --bench kernels
//! ```

mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use single_svdlib::{dense, MaskedCsMat, SparseMat, SparseMatDense};
use std::hint::black_box;

fn sparse_products(c: &mut Criterion) {
    let a = common::counts(60_000, 4_000, 120, 16, 7);
    let (rows, cols, nnz) = (a.rows(), a.cols(), a.nnz());

    let mut group = c.benchmark_group("sparse_product");
    group.throughput(Throughput::Elements(nnz as u64));

    let x = vec![1.0f64; cols];
    let mut y = vec![0.0f64; rows];
    group.bench_function("mul_vec/gather", |b| {
        b.iter(|| a.mul_vec(black_box(&x), black_box(&mut y), false))
    });

    let xt = vec![1.0f64; rows];
    let mut yt = vec![0.0f64; cols];
    group.bench_function("mul_vec/scatter", |b| {
        b.iter(|| a.mul_vec(black_box(&xt), black_box(&mut yt), true))
    });

    // Block width decides whether the scatter kernel splits into column blocks to stay
    // inside its scratch budget, so sweep it.
    for k in [8usize, 32, 64] {
        let rhs = Array2::<f64>::from_elem((cols, k), 1.0);
        let mut out = Array2::<f64>::zeros((rows, k));
        group.throughput(Throughput::Elements((nnz * k) as u64));
        group.bench_with_input(BenchmarkId::new("mul_dense/gather", k), &k, |b, _| {
            b.iter(|| a.mul_dense(black_box(rhs.view()), black_box(out.view_mut()), false))
        });

        let rhs_t = Array2::<f64>::from_elem((rows, k), 1.0);
        let mut out_t = Array2::<f64>::zeros((cols, k));
        group.bench_with_input(BenchmarkId::new("mul_dense/scatter", k), &k, |b, _| {
            b.iter(|| a.mul_dense(black_box(rhs_t.view()), black_box(out_t.view_mut()), true))
        });
    }
    group.finish();
}

/// A view costs a lookup per non-zero; an extraction costs one copy. Which wins depends
/// on how many products follow, so time both.
fn masked_view(c: &mut Criterion) {
    let a = common::counts(60_000, 4_000, 120, 16, 7);
    let selected = common::every_nth(a.cols(), 8);
    let view = MaskedCsMat::with_columns(&a, &selected);
    let extracted = view.to_sparse();

    let mut group = c.benchmark_group("masked");
    let x = vec![1.0f64; view.cols()];
    let mut y = vec![0.0f64; view.rows()];

    group.bench_function("mul_vec/view", |b| {
        b.iter(|| view.mul_vec(black_box(&x), black_box(&mut y), false))
    });
    group.bench_function("mul_vec/extracted", |b| {
        b.iter(|| extracted.mul_vec(black_box(&x), black_box(&mut y), false))
    });
    group.bench_function("to_sparse", |b| b.iter(|| black_box(view.to_sparse())));
    group.finish();
}

/// The explained-variance denominator. The centered form also walks the column means,
/// so it is measurably more work than the plain one.
fn norms(c: &mut Criterion) {
    let a = common::counts(60_000, 4_000, 120, 16, 7);
    let means = a.col_means();

    let mut group = c.benchmark_group("norm");
    group.throughput(Throughput::Elements(a.nnz() as u64));
    group.bench_function("squared_frobenius", |b| {
        b.iter(|| black_box(a.squared_frobenius()))
    });
    group.bench_function("centered_squared_frobenius", |b| {
        b.iter(|| black_box(a.centered_squared_frobenius(means.view())))
    });
    group.bench_function("col_means", |b| b.iter(|| black_box(a.col_means())));
    group.finish();
}

/// TSQR only switches to panels above `MIN_ROWS_FOR_PANELS`, so cover both regimes.
fn tsqr(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsqr");
    for (rows, cols) in [(1_024usize, 32usize), (8_192, 32), (65_536, 64)] {
        let m = common::block(rows, cols, 11);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        // `tsqr` factors in place, so batch a fresh copy per iteration.
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}")),
            &m,
            |b, m| {
                b.iter_batched_ref(
                    || m.clone(),
                    |m| black_box(dense::tsqr(m).unwrap()),
                    criterion::BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(benches, sparse_products, masked_view, norms, tsqr);
criterion_main!(benches);
