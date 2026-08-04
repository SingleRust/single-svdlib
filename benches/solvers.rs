//! End-to-end solves, at the shape and rank a single-cell PCA uses.
//!
//! Slower than the kernel benches, and the only place a restart-policy regression shows
//! up — that leaves every kernel untouched but doubles the work.
//!
//! ```text
//! cargo bench --bench solvers
//! ```

mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use single_svdlib::{irlba, randomized, MaskedCsMat, SparseMatDense};
use std::hint::black_box;
use std::time::Duration;

/// Big enough for the parallel paths to engage, small enough to finish in minutes.
fn fixture() -> single_svdlib::SvdMat<f64> {
    common::counts(40_000, 3_000, 100, 16, 7)
}

fn irlba_ranks(c: &mut Criterion) {
    let a = fixture();
    let mut group = c.benchmark_group("irlba");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));

    for rank in [10usize, 30, 50] {
        group.bench_with_input(BenchmarkId::new("plain", rank), &rank, |b, &rank| {
            b.iter(|| black_box(irlba::svd_seed(&a, rank, 42).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("centered", rank), &rank, |b, &rank| {
            b.iter(|| black_box(irlba::svd_centered(&a, rank, Some(42)).unwrap()))
        });
    }
    group.finish();
}

/// `work` moves total time more than anything else: a wider basis costs more per restart
/// but needs far fewer of them. Worth 2.2x at 400k x 30k.
fn irlba_work(c: &mut Criterion) {
    let a = fixture();
    let rank = 50usize;
    let means = a.col_means();

    let mut group = c.benchmark_group("irlba_work");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    for work in [rank + 7, rank + 30, 2 * rank] {
        group.bench_with_input(BenchmarkId::from_parameter(work), &work, |b, &work| {
            b.iter(|| {
                black_box(
                    irlba::svd_with(
                        &a,
                        &irlba::IrlbaConfig::new(rank)
                            .seed(42)
                            .work(work)
                            .mean_center(true),
                        Some(means.clone()),
                    )
                    .unwrap(),
                )
            })
        });
    }
    group.finish();
}

fn randomized_sketches(c: &mut Criterion) {
    let a = fixture();
    let rank = 50usize;

    let mut group = c.benchmark_group("randomized");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    for q in [2usize, 4] {
        group.bench_with_input(BenchmarkId::new("power", q), &q, |b, &q| {
            b.iter(|| {
                black_box(
                    randomized::svd_with(
                        &a,
                        &randomized::RandomizedConfig::new(rank)
                            .seed(42)
                            .power_iterations(q),
                        None,
                    )
                    .unwrap(),
                )
            })
        });
    }
    group.bench_function("block_krylov/2", |b| {
        b.iter(|| black_box(randomized::svd_block_krylov(&a, rank, 2, Some(42)).unwrap()))
    });
    group.finish();
}

/// Extraction pays one copy and makes every later product cheaper, so time both routes.
fn masked_pca(c: &mut Criterion) {
    let a = fixture();
    let selected = common::every_nth(a.cols(), 6);
    let view = MaskedCsMat::with_columns(&a, &selected);
    let extracted = view.to_sparse();
    let rank = 30usize;

    let mut group = c.benchmark_group("masked_pca");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    group.bench_function("view", |b| {
        b.iter(|| black_box(irlba::svd_centered(&view, rank, Some(42)).unwrap()))
    });
    group.bench_function("extracted", |b| {
        b.iter(|| black_box(irlba::svd_centered(&extracted, rank, Some(42)).unwrap()))
    });
    group.bench_function("extract_then_solve", |b| {
        b.iter(|| {
            let sub = view.to_sparse();
            black_box(irlba::svd_centered(&sub, rank, Some(42)).unwrap())
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    irlba_ranks,
    irlba_work,
    randomized_sketches,
    masked_pca
);
criterion_main!(benches);
