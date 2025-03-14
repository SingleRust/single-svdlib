use std::fmt::Debug;
use std::iter::Sum;
use std::mem;
use std::ops::{AddAssign, MulAssign, Neg, SubAssign};
use ndarray::{Array, Array1, Array2};
use num_traits::{Float, FromPrimitive, One, Zero};
use num_traits::real::Real;
use rand::{thread_rng, Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::error::SvdLibError;

pub trait SMat<T: Float> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn nnz(&self) -> usize;
    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool); // y = A*x
}

/// Singular Value Decomposition Components
///
/// # Fields
/// - d:  Dimensionality (rank), the number of rows of both `ut`, `vt` and the length of `s`
/// - ut: Transpose of left singular vectors, the vectors are the rows of `ut`
/// - s:  Singular values (length `d`)
/// - vt: Transpose of right singular vectors, the vectors are the rows of `vt`
/// - diagnostics: Computational diagnostics
#[derive(Debug, Clone, PartialEq)]
pub struct SvdRec<T: Float> {
    pub d: usize,
    pub ut: Array2<T>,
    pub s: Array1<T>,
    pub vt: Array2<T>,
    pub diagnostics: Diagnostics<T>
}

/// Computational Diagnostics
///
/// # Fields
/// - non_zero:  Number of non-zeros in the matrix
/// - dimensions: Number of dimensions attempted (bounded by matrix shape)
/// - iterations: Number of iterations attempted (bounded by dimensions and matrix shape)
/// - transposed:  True if the matrix was transposed internally
/// - lanczos_steps: Number of Lanczos steps performed
/// - ritz_values_stabilized: Number of ritz values
/// - significant_values: Number of significant values discovered
/// - singular_values: Number of singular values returned
/// - end_interval: left, right end of interval containing unwanted eigenvalues
/// - kappa: relative accuracy of ritz values acceptable as eigenvalues
/// - random_seed: Random seed provided or the seed generated
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics<T: Float> {
    pub non_zero: usize,
    pub dimensions: usize,
    pub iterations: usize,
    pub transposed: bool,
    pub lanczos_steps: usize,
    pub ritz_values_stabilized: usize,
    pub significant_values: usize,
    pub singular_values: usize,
    pub end_interval: [T; 2],
    pub kappa: T,
    pub random_seed: u32
}

/// Trait for floating point types that can be used with the SVD algorithm
pub trait SvdFloat: Float + FromPrimitive + Debug + Send + Sync + Zero + One + AddAssign + SubAssign + MulAssign + Neg<Output = Self> + Sum {
    fn eps() -> Self;
    fn eps34() -> Self;
    fn compare(a: Self, b: Self) -> bool;
}

impl SvdFloat for f32 {
    fn eps() -> Self {
        f32::EPSILON
    }

    fn eps34() -> Self {
        f32::EPSILON.powf(0.75)
    }

    fn compare(a: Self, b: Self) -> bool {
        (b - a).abs() < f32::EPSILON
    }
}

impl SvdFloat for f64 {
    fn eps() -> Self {
        f64::EPSILON
    }

    fn eps34() -> Self {
        f64::EPSILON.powf(0.75)
    }

    fn compare(a: Self, b: Self) -> bool {
        (b - a).abs() < f64::EPSILON
    }
}

/// SVD at full dimensionality, calls `svdLAS2` with the highlighted defaults
///
/// svdLAS2(A, `0`, `0`, `&[-1.0e-30, 1.0e-30]`, `1.0e-6`, `0`)
///
/// # Parameters
/// - A: Sparse matrix
pub fn svd<T, M>(a: &M) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T>
{
    let eps_small = T::from_f64(-1.0e-30).unwrap();
    let eps_large = T::from_f64(1.0e-30).unwrap();
    let kappa = T::from_f64(1.0e-6).unwrap();
    svd_las2(a, 0, 0, &[eps_small, eps_large], kappa, 0)
}

/// SVD at desired dimensionality, calls `svdLAS2` with the highlighted defaults
///
/// svdLAS2(A, dimensions, `0`, `&[-1.0e-30, 1.0e-30]`, `1.0e-6`, `0`)
///
/// # Parameters
/// - A: Sparse matrix
/// - dimensions: Upper limit of desired number of dimensions, bounded by the matrix shape
pub fn svd_dim<T, M>(a: &M, dimensions: usize) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T> {
    let eps_small = T::from_f64(-1.0e-30).unwrap();
    let eps_large = T::from_f64(1.0e-30).unwrap();
    let kappa = T::from_f64(1.0e-6).unwrap();

    svd_las2(a, dimensions, 0, &[eps_small, eps_large], kappa, 0)
}

/// SVD at desired dimensionality with supplied seed, calls `svdLAS2` with the highlighted defaults
///
/// svdLAS2(A, dimensions, `0`, `&[-1.0e-30, 1.0e-30]`, `1.0e-6`, random_seed)
///
/// # Parameters
/// - A: Sparse matrix
/// - dimensions: Upper limit of desired number of dimensions, bounded by the matrix shape
/// - random_seed: A supplied seed `if > 0`, otherwise an internal seed will be generated
pub fn svd_dim_seed<T, M>(a: &M, dimensions: usize, random_seed: u32) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T> {
    let eps_small = T::from_f64(-1.0e-30).unwrap();
    let eps_large = T::from_f64(1.0e-30).unwrap();
    let kappa = T::from_f64(1.0e-6).unwrap();

    svd_las2(a, dimensions, 0, &[eps_small, eps_large], kappa, random_seed)
}

/// Compute a singular value decomposition
///
/// # Parameters
///
/// - A: Sparse matrix
/// - dimensions: Upper limit of desired number of dimensions (0 = max),
///       where "max" is a value bounded by the matrix shape, the smaller of
///       the matrix rows or columns. e.g. `A.nrows().min(A.ncols())`
/// - iterations: Upper limit of desired number of lanczos steps (0 = max),
///       where "max" is a value bounded by the matrix shape, the smaller of
///       the matrix rows or columns. e.g. `A.nrows().min(A.ncols())`
///       iterations must also be in range [`dimensions`, `A.nrows().min(A.ncols())`]
/// - end_interval: Left, right end of interval containing unwanted eigenvalues,
///       typically small values centered around zero, e.g. `[-1.0e-30, 1.0e-30]`
/// - kappa: Relative accuracy of ritz values acceptable as eigenvalues, e.g. `1.0e-6`
/// - random_seed: A supplied seed `if > 0`, otherwise an internal seed will be generated
pub fn svd_las2<T, M>(
    a: &M,
    dimensions: usize,
    iterations: usize,
    end_interval: &[T; 2],
    kappa: T,
    random_seed: u32
) -> Result<SvdRec<T>, SvdLibError>
where
    T: SvdFloat,
    M: SMat<T> {
    let random_seed = match random_seed > 0 {
        true => random_seed,
        false => thread_rng().gen::<_>()
    };

    let min_nrows_ncols = a.nrows().min(a.ncols());

    let dimensions = match dimensions {
        n if n == 0 || n > min_nrows_ncols => min_nrows_ncols,
        _ => dimensions
    };

    let iterations = match iterations {
        n if n == 0 || n > min_nrows_ncols => min_nrows_ncols,
        n if n < dimensions => dimensions,
        _ => iterations
    };

    if dimensions < 2 {
        return Err(SvdLibError::Las2Error(format!(
            "svd_las2: insufficient dimensions: {dimensions}"
        )))
    }

    assert!(dimensions > 1 && dimensions <= min_nrows_ncols);
    assert!(iterations >= dimensions && iterations <= min_nrows_ncols);

    let transposed = (a.ncols() as f64) >= ((a.nrows() as f64) * 1.2);
    let nrows = if transposed { a.ncols() } else { a.nrows() };
    let ncols = if transposed { a.nrows() } else { a.ncols() };

    let mut wrk = WorkSpace::new(nrows, ncols, transposed, iterations)?;
    let mut store = Store::new(ncols)?;

    let mut neig = 0;
    let steps = lanso(
        a,
        dimensions,
        iterations,
        end_interval,
        &mut wrk,
        &mut neig,
        &mut store,
        random_seed
    )?;

    let kappa = kappa.abs().max(T::eps34());
    let mut r = ritvec(a, dimensions, kappa, &mut wrk, steps, neig, &mut store)?;

    if transposed {
        mem::swap(&mut r.Ut, &mut r.Vt);
    }

    Ok(SvdRec {
        // Dimensionality (number of Ut,Vt rows & length of S)
        d: r.d,
        ut: Array2::from_shape_vec((r.d, r.Ut.cols), r.Ut.value)?,
        s: Array::from_shape_vec(r.d, r.S)?,
        vt: Array2::from_shape_vec((r.d, r.Vt.cols), r.Vt.value)?,
        diagnostics: Diagnostics {
            non_zero: a.nnz(),
            dimensions: dimensions,
            iterations: iterations,
            transposed: transposed,
            lanczos_steps: steps + 1,
            ritz_values_stabilized: neig,
            significant_values: r.d,
            singular_values: r.nsig,
            end_interval: *end_interval,
            kappa: kappa,
            random_seed: random_seed,
        },
    })
}

const MAXLL: usize = 2;

#[derive(Debug, Clone, PartialEq)]
struct Store<T: Float> {
    n: usize,
    vecs: Vec<Vec<T>>
}

impl<T: Float + Zero + Clone> Store<T> {
    fn new(n: usize) -> Result<Self, SvdLibError> {
        Ok(Self { n, vecs: vec![] })
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

    fn retrq(&mut self, idx: usize) -> &[T] {
        &self.vecs[idx + MAXLL]
    }

    fn retrp(&mut self, idx: usize) -> &[T] {
        &self.vecs[idx]
    }
}

#[derive(Debug, Clone, PartialEq)]
struct WorkSpace<T: Float> {
    nrows: usize,
    ncols: usize,
    transposed: bool,
    w0: Vec<T>,     // workspace 0
    w1: Vec<T>,     // workspace 1
    w2: Vec<T>,     // workspace 2
    w3: Vec<T>,     // workspace 3
    w4: Vec<T>,     // workspace 4
    w5: Vec<T>,     // workspace 5
    alf: Vec<T>,    // array to hold diagonal of the tridiagonal matrix T
    eta: Vec<T>,    // orthogonality estimate of Lanczos vectors at step j
    oldeta: Vec<T>, // orthogonality estimate of Lanczos vectors at step j-1
    bet: Vec<T>,    // array to hold off-diagonal of T
    bnd: Vec<T>,    // array to hold the error bounds
    ritz: Vec<T>,   // array to hold the ritz values
    temp: Vec<T>,   // array to hold the temp values
}

impl<T: Float + Zero + FromPrimitive> WorkSpace<T> {
    fn new(nrows: usize, ncols: usize, transposed: bool, iterations: usize) -> Result<Self, SvdLibError> {
        Ok(Self {
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
            bnd: vec![T::from_f64(f64::MAX).unwrap(); 1 + iterations],
            temp: vec![T::zero(); nrows],
        })
    }
}

/* Row-major dense matrix.  Rows are consecutive vectors. */
#[derive(Debug, Clone, PartialEq)]
struct DMat<T: Float> {
    cols: usize,
    value: Vec<T>,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq)]
struct SVDRawRec<T: Float> {
    d: usize,
    nsig: usize,
    Ut: DMat<T>,
    S: Vec<T>,
    Vt: DMat<T>,
}

fn compare<T: SvdFloat>(computed: T, expected: T) -> bool {
    T::compare(computed, expected)
}

/* Function sorts array1 and array2 into increasing order for array1 */
fn insert_sort<T: PartialOrd>(n: usize, array1: &mut [T], array2: &mut [T]) {
    for i in 1..n {
        for j in (1..i + 1).rev() {
            if array1[j - 1] <= array1[j] {
                break;
            }
            array1.swap(j - 1, j);
            array2.swap(j - 1, j);
        }
    }
}

#[allow(non_snake_case)]
#[rustfmt::skip]
fn svd_opb<T: Float>(A: &dyn SMat<T>, x: &[T], y: &mut [T], temp: &mut [T], transposed: bool) {
    let nrows = if transposed { A.ncols() } else { A.nrows() };
    let ncols = if transposed { A.nrows() } else { A.ncols() };
    assert_eq!(x.len(), ncols, "svd_opb: x must be A.ncols() in length, x = {}, A.ncols = {}", x.len(), ncols);
    assert_eq!(y.len(), ncols, "svd_opb: y must be A.ncols() in length, y = {}, A.ncols = {}", y.len(), ncols);
    assert_eq!(temp.len(), nrows, "svd_opa: temp must be A.nrows() in length, temp = {}, A.nrows = {}", temp.len(), nrows);
    A.svd_opa(x, temp, transposed); // temp = (A * x)
    A.svd_opa(temp, y, !transposed); // y = A' * (A * x) = A' * temp
}

// constant times a vector plus a vector
fn svd_daxpy<T: Float + AddAssign>(da: T, x: &[T], y: &mut [T]) {
    for (xval, yval) in x.iter().zip(y.iter_mut()) {
        *yval += da * *xval
    }
}

// finds the index of element having max absolute value
fn svd_idamax<T: Float>(n: usize, x: &[T]) -> usize {
    assert!(n > 0, "svd_idamax: unexpected inputs!");

    match n {
        1 => 0,
        _ => {
            let mut imax = 0;
            for (i, xval) in x.iter().enumerate().take(n).skip(1) {
                if xval.abs() > x[imax].abs() {
                    imax = i;
                }
            }
            imax
        }
    }
}

// returns |a| if b is positive; else fsign returns -|a|
fn svd_fsign<T: Float>(a: T, b: T) -> T {
    match (a >= T::zero() && b >= T::zero()) || (a < T::zero() && b < T::zero()) {
        true => a,
        false => -a,
    }
}

// finds sqrt(a^2 + b^2) without overflow or destructive underflow
fn svd_pythag<T: SvdFloat + FromPrimitive>(a: T, b: T) -> T {
    match a.abs().max(b.abs()) {
        n if n > T::zero() => {
            let mut p = n;
            let mut r = (a.abs().min(b.abs()) / p).powi(2);
            let four = T::from_f64(4.0).unwrap();
            let two = T::from_f64(2.0).unwrap();
            let mut t = four + r;
            while !compare(t, four) {
                let s = r / t;
                let u = T::one() + two * s;
                p = p * u;
                r = (s / u).powi(2);
                t = four + r;
            }
            p
        }
        _ => T::zero(),
    }
}

// dot product of two vectors
fn svd_ddot<T: Float + Sum<T>>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y).map(|(a, b)| *a * *b).sum()
}

// norm (length) of a vector
fn svd_norm<T: Float + Sum<T>>(x: &[T]) -> T {
    svd_ddot(x, x).sqrt()
}

// scales an input vector 'x', by a constant, storing in 'y'
fn svd_datx<T: Float + Sum<T>>(d: T, x: &[T], y: &mut [T]) {
    for (i, xval) in x.iter().enumerate() {
        y[i] = d * *xval;
    }
}

// scales an input vector 'x' by a constant, modifying 'x'
fn svd_dscal<T: Float + MulAssign>(d: T, x: &mut [T]) {
    for elem in x.iter_mut() {
        *elem *= d;
    }
}

// copies a vector x to a vector y (reversed direction)
fn svd_dcopy<T: Float + Copy>(n: usize, offset: usize, x: &[T], y: &mut [T]) {
    if n > 0 {
        let start = n - 1;
        for i in 0..n {
            y[offset + start - i] = x[offset + i];
        }
    }
}

fn imtqlb<T: SvdFloat>(n: usize, d: &mut [T], e: &mut [T], bnd: &mut [T]) -> Result<(), SvdLibError> {
    if n == 1 {
        return Ok(());
    }

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
        while iteration <= 30 {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = d[m].abs() + d[m + 1].abs();
                if compare(test, test + e[m].abs()) {
                    break; // convergence = true;
                }
                m += 1;
            }
            let mut p = d[l];
            let mut f = bnd[l];
            if m == l {
                // order the eigenvalues
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
                iteration = 31;
            } else {
                if iteration == 30 {
                    return Err(SvdLibError::ImtqlbError(
                        "imtqlb no convergence to an eigenvalue after 30 iterations".to_string(),
                    ));
                }
                iteration += 1;
                // ........ form shift ........
                let two = T::from_f64(2.0).unwrap();
                let mut g = (d[l + 1] - p) / (two * e[l]);
                let mut r = svd_pythag(g, T::one());
                g = d[m] - p + e[l] / (g + svd_fsign(r, g));
                let mut s = T::one();
                let mut c = T::one();
                p = T::zero();

                assert!(m > 0, "imtqlb: expected 'm' to be non-zero");
                i = m - 1;
                let mut underflow = false;
                while !underflow && i >= l {
                    f = s * e[i];
                    let b = c * e[i];
                    r = svd_pythag(f, g);
                    e[i + 1] = r;
                    if compare(r, T::zero()) {
                        underflow = true;
                        break;
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
                // ........ recover from underflow .........
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
    Ok(())
}

#[allow(non_snake_case)]
fn startv<T: SvdFloat>(
    A: &dyn SMat<T>,
    wrk: &mut WorkSpace<T>,
    step: usize,
    store: &mut Store<T>,
    random_seed: u32,
) -> Result<T, SvdLibError> {
    // get initial vector; default is random
    let mut rnm2 = svd_ddot(&wrk.w0, &wrk.w0);
    for id in 0..3 {
        if id > 0 || step > 0 || compare(rnm2, T::zero()) {
            let mut bytes = [0; 32];
            for (i, b) in random_seed.to_le_bytes().iter().enumerate() {
                bytes[i] = *b;
            }
            let mut seeded_rng = StdRng::from_seed(bytes);
            for val in wrk.w0.iter_mut() {
                *val = T::from_f64(seeded_rng.gen_range(-1.0..1.0)).unwrap();
            }
        }
        wrk.w3.copy_from_slice(&wrk.w0);

        // apply operator to put r in range (essential if m singular)
        svd_opb(A, &wrk.w3, &mut wrk.w0, &mut wrk.temp, wrk.transposed);
        wrk.w3.copy_from_slice(&wrk.w0);
        rnm2 = svd_ddot(&wrk.w3, &wrk.w3);
        if rnm2 > T::zero() {
            break;
        }
    }

    if rnm2 <= T::zero() {
        return Err(SvdLibError::StartvError(format!("rnm2 <= 0.0, rnm2 = {rnm2:?}")));
    }

    if step > 0 {
        for i in 0..step {
            let v = store.retrq(i);
            svd_daxpy(-svd_ddot(&wrk.w3, v), v, &mut wrk.w0);
        }

        // make sure q[step] is orthogonal to q[step-1]
        svd_daxpy(-svd_ddot(&wrk.w4, &wrk.w0), &wrk.w2, &mut wrk.w0);
        wrk.w3.copy_from_slice(&wrk.w0);

        rnm2 = match svd_ddot(&wrk.w3, &wrk.w3) {
            dot if dot <= T::eps() * rnm2 => T::zero(),
            dot => dot,
        }
    }
    Ok(rnm2.sqrt())
}

#[allow(non_snake_case)]
fn stpone<T: SvdFloat>(
    A: &dyn SMat<T>,
    wrk: &mut WorkSpace<T>,
    store: &mut Store<T>,
    random_seed: u32,
) -> Result<(T, T), SvdLibError> {
    // get initial vector; default is random
    let mut rnm = startv(A, wrk, 0, store, random_seed)?;
    if compare(rnm, T::zero()) {
        return Err(SvdLibError::StponeError("rnm == 0.0".to_string()));
    }

    // normalize starting vector
    svd_datx(rnm.recip(), &wrk.w0, &mut wrk.w1);
    svd_dscal(rnm.recip(), &mut wrk.w3);

    // take the first step
    svd_opb(A, &wrk.w3, &mut wrk.w0, &mut wrk.temp, wrk.transposed);
    wrk.alf[0] = svd_ddot(&wrk.w0, &wrk.w3);
    svd_daxpy(-wrk.alf[0], &wrk.w1, &mut wrk.w0);
    let t = svd_ddot(&wrk.w0, &wrk.w3);
    wrk.alf[0] += t;
    svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
    wrk.w4.copy_from_slice(&wrk.w0);
    rnm = svd_norm(&wrk.w4);
    let anorm = rnm + wrk.alf[0].abs();
    Ok((rnm, T::eps().sqrt() * anorm))
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn lanczos_step<T: SvdFloat>(
    A: &dyn SMat<T>,
    wrk: &mut WorkSpace<T>,
    first: usize,
    last: usize,
    ll: &mut usize,
    enough: &mut bool,
    rnm: &mut T,
    tol: &mut T,
    store: &mut Store<T>,
) -> Result<usize, SvdLibError> {
    let eps1 = T::eps() * T::from_f64(wrk.ncols as f64).unwrap().sqrt();
    let mut j = first;
    let four = T::from_f64(4.0).unwrap();

    while j < last {
        mem::swap(&mut wrk.w1, &mut wrk.w2);
        mem::swap(&mut wrk.w3, &mut wrk.w4);

        store.storq(j - 1, &wrk.w2);
        if j - 1 < MAXLL {
            store.storp(j - 1, &wrk.w4);
        }
        wrk.bet[j] = *rnm;

        // restart if invariant subspace is found
        if compare(*rnm, T::zero()) {
            *rnm = startv(A, wrk, j, store, 0)?;
            if compare(*rnm, T::zero()) {
                *enough = true;
            }
        }

        if *enough {
            mem::swap(&mut wrk.w1, &mut wrk.w2);
            break;
        }

        // take a lanczos step
        svd_datx(rnm.recip(), &wrk.w0, &mut wrk.w1);
        svd_dscal(rnm.recip(), &mut wrk.w3);
        svd_opb(A, &wrk.w3, &mut wrk.w0, &mut wrk.temp, wrk.transposed);
        svd_daxpy(-*rnm, &wrk.w2, &mut wrk.w0);
        wrk.alf[j] = svd_ddot(&wrk.w0, &wrk.w3);
        svd_daxpy(-wrk.alf[j], &wrk.w1, &mut wrk.w0);

        // orthogonalize against initial lanczos vectors
        if j <= MAXLL && wrk.alf[j - 1].abs() > four * wrk.alf[j].abs() {
            *ll = j;
        }
        for i in 0..(j - 1).min(*ll) {
            let v1 = store.retrp(i);
            let t = svd_ddot(v1, &wrk.w0);
            let v2 = store.retrq(i);
            svd_daxpy(-t, v2, &mut wrk.w0);
            wrk.eta[i] = eps1;
            wrk.oldeta[i] = eps1;
        }

        // extended local reorthogonalization
        let t = svd_ddot(&wrk.w0, &wrk.w4);
        svd_daxpy(-t, &wrk.w2, &mut wrk.w0);
        if wrk.bet[j] > T::zero() {
            wrk.bet[j] += t;
        }
        let t = svd_ddot(&wrk.w0, &wrk.w3);
        svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
        wrk.alf[j] += t;
        wrk.w4.copy_from_slice(&wrk.w0);
        *rnm = svd_norm(&wrk.w4);
        let anorm = wrk.bet[j] + wrk.alf[j].abs() + *rnm;
        *tol = T::eps().sqrt() * anorm;

        // update the orthogonality bounds
        ortbnd(wrk, j, *rnm, eps1);

        // restore the orthogonality state when needed
        purge(wrk.ncols, *ll, wrk, j, rnm, *tol, store);
        if *rnm <= *tol {
            *rnm = T::zero();
        }
        j += 1;
    }
    Ok(j)
}

fn purge<T: SvdFloat>(
    n: usize,
    ll: usize,
    wrk: &mut WorkSpace<T>,
    step: usize,
    rnm: &mut T,
    tol: T,
    store: &mut Store<T>,
) {
    if step < ll + 2 {
        return;
    }

    let reps = T::eps().sqrt();
    let eps1 = T::eps() * T::from_f64(n as f64).unwrap().sqrt();
    let two = T::from_f64(2.0).unwrap();

    let k = svd_idamax(step - (ll + 1), &wrk.eta) + ll;
    if wrk.eta[k].abs() > reps {
        let reps1 = eps1 / reps;
        let mut iteration = 0;
        let mut flag = true;
        while iteration < 2 && flag {
            if *rnm > tol {
                // bring in a lanczos vector t and orthogonalize both r and q against it
                let mut tq = T::zero();
                let mut tr = T::zero();
                for i in ll..step {
                    let v = store.retrq(i);
                    let t = svd_ddot(v, &wrk.w3);
                    tq += t.abs();
                    svd_daxpy(-t, v, &mut wrk.w1);
                    let t = svd_ddot(v, &wrk.w4);
                    tr += t.abs();
                    svd_daxpy(-t, v, &mut wrk.w0);
                }
                wrk.w3.copy_from_slice(&wrk.w1);
                let t = svd_ddot(&wrk.w0, &wrk.w3);
                tr += t.abs();
                svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
                wrk.w4.copy_from_slice(&wrk.w0);
                *rnm = svd_norm(&wrk.w4);
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

fn ortbnd<T: SvdFloat>(wrk: &mut WorkSpace<T>, step: usize, rnm: T, eps1: T) {
    if step < 1 {
        return;
    }
    if !compare(rnm, T::zero()) && step > 1 {
        wrk.oldeta[0] =
            (wrk.bet[1] * wrk.eta[1] + (wrk.alf[0] - wrk.alf[step]) * wrk.eta[0] - wrk.bet[step] * wrk.oldeta[0]) / rnm
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

fn error_bound<T: SvdFloat>(
    enough: &mut bool,
    endl: T,
    endr: T,
    ritz: &mut [T],
    bnd: &mut [T],
    step: usize,
    tol: T,
) -> usize {
    assert!(step > 0, "error_bound: expected 'step' to be non-zero");

    // massage error bounds for very close ritz values
    let mid = svd_idamax(step + 1, bnd);
    let sixteen = T::from_f64(16.0).unwrap();

    let mut i = ((step + 1) + (step - 1)) / 2;
    while i > mid + 1 {
        if (ritz[i - 1] - ritz[i]).abs() < T::eps34() * ritz[i].abs() && bnd[i] > tol && bnd[i - 1] > tol {
            bnd[i - 1] = (bnd[i].powi(2) + bnd[i - 1].powi(2)).sqrt();
            bnd[i] = T::zero();
        }
        i -= 1;
    }

    let mut i = ((step + 1) - (step - 1)) / 2;
    while i + 1 < mid {
        if (ritz[i + 1] - ritz[i]).abs() < T::eps34() * ritz[i].abs() && bnd[i] > tol && bnd[i + 1] > tol {
            bnd[i + 1] = (bnd[i].powi(2) + bnd[i + 1].powi(2)).sqrt();
            bnd[i] = T::zero();
        }
        i += 1;
    }

    // refine the error bounds
    let mut neig = 0;
    let mut gapl = ritz[step] - ritz[0];
    for i in 0..=step {
        let mut gap = gapl;
        if i < step {
            gapl = ritz[i + 1] - ritz[i];
        }
        gap = gap.min(gapl);
        if gap > bnd[i] {
            bnd[i] *= bnd[i] / gap;
        }
        if bnd[i] <= sixteen * T::eps() * ritz[i].abs() {
            neig += 1;
            if !*enough {
                *enough = endl < ritz[i] && ritz[i] < endr;
            }
        }
    }
    neig
}

fn imtql2<T: SvdFloat>(nm: usize, n: usize, d: &mut [T], e: &mut [T], z: &mut [T]) -> Result<(), SvdLibError> {
    if n == 1 {
        return Ok(());
    }
    assert!(n > 1, "imtql2: expected 'n' to be > 1");
    let two = T::from_f64(2.0).unwrap();

    let last = n - 1;

    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[last] = T::zero();

    let nnm = n * nm;
    for l in 0..n {
        let mut iteration = 0;

        // look for small sub-diagonal element
        while iteration <= 30 {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = d[m].abs() + d[m + 1].abs();
                if compare(test, test + e[m].abs()) {
                    break; // convergence = true;
                }
                m += 1;
            }
            if m == l {
                break;
            }

            // error -- no convergence to an eigenvalue after 30 iterations.
            if iteration == 30 {
                return Err(SvdLibError::Imtql2Error(
                    "imtql2 no convergence to an eigenvalue after 30 iterations".to_string(),
                ));
            }
            iteration += 1;

            // form shift
            let mut g = (d[l + 1] - d[l]) / (two * e[l]);
            let mut r = svd_pythag(g, T::one());
            g = d[m] - d[l] + e[l] / (g + svd_fsign(r, g));

            let mut s = T::one();
            let mut c = T::one();
            let mut p = T::zero();

            assert!(m > 0, "imtql2: expected 'm' to be non-zero");
            let mut i = m - 1;
            let mut underflow = false;
            while !underflow && i >= l {
                let mut f = s * e[i];
                let b = c * e[i];
                r = svd_pythag(f, g);
                e[i + 1] = r;
                if compare(r, T::zero()) {
                    underflow = true;
                } else {
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + two * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;

                    // form vector
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
            } /* end while (underflow != FALSE && i >= l) */
            /*........ recover from underflow .........*/
            if underflow {
                d[i + 1] -= p;
            } else {
                d[l] -= p;
                e[l] = g;
            }
            e[m] = T::zero();
        }
    }

    // order the eigenvalues
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

        // ...and corresponding eigenvectors
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

fn rotate_array<T: Float + Copy>(a: &mut [T], x: usize) {
    let n = a.len();
    let mut j = 0;
    let mut start = 0;
    let mut t1 = a[0];

    for _ in 0..n {
        j = match j >= x {
            true => j - x,
            false => j + n - x,
        };

        let t2 = a[j];
        a[j] = t1;

        if j == start {
            j += 1;
            start = j;
            t1 = a[j];
        } else {
            t1 = t2;
        }
    }
}

#[allow(non_snake_case)]
fn ritvec<T: SvdFloat>(
    A: &dyn SMat<T>,
    dimensions: usize,
    kappa: T,
    wrk: &mut WorkSpace<T>,
    steps: usize,
    neig: usize,
    store: &mut Store<T>,
) -> Result<SVDRawRec<T>, SvdLibError> {
    let js = steps + 1;
    let jsq = js * js;
    let mut s = vec![T::zero(); jsq];

    // initialize s to an identity matrix
    for i in (0..jsq).step_by(js + 1) {
        s[i] = T::one();
    }

    let mut Vt = DMat {
        cols: wrk.ncols,
        value: vec![T::zero(); wrk.ncols * dimensions],
    };

    svd_dcopy(js, 0, &wrk.alf, &mut Vt.value);
    svd_dcopy(steps, 1, &wrk.bet, &mut wrk.w5);

    // on return from imtql2(), `R.Vt.value` contains eigenvalues in
    // ascending order and `s` contains the corresponding eigenvectors
    imtql2(js, js, &mut Vt.value, &mut wrk.w5, &mut s)?;

    let mut nsig = 0;
    let mut x = 0;
    let mut id2 = jsq - js;
    for k in 0..js {
        if wrk.bnd[k] <= kappa * wrk.ritz[k].abs() && k + 1 > js - neig {
            x = match x {
                0 => dimensions - 1,
                _ => x - 1,
            };

            let offset = x * Vt.cols;
            Vt.value[offset..offset + Vt.cols].fill(T::zero());
            let mut idx = id2 + js;
            for i in 0..js {
                idx -= js;
                if s[idx] != T::zero() {
                    for (j, item) in store.retrq(i).iter().enumerate().take(Vt.cols) {
                        Vt.value[j + offset] += s[idx] * *item;
                    }
                }
            }
            nsig += 1;
        }
        id2 += 1;
    }

    // Rotate the singular vectors and values.
    // `x` is now the location of the highest singular value.
    if x > 0 {
        rotate_array(&mut Vt.value, x * Vt.cols);
    }

    // final dimension size
    let d = dimensions.min(nsig);
    let mut S = vec![T::zero(); d];
    let mut Ut = DMat {
        cols: wrk.nrows,
        value: vec![T::zero(); wrk.nrows * d],
    };
    Vt.value.resize(Vt.cols * d, T::zero());

    let mut tmp_vec = vec![T::zero(); Vt.cols];
    for (i, sval) in S.iter_mut().enumerate() {
        let vt_offset = i * Vt.cols;
        let ut_offset = i * Ut.cols;

        let vt_vec = &Vt.value[vt_offset..vt_offset + Vt.cols];
        let ut_vec = &mut Ut.value[ut_offset..ut_offset + Ut.cols];

        // multiply by matrix B first
        svd_opb(A, vt_vec, &mut tmp_vec, &mut wrk.temp, wrk.transposed);
        let t = svd_ddot(vt_vec, &tmp_vec);

        // store the Singular Value at S[i]
        *sval = t.sqrt();

        svd_daxpy(-t, vt_vec, &mut tmp_vec);
        wrk.bnd[js] = svd_norm(&tmp_vec) * sval.recip();

        // multiply by matrix A to get (scaled) left s-vector
        A.svd_opa(vt_vec, ut_vec, wrk.transposed);
        svd_dscal(sval.recip(), ut_vec);
    }

    Ok(SVDRawRec {
        // Dimensionality (rank)
        d,

        // Significant values
        nsig,

        // DMat Ut  Transpose of left singular vectors. (d by m)
        //          The vectors are the rows of Ut.
        Ut,

        // Array of singular values. (length d)
        S,

        // DMat Vt  Transpose of right singular vectors. (d by n)
        //          The vectors are the rows of Vt.
        Vt,
    })
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn lanso<T: SvdFloat>(
    A: &dyn SMat<T>,
    dim: usize,
    iterations: usize,
    end_interval: &[T; 2],
    wrk: &mut WorkSpace<T>,
    neig: &mut usize,
    store: &mut Store<T>,
    random_seed: u32,
) -> Result<usize, SvdLibError> {
    let (endl, endr) = (end_interval[0], end_interval[1]);

    /* take the first step */
    let rnm_tol = stpone(A, wrk, store, random_seed)?;
    let mut rnm = rnm_tol.0;
    let mut tol = rnm_tol.1;

    let eps1 = T::eps() * T::from_f64(wrk.ncols as f64).unwrap().sqrt();
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

        // the actual lanczos loop
        let steps = lanczos_step(A, wrk, first, last, &mut ll, &mut enough, &mut rnm, &mut tol, store)?;
        j = match enough {
            true => steps - 1,
            false => last - 1,
        };

        first = j + 1;
        wrk.bet[first] = rnm;

        // analyze T
        let mut l = 0;
        for _ in 0..j {
            if l > j {
                break;
            }

            let mut i = l;
            while i <= j {
                if compare(wrk.bet[i + 1], T::zero()) {
                    break;
                }
                i += 1;
            }
            i = i.min(j);

            // now i is at the end of an unreduced submatrix
            let sz = i - l;
            svd_dcopy(sz + 1, l, &wrk.alf, &mut wrk.ritz);
            svd_dcopy(sz, l + 1, &wrk.bet, &mut wrk.w5);

            imtqlb(sz + 1, &mut wrk.ritz[l..], &mut wrk.w5[l..], &mut wrk.bnd[l..])?;

            for m in l..=i {
                wrk.bnd[m] = rnm * wrk.bnd[m].abs();
            }
            l = i + 1;
        }

        // sort eigenvalues into increasing order
        insert_sort(j + 1, &mut wrk.ritz, &mut wrk.bnd);

        *neig = error_bound(&mut enough, endl, endr, &mut wrk.ritz, &mut wrk.bnd, j, tol);

        // should we stop?
        if *neig < dim {
            if *neig == 0 {
                last = first + 9;
                intro = first;
            } else {
                last = first + 3.max(1 + ((j - intro) * (dim - *neig)) / *neig);
            }
            last = last.min(iterations);
        } else {
            enough = true
        }
        enough = enough || first >= iterations;
    }
    store.storq(j, &wrk.w1);
    Ok(j)
}

impl<T: SvdFloat + 'static> SvdRec<T> {
    pub fn recompose(&self) -> Array2<T> {
        let sdiag = Array2::from_diag(&self.s);
        self.ut.t().dot(&sdiag).dot(&self.vt)
    }
}

#[rustfmt::skip]
impl<T: Float + Zero + AddAssign + Clone> SMat<T> for nalgebra_sparse::csc::CscMatrix<T> {
    fn nrows(&self) -> usize { self.nrows() }
    fn ncols(&self) -> usize { self.ncols() }
    fn nnz(&self) -> usize { self.nnz() }

    /// takes an n-vector x and returns A*x in y
    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = if transposed { self.ncols() } else { self.nrows() };
        let ncols = if transposed { self.nrows() } else { self.ncols() };
        assert_eq!(x.len(), ncols, "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}", x.len(), ncols);
        assert_eq!(y.len(), nrows, "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}", y.len(), nrows);

        let (major_offsets, minor_indices, values) = self.csc_data();

        for y_val in y.iter_mut() {
            *y_val = T::zero();
        }

        if transposed {
            for (i, yval) in y.iter_mut().enumerate() {
                for j in major_offsets[i]..major_offsets[i + 1] {
                    *yval += values[j] * x[minor_indices[j]];
                }
            }
        } else {
            for (i, xval) in x.iter().enumerate() {
                for j in major_offsets[i]..major_offsets[i + 1] {
                    y[minor_indices[j]] += values[j] * *xval;
                }
            }
        }
    }
}

#[rustfmt::skip]
impl<T: Float + Zero + AddAssign + Clone> SMat<T> for nalgebra_sparse::csr::CsrMatrix<T> {
    fn nrows(&self) -> usize { self.nrows() }
    fn ncols(&self) -> usize { self.ncols() }
    fn nnz(&self) -> usize { self.nnz() }

    /// takes an n-vector x and returns A*x in y
    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = if transposed { self.ncols() } else { self.nrows() };
        let ncols = if transposed { self.nrows() } else { self.ncols() };
        assert_eq!(x.len(), ncols, "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}", x.len(), ncols);
        assert_eq!(y.len(), nrows, "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}", y.len(), nrows);

        let (major_offsets, minor_indices, values) = self.csr_data();

        for y_val in y.iter_mut() {
            *y_val = T::zero();
        }

        if !transposed {
            for (i, yval) in y.iter_mut().enumerate() {
                for j in major_offsets[i]..major_offsets[i + 1] {
                    *yval += values[j] * x[minor_indices[j]];
                }
            }
        } else {
            for (i, xval) in x.iter().enumerate() {
                for j in major_offsets[i]..major_offsets[i + 1] {
                    y[minor_indices[j]] += values[j] * *xval;
                }
            }
        }
    }
}

#[rustfmt::skip]
impl<T: Float + Zero + AddAssign + Clone> SMat<T> for nalgebra_sparse::coo::CooMatrix<T> {
    fn nrows(&self) -> usize { self.nrows() }
    fn ncols(&self) -> usize { self.ncols() }
    fn nnz(&self) -> usize { self.nnz() }

    /// takes an n-vector x and returns A*x in y
    fn svd_opa(&self, x: &[T], y: &mut [T], transposed: bool) {
        let nrows = if transposed { self.ncols() } else { self.nrows() };
        let ncols = if transposed { self.nrows() } else { self.ncols() };
        assert_eq!(x.len(), ncols, "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}", x.len(), ncols);
        assert_eq!(y.len(), nrows, "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}", y.len(), nrows);

        for y_val in y.iter_mut() {
            *y_val = T::zero();
        }

        if transposed {
            for (i, j, v) in self.triplet_iter() {
                y[j] += *v * x[i];
            }
        } else {
            for (i, j, v) in self.triplet_iter() {
                y[i] += *v * x[j];
            }
        }
    }
}

