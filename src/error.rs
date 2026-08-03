use thiserror::Error;

/// Everything this crate can fail with.
///
/// In 1.x the Lanczos path returned `SvdLibError` while the randomized path returned
/// `anyhow::Error`; both now return this type.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SvdLibError {
    /// A caller-supplied parameter was rejected before any work started.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The operands' shapes are inconsistent.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A tridiagonal/bidiagonal eigenproblem failed to converge.
    ///
    /// `stage` names the kernel (`imtqlb`, `imtql2`, ...) so the origin stays visible
    /// without a separate variant per call site.
    #[error("{stage}: no convergence after {iterations} iterations")]
    NoConvergence {
        stage: &'static str,
        iterations: usize,
    },

    /// The algorithm ran but could not produce the requested number of dimensions.
    #[error("{stage}: {message}")]
    Failed {
        stage: &'static str,
        message: String,
    },

    /// A dense factorization from the linear-algebra backend failed.
    #[error("dense {factorization} failed: {message}")]
    DenseFactorization {
        factorization: &'static str,
        message: String,
    },

    #[error("ndarray shape error: {0}")]
    Shape(String),
}

impl From<ndarray::ShapeError> for SvdLibError {
    fn from(e: ndarray::ShapeError) -> Self {
        SvdLibError::Shape(e.to_string())
    }
}

impl SvdLibError {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        SvdLibError::InvalidArgument(msg.into())
    }

    pub(crate) fn failed(stage: &'static str, msg: impl Into<String>) -> Self {
        SvdLibError::Failed {
            stage,
            message: msg.into(),
        }
    }

    pub(crate) fn shape(msg: impl Into<String>) -> Self {
        SvdLibError::ShapeMismatch(msg.into())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SvdLibError>;
