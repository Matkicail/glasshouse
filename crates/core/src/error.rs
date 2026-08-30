//! The one error type. Messages say what was wrong, how many rows, and what to do about it.

use std::fmt;

/// Input that a model or metric refuses to work on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlassError {
    /// Two arrays that must be the same length are not.
    LengthMismatch {
        /// Name of the first array (e.g. `y`).
        left: &'static str,
        left_len: usize,
        /// Name of the second array (e.g. `mu`).
        right: &'static str,
        right_len: usize,
    },
    /// Values outside the support of the family / metric.
    InvalidValues {
        /// Which array.
        name: &'static str,
        /// How many rows were bad.
        count: usize,
        /// The rule they broke, e.g. `must be >= 0`.
        rule: &'static str,
        /// What the caller should do.
        fix: &'static str,
    },
    /// Nothing to compute on.
    Empty { name: &'static str },
}

impl fmt::Display for GlassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                left,
                left_len,
                right,
                right_len,
            } => write!(
                f,
                "{left} has {left_len} rows but {right} has {right_len}; they must be the same length"
            ),
            Self::InvalidValues {
                name,
                count,
                rule,
                fix,
            } => write!(f, "{name} {rule}, but {count} row(s) are not — {fix}"),
            Self::Empty { name } => write!(f, "{name} is empty; nothing to compute"),
        }
    }
}

impl std::error::Error for GlassError {}

/// Check that two arrays are the same length.
///
/// # Errors
/// [`GlassError::LengthMismatch`] if they are not.
pub fn same_length<A, B>(
    left: &'static str,
    a: &[A],
    right: &'static str,
    b: &[B],
) -> Result<(), GlassError> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(GlassError::LengthMismatch {
            left,
            left_len: a.len(),
            right,
            right_len: b.len(),
        })
    }
}

/// Check every value satisfies `ok`; report how many did not.
///
/// # Errors
/// [`GlassError::InvalidValues`] naming the rule and the fix.
pub fn all_values(
    name: &'static str,
    values: &[f64],
    rule: &'static str,
    fix: &'static str,
    ok: impl Fn(f64) -> bool,
) -> Result<(), GlassError> {
    let count = values.iter().filter(|&&v| !ok(v)).count();
    if count == 0 {
        Ok(())
    } else {
        Err(GlassError::InvalidValues {
            name,
            count,
            rule,
            fix,
        })
    }
}
