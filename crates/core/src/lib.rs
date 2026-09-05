//! glasshouse-core: the numerics. Pure Rust, no Python types, returns plain Rust values.
//!
//! Rules that live here (see CLAUDE.md): one implementation per formula, every metric takes
//! weights, and bad input fails early and clearly with reasons why.

pub mod calibration;
pub mod classification;
pub mod constraints;
pub mod error;
pub mod family;
pub mod glm;
pub mod linalg;
pub mod link;
pub mod metrics;
pub mod ranking;
pub mod regression;
pub mod splines;

pub use error::GlassError;
pub use family::Family;
pub use link::Link;
