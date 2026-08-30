//! glasshouse-core: the numerics. Pure Rust, no Python types, returns plain Rust values.
//!
//! Rules that live here (see CLAUDE.md): one implementation per formula, every metric takes
//! weights, and bad input fails early and clearly with reasons why.

pub mod error;
pub mod family;
pub mod metrics;

pub use error::GlassError;
pub use family::Family;
