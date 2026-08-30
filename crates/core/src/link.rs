//! Link functions: how the linear predictor `eta` maps to the mean `mu`.

use crate::error::GlassError;

/// `g(mu) = eta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// `mu = eta`.
    Identity,
    /// `mu = exp(eta)`: multiplicative effects, the workhorse for counts and amounts.
    Log,
    /// `mu = 1 / (1 + exp(-eta))`: probabilities.
    Logit,
}

impl Link {
    /// Look a link up by name.
    ///
    /// # Errors
    /// Unknown name.
    pub fn parse(name: &str) -> Result<Self, GlassError> {
        match name {
            "identity" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            "logit" => Ok(Self::Logit),
            _ => Err(GlassError::BadArgument {
                name: "link",
                problem: "unknown link",
                fix: "one of: identity, log, logit",
            }),
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Log => "log",
            Self::Logit => "logit",
        }
    }

    /// `g(mu)`.
    #[inline]
    #[must_use]
    pub fn link(self, mu: f64) -> f64 {
        match self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Logit => (mu / (1.0 - mu)).ln(),
        }
    }

    /// `g^{-1}(eta)`.
    #[inline]
    #[must_use]
    pub fn inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => eta,
            Self::Log => eta.exp(),
            Self::Logit => 1.0 / (1.0 + (-eta).exp()),
        }
    }

    /// `d mu / d eta`, evaluated at `eta`.
    #[inline]
    #[must_use]
    pub fn mu_eta(self, eta: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Log => eta.exp(),
            Self::Logit => {
                let p = self.inverse(eta);
                p * (1.0 - p)
            }
        }
    }
}
