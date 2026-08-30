"""Type stubs for the Rust extension. Keep in step with crates/py/src/lib.rs."""

import numpy as np
import numpy.typing as npt

def poisson_deviance(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
