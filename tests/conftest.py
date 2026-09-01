"""Session setup. Currently one thing: keep macOS from crashing on two OpenMP runtimes.

LightGBM's macOS wheel needs Homebrew's libomp; glum/tabmat wheels bundle their own OpenMP.
Loading both into one process is the classic duplicate-runtime segfault (it dies in tabmat's
first parallel call). Intel's documented escape hatch is KMP_DUPLICATE_LIB_OK, set before the
second runtime loads — which is why this lives in conftest and not in a test. Thread caps
keep the CI runners honest too. Linux and Windows are unaffected and left alone.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
