"""Collection rules for the whole repo (this file is not shipped in the wheel).

The research package's docstring examples need torch; without it (the macOS and Windows CI
runners, or a core-only install) they are not collected, the same way the research tests
skip themselves.
"""

try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore_glob = ["python/glasshouse/research/*"]
