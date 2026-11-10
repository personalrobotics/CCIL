"""CCIL: continuity-based corrective labels for imitation learning."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ccil")  # single source: pyproject.toml
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

from .errors import InputError  # noqa: E402  (every refused input; the CLI exits 2 on it)

__all__ = ["InputError", "__version__"]
