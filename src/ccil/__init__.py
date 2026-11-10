"""CCIL: continuity-based corrective labels for imitation learning."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ccil")  # single source: pyproject.toml
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

from .errors import InputError  # noqa: E402  (every refused input; the CLI exits 2 on it)

# The golden-path entry points (docs/contract.md), loaded on first use so `import ccil` stays light.
_API = ("Config", "CurateResult", "DiagnoseResult", "GenerateResult", "Inspection", "curate", "diagnose", "generate",
        "inspect", "resolve_config")


def __getattr__(name):
    if name in _API:
        from . import api
        return getattr(api, name)
    if name == "CurationPolicy":
        from .core.curate import CurationPolicy
        return CurationPolicy
    raise AttributeError(f"module 'ccil' has no attribute {name!r}")


__all__ = ["InputError", "CurationPolicy", "__version__", *_API]
