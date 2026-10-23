"""Minimal stand-in for gym 0.19, enough to import correct_il without the legacy stack.

Only what correct_il's module-level code touches exists here: the environment
registry used by correct_il/envs/__init__.py. Constructing an environment is
not supported.
"""
from . import envs  # noqa: F401


class Env:
    """Placeholder for annotations such as `env: gym.Env`; never instantiable."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("legacy harness: gym environments are not available")
