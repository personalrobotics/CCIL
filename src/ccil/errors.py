"""The error every refused input raises: the CLI exits 2 on it (docs/contract.md, exit codes)."""


class InputError(ValueError):
    """An invalid configuration, dataset, checkpoint, artifact or combination; the message lists every problem found.

    Any other exception is a defect: the CLI exits 1 with its traceback."""
