class _Registry:
    def __init__(self):
        self.env_specs = {}


registry = _Registry()


def register(id, entry_point, *args, **kwargs):
    registry.env_specs[id] = {"entry_point": entry_point, "kwargs": kwargs}
