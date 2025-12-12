# ShIeLD/utils/testing.py


class SerialPool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        for x in iterable:
            yield func(x)


def patch_create_graphs_mp(create_graphs_module):
    """Patch multiprocessing in create_graphs to be deterministic in tests."""
    create_graphs_module.mp.Pool = SerialPool
    create_graphs_module.mp.cpu_count = lambda: 2
    create_graphs_module.mp.set_start_method = lambda *a, **k: None
