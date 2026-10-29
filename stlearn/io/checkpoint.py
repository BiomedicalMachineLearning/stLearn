import spatialdata

import stlearn


def checkpoint(sdata, name, compute, done=None, rerun=False):
    """The table `name` from the store if it is there (and `done` says it has this
    stage's results), else compute() saved as `name`. compute() may update the
    stored table in place and return None."""
    stored = sdata.tables.get(name)
    if stored is not None and not rerun and (done is None or done(stored)):
        print(f"{name}: loaded from {sdata.path}")
        return stored
    adata = compute()
    if adata is None:
        adata = stored
    sdata.tables[name] = adata
    stlearn.io.save_table(sdata, name)
    return adata


def store_checkpoint(path, compute):
    """The SpatialData store at `path` if it exists, else compute() written there."""
    if not path.exists():
        compute().write(path)
    return spatialdata.read_zarr(path)
