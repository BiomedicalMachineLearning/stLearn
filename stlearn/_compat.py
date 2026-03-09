import spatialdata as sd
from anndata import AnnData


def get_adata(data, table_key="table"):
    """Extract AnnData from either SpatialData or AnnData input."""
    if isinstance(data, sd.SpatialData):
        return data.tables[table_key]
    elif isinstance(data, AnnData):
        return data
    else:
        raise TypeError(f"Expected SpatialData or AnnData, got {type(data)}")


def is_spatial_data(data):
    return isinstance(data, sd.SpatialData)
