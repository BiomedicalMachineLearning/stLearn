from pathlib import Path

import geopandas as gpd

from stlearn._compat import get_adata, is_spatial_data


def polygon_annotations(
    data,
    annotations,
    label_column="label",
    obs_key="region",
    spatial_key="spatial",
    table_key="table",
    copy=False,
):
    """
    Annotate cells/spots by spatial overlap with polygon regions.

    Parameters
    ----------
    data
        SpatialData or AnnData object with spatial coordinates.
    annotations
        GeoDataFrame or path to GeoJSON/shapefile.
    ...
    """
    adata = get_adata(data, table_key)
    if copy:
        adata = adata.copy()

    if isinstance(annotations, (str, Path)):
        annotations = gpd.read_file(annotations)

    coords = adata.obsm[spatial_key]
    points = gpd.GeoDataFrame(
        index=adata.obs_names,
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
    )

    joined = gpd.sjoin(points, annotations, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    adata.obs[obs_key] = (
        joined.reindex(adata.obs_names)[label_column].astype("category").values
    )

    # If SpatialData, also store the polygons as a shapes element
    if is_spatial_data(data):
        import spatialdata.models as models

        parsed = models.ShapesModel.parse(annotations)
        data.shapes[obs_key] = parsed
        data.tables[table_key] = adata

    n_annotated = adata.obs[obs_key].notna().sum()
    print(
        f"Added polygon annotations to adata.obs['{obs_key}']: "
        f"{n_annotated}/{adata.n_obs} cells/spots annotated"
    )

    if is_spatial_data(data):
        return data
    return adata if copy else None
