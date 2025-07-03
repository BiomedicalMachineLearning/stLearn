from pathlib import Path
import numpy as np
import pandas as pd

def apply_alignment_transformation(
    coordinates: pd.DataFrame,
    transform_mat: np.ndarray,
    pixel_size_microns: float = 0.2125,
) -> pd.DataFrame:
    """
    Apply transformation matrix to convert coordinates between spaces.

    From https://kb.10xgenomics.com/hc/en-us/articles/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data

    Parameters
    ----------
    coordinates
        DataFrame with columns ['x_centroid', 'y_centroid'] in microns
    transform_mat
        Transformation matrix from Xenium project.
    pixel_size_microns
        Pixel size in microns

    Returns
    -------
    pd.DataFrame
        Transformed coordinates
    """

    # Microns to pixels and use inverse transformation matrix
    coords_pixels = coordinates.values / pixel_size_microns
    transform_mat_inv = np.linalg.inv(transform_mat)
    coords_homogeneous = np.column_stack([coords_pixels, np.ones(len(coords_pixels))])
    transformed_coords = np.dot(coords_homogeneous, transform_mat_inv.T)

    # Extract x, y coordinates (ignore homogeneous coordinate)
    result_coords = transformed_coords[:, :2]
    return pd.DataFrame(result_coords, columns=coordinates.columns)
