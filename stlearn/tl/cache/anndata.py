import anndata as ad
import numpy as np
import pandas as pd


def write_subset_h5ad(adata, filename, obsm_keys=None, uns_keys=None):
    """Write only specific obsm and uns components to H5AD"""

    # Create a minimal AnnData object with the same structure
    minimal_adata = ad.AnnData(
        X=np.zeros((adata.n_obs, 1)),
        obs=adata.obs.index.to_frame(name="cell_id"),
        var=pd.DataFrame(index=["placeholder"]),
    )

    if obsm_keys:
        for key in obsm_keys:
            if key in adata.obsm:
                value = adata.obsm[key]
                if isinstance(value, list):
                    value = np.array(value)
                minimal_adata.obsm[key] = value
                print(f"Added obsm['{key}'] with shape {value.shape}")
            else:
                print(f"Warning: obsm['{key}'] not found")

    if uns_keys:
        for key in uns_keys:
            if key in adata.uns:
                minimal_adata.uns[key] = adata.uns[key]
                print(f"Added uns['{key}']")
            else:
                print(f"Warning: uns['{key}'] not found")

    minimal_adata.write_h5ad(filename, compression="gzip", compression_opts=9)
    print(f"Wrote subset to {filename}")


def merge_h5ad_into_adata(adata_main, h5ad_file):
    adata_subset = ad.read_h5ad(h5ad_file)
    print(f"Reading {h5ad_file}")

    for key, value in adata_subset.obsm.items():
        adata_main.obsm[key] = value
        print(f"Added obsm['{key}'] with shape {value.shape}")

    for key, value in adata_subset.uns.items():
        adata_main.uns[key] = value
        print(f"Added uns['{key}']")

    return adata_main
