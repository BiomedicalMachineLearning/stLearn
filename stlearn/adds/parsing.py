from typing import Optional, Union
from anndata import AnnData
from matplotlib import pyplot as plt
from pathlib import Path
import os
import sys


def parsing(
    adata: AnnData,
    coordinates_file: Union[Path, str],
    copy: bool = True,
) -> Optional[AnnData]:

    # Get a map of the new coordinates
    new_coordinates = dict()
    with open(coordinates_file, "r") as filehandler:
        for line in filehandler.readlines():
            tokens = line.split()
            assert(len(tokens) == 6 or len(tokens) == 4)
            if tokens[0] != "x":
                old_x = int(tokens[0])
                old_y = int(tokens[1])
                new_x = round(float(tokens[2]), 2)
                new_y = round(float(tokens[3]), 2)
                if len(tokens) == 6:
                    pixel_x = float(tokens[4])
                    pixel_y = float(tokens[5])
                    new_coordinates[(old_x, old_y)] = (pixel_x, pixel_y)
                else:
                    raise ValueError("Error, output format is pixel coordinates but\n "
                                     "the coordinates file only contains 4 columns\n")

    counts_table = adata.to_df()
    new_index_values = list()

    imgcol = []
    imgrow = []
    for index in counts_table.index:
        tokens = index.split("x")
        x = int(tokens[0])
        y = int(tokens[1])
        try:
            new_x, new_y = new_coordinates[(x, y)]
            imgcol.append(new_x)
            imgrow.append(new_y)

            new_index_values.append("{0}x{1}".format(new_x, new_y))
        except KeyError:
            counts_table.drop(index, inplace=True)

    # Assign the new indexes
    #counts_table.index = new_index_values

    # Remove genes that have now a total count of zero
    counts_table = counts_table.transpose(
    )[counts_table.sum(axis=0) > 0].transpose()

    adata = AnnData(counts_table)

    adata.obs["imagecol"] = imgcol
    adata.obs["imagerow"] = imgrow

    return adata if copy else None
