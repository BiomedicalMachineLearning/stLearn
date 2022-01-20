"""
Helper functions for porting R code into python/stLearn.
"""

import os

ro = None
pandas2ri = None
localconverter = None


def rpy2_setup(r_path):
    """Sets up rpy2."""
    os.environ["R_HOME"] = r_path

    import rpy2.robjects as robjects_
    from rpy2.robjects import pandas2ri as pandas2ri_
    from rpy2.robjects.conversion import localconverter as localconverter_

    global ro, pandas2ri, localconverter
    ro = robjects_
    pandas2ri = pandas2ri_
    localconverter = localconverter_
