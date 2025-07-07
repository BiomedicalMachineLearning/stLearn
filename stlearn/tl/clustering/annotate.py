from anndata import AnnData
from bokeh.io import output_notebook
from bokeh.plotting import show

from stlearn.pl.classes_bokeh import Annotate


def annotate_interactive(
    adata: AnnData,
):
    """\
    Allow user to manually define the clusters

    Parameters
    -------------------------------------
    adata
        Annotated data matrix.
    """

    bokeh_object = Annotate(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)
