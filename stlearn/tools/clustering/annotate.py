from anndata import AnnData
from stlearn.plotting.classes_bokeh import Annotate
from bokeh.io import output_notebook
from bokeh.plotting import show


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
