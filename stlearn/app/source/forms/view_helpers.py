""" Helper functions for views.py.
"""

import numpy


def getVal(form, element):
    return getattr(form, element).data


def getData(form):
    """Retrieves the data from the form and places into dictionary."""
    params = {}
    form_elements = form.elements
    form_fields = form.element_fields
    for i, element in enumerate(form_elements):
        if form_fields[i] != "Title":
            data = getVal(form, element)
            params[element] = data
    return params


def getLR(lr_input, gene_names):
    """Returns list of lr_inputs and error message, if any."""
    if lr_input == "":
        return None, "ERROR: LR pairs required input."

    try:
        lrs = [lr.strip(" ") for lr in lr_input.split(",")]
        absent_genes = []
        for lr in lrs:
            genes = lr.split("_")
            absent_genes.extend([gene for gene in genes if gene not in gene_names])

        if len(absent_genes) != 0:
            return None, f"ERROR: inputted genes not found {absent_genes}."

        return lrs, ""

    except:
        return None, "ERROR: LR pairs misformatted."
