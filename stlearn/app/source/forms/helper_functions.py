# Purpose of this script is to write the functions that help facilitate
# subsetting of the data depending on the users input
import numpy


def printOut(text, fileName="stdout.txt", close=True, file=None):
    """Prints to the specified file name. Used for debugging.
    If close is Fale, returns open file.
    """

    if type(file) == type(None):
        file = open(fileName, "w")

    print(text, file=file)

    if close:
        file.close()
    else:
        return file


def filterOptions(metaDataSets, options):
    """Returns options that overlap with keys in metaDataSets dictionary"""
    if type(options) == type(None):
        options = list(metaDataSets.keys())
    else:
        options = [option for option in options if option in metaDataSets.keys()]

    return options


def addChoices(metaDataSets, options, elementValues):
    """Helper function which generates choices for SelectMultiField"""
    for option in options:
        choices = [(optioni, optioni) for optioni in metaDataSets[option]]
        elementValues.append(choices)


# TODO update this so has 'options' as input
def subsetSCA(sca, subsetForm):
    """Subsets the SCA based on the selected fields and the inputted genes."""

    # Getting the attached fields from the form which refer subset options #
    options = filterOptions(sca.metaDataSets, subsetForm.elements)

    # Subsetting based on selection #
    conditionSelection = {}  # selection dictionary
    for i, option in enumerate(options):
        selected = getattr(subsetForm, option).data
        if len(selected) != 0:
            conditionSelection[option] = selected

    # Subsetting based on conditions #
    if len(conditionSelection) != 0:
        sca = sca.createConditionSubset("subset", conditionSelection)

    # Subsetting based on inputted genes #
    geneList = getattr(subsetForm, "Select Cells Expressing Gene/s").data.split(",")
    if geneList != [""]:
        # Filter to just the genes which express all of the inputted genes #
        sca = sca.createGeneExprsSubset(
            "subset", genesToFilter=geneList, cutoff=0, keep=True, useOr=False
        )

    return sca, conditionSelection, geneList
