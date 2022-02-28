"""Purpose of this script is to create general forms that are programmable with
	particular input. Will impliment forms for subsetting the data and
	visualisation options in a general way so can be used with any
	SingleCellAnalysis dataset.
"""

import sys
from flask_wtf import FlaskForm

# from flask_wtf.file import FileField
from wtforms import SelectMultipleField, SelectField
import wtforms


def createSuperForm(elements, element_fields, element_values, validators=None):
    """ Creates a general form; goal is to create a fully programmable form \
	that essentially governs all the options the user will select.

	Args:
		elements (list<str>): Element names to be rendered on the page, in \
							  order of how they will appear on the page.

		element_fields (list<str>): The names of the fields to be rendered. \
								Each field is in same order as 'elements'. \
								Currently supported are: \
								'Title', 'SelectMultipleField', 'SelectField', \
								'StringField', 'Text', 'List'.

		element_values (list<object>): The information which will be put into \
									the field. Changes depending on field: \

									'Title' and 'Text': 'object' is a string
									containing the title which will be added as \
									a heading when rendered on the page.

									'SelectMultipleField' and 'SelectField':
									'object' is list of options to select from.

									'StringField':
									The example values to display within the \
									fields text area. The 'placeholder' option.

									'List':
									A list of objects which will be attached \
									to the form.

		validators (list<FunctionHandles>): A list of functions which take the \
						form as input, used to construct the form validator. \
						Form validator constructed by calling these \
						sequentially with form 'self' as input.

	Args:
		form (list<WTForm>): A WTForm which has attached as variable all the \
		fields mentioned, so then when rendered as input to
		'SuperDataDisplay.html' shows the form.
	"""

    class SuperForm(FlaskForm):
        """A base form on which all of the fields will be added."""

    if type(validators) == type(None):
        validators = [None] * len(elements)

    # Add the information #
    SuperForm.elements = elements
    SuperForm.element_fields = element_fields

    multiSelectLeft = True  # Places multi-select field to left, alternatives
    # if many multi-selects in row
    for i, element in enumerate(elements):
        fieldName = element_fields[i]

        # Adding each element as the appropriate field to the form #
        if fieldName == "SelectMultipleField":
            setattr(
                SuperForm,
                element,
                SelectMultipleField(element, choices=element_values[i]),
            )
            # The point of this number is to give an order for the attributes,
            # so that odd numbers get rendered to right of page, even numbers
            # left.
            setattr(SuperForm, element + "_number", int(multiSelectLeft))
            # inverts, so if left, goes right for the next multiSelectField
            multiSelectLeft = multiSelectLeft == False

        else:
            multiSelectLeft = True  # Reset the MultiSelectField position

            if fieldName in ["Title", "List"]:
                setattr(SuperForm, element, element_values[i])

            elif fieldName == "SelectField":
                setattr(
                    SuperForm,
                    element,
                    SelectField(
                        element, choices=element_values[i], validators=validators[i]
                    ),
                )

            # elif fieldName == 'FileField':
            # 	setattr(SuperForm, element, FileField(validators=validators[i]))
            # 	setattr(SuperForm, element + '_placeholder',  # Setting default
            # 			element_values[i])

            elif fieldName in [
                "StringField",
                "IntegerField",
                "BooleanField",
                "FileField",
                "FloatField",
            ]:
                FieldClass = getattr(wtforms, fieldName)
                setattr(
                    SuperForm, element, FieldClass(element, validators=validators[i])
                )
                setattr(
                    SuperForm,
                    element + "_placeholder",  # Setting default
                    element_values[i],
                )

    return SuperForm


def getPreprocessForm():
    """Gets the preprocessing form generated from the superform above.

    Returns:
            FlaskForm: With attributes that allow for inputs that are related to
                                    pre-processing.
    """
    elements = [
        "Spot Quality Control Filtering",  # Title
        "Minimum genes per spot",
        "Minimum counts per spot",
        "Gene Quality Control Filtering",  # Title
        "Minimum spots per gene",
        "Minimum counts per gene",
        "Normalisation, Log-transform, & Scaling",  # Title
        "Normalize total",
        "Log 1P",
        "Scale",
    ]
    element_fields = [
        "Title",
        "IntegerField",
        "IntegerField",
        "Title",
        "IntegerField",
        "IntegerField",
        "Title",
        "BooleanField",
        "BooleanField",
        "BooleanField",
    ]
    element_values = ["", 200, 300, "", 3, 5, "", True, True, True]
    return createSuperForm(elements, element_fields, element_values)


def getLRForm():
    """Gets the LR form generated from the superform above.

        Returns:
                FlaskForm: With attributes that allow for inputs that are \
                            related to LR analysis.
    """
    elements = [
        "Species",
        "Spot neighbourhood (-1: smallest neighbourhood, 0: within-spot mode)",
        "Minimum spots with LR scores",
        "N random gene pairs (permutations)",
        "CPUs",
    ]
    element_fields = [
        "SelectField",
        "IntegerField",
        "IntegerField",
        "IntegerField",
        "IntegerField",
    ]
    element_values = [
        [("Human", "Human"), ("Mouse", "Mouse")],
        -1,
        20,
        100,
        2,
    ]
    return createSuperForm(elements, element_fields, element_values)


def getCCIForm(adata):
    """Gets the CCI form generated from the superform above.

    Returns:
            FlaskForm: With attributes that allow for inputs that are
                                                    related to CCI analysis.
    """
    elements = [
        "Cell information (only discrete labels available, unless mixture already in anndata.uns)",
        "Minimum spots for LR to be considered",
        "Spot mixture (only if the 'Cell Information' label selected available in anndata.uns)",
        "Cell proportion cutoff (value above which cell is considered in spot if 'Spot mixture' selected)",
        "Permutations (recommend atleast 1000)",
    ]
    element_fields = [
        "SelectField",
        "IntegerField",
        "BooleanField",
        "FloatField",
        "IntegerField",
    ]
    if type(adata) == type(None):
        fields = []
        mix = False
    else:
        fields = [
            key for key in adata.obs.keys() if type(adata.obs[key].values[0]) == str
        ]
        mix = fields[0] in adata.uns.keys()
    element_values = [fields, 20, mix, 0.2, 100]
    return createSuperForm(elements, element_fields, element_values)


def getCCIForm_old():
    """Gets the CCI form generated from the superform above.

    Returns:
            FlaskForm: With attributes that allow for inputs that are related to
                                    CCI analysis.
    """
    elements = [
        "* Cell Heterogeneity File",
        "Neighbourhood distance (0 indicates within-spot mode)",
        "** L-R pair input (e.g. L1_R1, L2_R2, ...)",
        "Permutations (0 indicates no permutation testing)",
    ]
    element_fields = ["FileField", "IntegerField", "StringField", "IntegerField"]
    element_values = ["", 25, "", 0]
    return createSuperForm(elements, element_fields, element_values)


def getClusterForm():
    """Gets the Cluster form generated using superform above.

    Returns:
            FlaskForm: With attributes that allow input related to clustering.
    """
    elements = [
        "PCA components",
        "stSME normalisation",
        "Cluster method",
        "K",
        "Resolution",
        "Neighbours (for Louvain/Leiden)",
    ]
    element_fields = [
        "IntegerField",
        "BooleanField",
        "SelectField",
        "IntegerField",
        "FloatField",
        "IntegerField",
    ]
    element_values = [
        50,
        True,
        [("KMeans", "KMeans"), ("Louvain", "Louvain"), ("Leiden", "Leiden")],
        10,
        1.0,
        15,
    ]
    return createSuperForm(elements, element_fields, element_values)


def getPSTSForm(trajectory, clusts, options):
    """Gets the psts form generated using superform above.

    Args:
            cluster_set (numpy.array<str>): The clusters which can be selected as
                                                                            the root for psts analysis.

    Returns:
            FlaskForm: With attributes that allow input related to psts.
    """
    elements = [
        "Root cluster",
        "Reverse",
        "eps (max. dist. spot neighbourhood)",
        "Trajectory Select",
        "Select distance-based method",
    ]
    element_fields = [
        "SelectField",
        "BooleanField",
        "IntegerField",
        "SelectField",
        "SelectField",
    ]

    element_values = [clusts, False, 50, trajectory, options]
    return createSuperForm(elements, element_fields, element_values)


def getDEAForm(list_labels, methods):
    """Gets the psts form generated using superform above.

    Args:
            cluster_set (numpy.array<str>): The clusters which can be selected as
                                                                            the root for psts analysis.

    Returns:
            FlaskForm: With attributes that allow input related to psts.
    """
    elements = ["Use label", "Use method"]
    element_fields = [
        "SelectField",
        "SelectField",
    ]

    element_values = [list_labels, methods]
    return createSuperForm(elements, element_fields, element_values)


######################## Junk Code #############################################
# def getCCIForm(step_log):
# 	""" Gets the CCI form generated from the superform above.
#
# 	Returns:
# 		FlaskForm: With attributes that allow for inputs that are related to
# 					CCI analysis.
# 	"""
# 	elements, element_fields, element_values = [], [], []
# 	if type(step_log['cci_het']) == type(None):
# 		# Analysis type form version #
# 		analysis_elements = ['Cell Heterogeneity Information', # Title
# 							 'cci_het',
# 							 'Permutation Testing', # Title
# 							 'cci_perm']
# 		analysis_fields = ['Title', 'SelectField', 'Title', 'SelectField']
# 		label_transfer_options = ['Upload Cell Label Transfer',
# 								  'No Cell Label Transfer']
# 		permutation_options = ['With permutation testing',
# 							   'Without permutation testing']
# 		analysis_values = ['', label_transfer_options, '', permutation_options]
# 		elements += analysis_elements
# 		element_fields += analysis_fields
# 		element_values += analysis_values
#
# 	else:
# 		# Core elements regardless of CCI mode #
# 		elements += ['Neighbourhood distance',
# 					 'L-R pair input (e.g. L1_R1, L2_R2, ...)']
# 		element_fields += ['IntegerField', 'StringField']
# 		element_values += [5, '']
#
# 		if step_log['cci_perm']:
# 			# Including cell heterogeneity information #
# 			elements += ['Permutations']
# 			element_fields += ['IntegerField']
# 			element_values += [200]
#
# 	return createSuperForm(elements, element_fields, element_values, None)
