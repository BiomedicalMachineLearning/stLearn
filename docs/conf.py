import os
import sys
sys.path.insert(0, os.path.abspath(".."))
import stlearn

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'stLearn'
copyright = '2022-2025, Genomics and Machine Learning Lab'
author = 'Genomics and Machine Learning Lab'
release = stlearn.__version__
html_logo = "images/logo.png"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Configure nbsphinx
nbsphinx_execute = 'never'  # Don't re-execute notebooks
nbsphinx_allow_errors = True  # Allow notebooks with errors