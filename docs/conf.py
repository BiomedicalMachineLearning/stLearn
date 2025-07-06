import os
import sys
import re
import requests

sys.path.insert(0, os.path.abspath(".."))
import stlearn

def download_gdrive_file(file_id, filename):
    session = requests.Session()
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    response = session.get(url)

    form_action_match = re.search(r'action="([^"]+)"', response.text)
    if not form_action_match:
        raise Exception("Could not find form action URL")
    download_url = form_action_match.group(1)

    params = {}
    hidden_inputs = re.findall(
        r'<input type="hidden" name="([^"]+)" value="([^"]*)"', response.text)
    for name, value in hidden_inputs:
        params[name] = value

    response = session.get(download_url, params=params, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

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
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
def setup(app):
    if not os.path.isdir("./tutorials"):
        download_gdrive_file("1FNMzO4-KsHK8tPd8k5-sTiRS40S97Qs4", "tutorials.zip")
        os.system("unzip tutorials.zip")
    return

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# Configure nbsphinx
nbsphinx_execute = 'never'  # Don't re-execute notebooks
nbsphinx_allow_errors = True  # Allow notebooks with errors

# Autosummary
autosummary_generate = True
autosummary_imported_members = True

# Output directory for autosummary
autosummary_generate_overwrite = True

autodoc_mock_imports = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'scanpy', 'anndata',
    'matplotlib', 'seaborn', 'plotly', 'bokeh', 'cv2', 'PIL',
    'rpy2', 'louvain', 'numba', 'leidenalg',
    # Add any other packages causing import issues
]