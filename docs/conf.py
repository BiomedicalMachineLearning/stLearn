import os
import sys
import re
import requests

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
html_theme_options = {
    "source_repository": "https://github.com/BiomedicalMachineLearning/stLearn/",
    "source_branch": "master",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/BiomedicalMachineLearning/stLearn/",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
            "class": "",
        },
    ],
}

# Configure nbsphinx
nbsphinx_execute = 'never'  # Don't re-execute notebooks
nbsphinx_allow_errors = True  # Allow notebooks with errors

nbsphinx_thumbnails = {
    "tutorials/working_with_scanpy": "_static/images/scanpy.png",
}

# Autosummary
autosummary_generate = True
autosummary_imported_members = True

# Output directory for autosummary
autosummary_generate_overwrite = True
