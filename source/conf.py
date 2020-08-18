# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('conf'))
# element_conf needs to import this conf
sys.path.insert(0, os.path.abspath('.'))

project = 'oneDNN Graph Library'

from element_conf import *

html_js_files = ['custom.js']
html_static_path = ['_static']
templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
html_favicon = '_static/favicons.png'
html_logo = '_static/oneAPI-rgb-rev-100.png'

htmlhelp_basename = 'oneAPI-spec'

html_theme_options = {
    'includehidden': False,
    'collapse_navigation': False
}

html_context = {
    'display_github': True,
    'github_host': 'gitlab.devtools.intel.com/',
    'github_user': 'llga',
    'github_repo': 'llga-spec',
    'github_version': 'dszwicht/spec_migration/source/'
}

pygments_style = None

exclude_patterns = []

latex_documents = []

breathe_projects = {
    "oneDNN Graph Library": "../doxygen/xml"
}
breathe_default_project = "oneDNN Graph Library"
