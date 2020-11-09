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

from common_conf import *

project = 'oneDNN Graph Specification'

copyright = u'2020, Intel Corporation'
author = u'Intel'

version = env['llga_version']
release = version

html_js_files = ['custom.js']
html_static_path = ['_static']
templates_path = ['_templates']

html_title = '%s %s' % (project,env['llga_version'])
html_theme = 'sphinx_book_theme'
html_favicon = '_static/favicons.png'
html_logo = '_static/oneAPI-rgb-rev-100.png'

htmlhelp_basename = 'oneAPI-spec'

pygments_style = None

exclude_patterns = ['*.inc.rst']

latex_documents = []

breathe_projects = {
    "oneDNN Graph Library": "../doxygen/xml"
}
breathe_default_project = "oneDNN Graph Library"

# -- Options for spelling extension-------------------------------------------
spelling_warning = True
