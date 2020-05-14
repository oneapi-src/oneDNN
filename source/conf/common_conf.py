import os
import sys
import string

def path_relative_to_repo_root(relative_path):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, '../..'))
    return os.path.abspath(os.path.join(root_dir, relative_path))

# oneDAL uses custom API generator based on `breathe`.
# Extend path to let Sphinx find `dalapi` module:
sys.path.insert(0, path_relative_to_repo_root('source/elements/oneDAL'))

extensions = [
    'notfound.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.imgconverter',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinxcontrib.spelling',
    'sphinx_substitution_extensions',
    'breathe',
    'dalapi', # oneDAL API generator
]

env = {
    'oneapi_version': '0.7',
    'l0_version': '0.91',
}

prolog_template = string.Template("""
.. |dpcpp_full_name| replace:: oneAPI Data Parallel C++
.. |dpcpp_version| replace:: $oneapi_version
.. |dpl_full_name| replace:: oneAPI DPC++ Library
.. |dpl_version| replace:: $oneapi_version
.. |ccl_full_name| replace:: oneAPI Collective Communications Library
.. |ccl_version| replace:: $oneapi_version
.. |dal_full_name| replace:: oneAPI Data Analytics Library
.. |dal_short_name| replace:: oneDAL
.. |dal_version| replace:: $oneapi_version
.. |dal_namespace| replace:: daal
.. |dnn_full_name| replace:: oneAPI Deep Neural Network Library
.. |dnn_version| replace:: $oneapi_version
.. |l0_full_name| replace:: oneAPI Level Zero
.. |l0_version| replace:: $l0_version
.. |tbb_full_name| replace:: oneAPI Threading Building Blocks
.. |tbb_version| replace:: $oneapi_version
.. |vpl_full_name| replace:: oneAPI Video Processing Library
.. |vpl_version| replace:: $oneapi_version
.. |mkl_full_name| replace:: oneAPI Math Kernel Library
.. |mkl_version| replace:: $oneapi_version
.. _`Level Zero Specification`: https://spec.oneapi.com/versions/$oneapi_version/oneL0/index.html
""")

rst_prolog = prolog_template.substitute(env)


# for substitutions in code blocks and sphinx-prompts:
substitutions = [
    ('|dal_short_name|', 'oneDAL'),
    ('|daal_in_code|', 'daal')
    ]


primary_domain = 'cpp'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': '\\DeclareUnicodeCharacter{2208}{$\\in$}',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'extraclassoptions': 'openany,oneside'
}

def setup(app):
    add_custom_css = getattr(app,'add_css_file',getattr(app,'add_stylesheet'))
    add_custom_css('custom.css')
