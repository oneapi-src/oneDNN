extensions = [
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
    'sphinx-prompt',
    'sphinx_substitution_extensions',
    'breathe',
]

env = {
    'llga_version': '0.2',
}

primary_domain = 'cpp'

def setup(app):
    add_custom_css = getattr(app,'add_css_file',getattr(app,'add_stylesheet'))
    add_custom_css('custom.css')
