import os
import sys
module_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../toad'))
sys.path.insert(0, module_path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TOAD'
copyright = '2023, Sina Loriani'
author = 'Sina Loriani'
release = '0.2'
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.doctest',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.viewcode',
]

templates_path = ['_templates']
master_doc = 'sitemap'		# master toctree document
exclude_patterns = []
pygments_style = 'sphinx'	# syntax highlighting
modindex_common_prefix = ['toad.']	# ignored prefixes for module index sorting


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'python_docs_theme'
#html_theme = 'classic'
html_static_path = ['_static']
html_sidebars = {
	'**': [
		'globaltoc.html',
		'searchbox.html',
		]
	}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'toaddoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'toad.tex', u'toad Documentation',
    u'Sinal Loriani and pyunicorn authors', 'manual', False),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = '_static/logo.png'

# If true, show URL addresses after external links.
latex_show_urls = 'inline'
