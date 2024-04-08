import inspect          # needed in linkcode py-function to connect documentation to source link
import os
import sys
module_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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
    'sphinx.ext.linkcode',
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
#html_css_files = ['default.css']       # specify path to custom .css file in html_static_path
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


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'toad', u'toad Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'toad', u'toad Documentation',
     author, 'toad', 'One line description of project.',
     'Miscellaneous'),
]


# -- Function for sphinx extention linkcode -------------------------------

# Look at xarray documentation for a more elaborate example.

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object.
    The implementation is based on xarray doc/source/conf.py.
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
                
    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    # identify start and end line number of code in source file
    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    filename = info['module'].replace('.', '/')
    return "https://gitlab.pik-potsdam.de/sinal/toad/-/blob/main/%s.py%s" % (filename,linespec)