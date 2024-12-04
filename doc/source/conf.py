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

project = 'toad'
copyright = '2024, Lukas Röhrich'
author = 'Lukas Röhrich'
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.doctest',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 2		# depth of implicit target for cross references -> needed for git_version_control.rst

templates_path = ['_templates']
exclude_patterns = []
master_doc = 'sidebar_main_nav_links'
modindex_common_prefix = ['toad.']	# ignored prefixes for module index sorting


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static','resources']

# -> Theme Specific HTML ouptut options
html_sidebars = {
  "**": ['sidebar_main_nav_links.html', 'sidebar_toc.html']
}
html_logo = 'resources/toad.png'
html_permalinks_icon = "<span>¶</span>"                     # change the default anchor icon, paragraph mark only appears when hovering over the heading

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