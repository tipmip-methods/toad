import inspect  # needed in linkcode py-function to connect documentation to source link
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, module_path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "toad"
copyright = "2024, Lukas Röhrich"
author = "Lukas Röhrich"
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    #'sphinx.ext.viewcode',
    "myst_nb",  # allows to include Jupyter Notebooks and Markdowns
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 2  # depth of implicit target for cross references -> needed for git_version_control.rst

nb_execution_mode = "off"  # Prevent myst_nb from executing notebooks

templates_path = ["_templates"]
exclude_patterns = []
master_doc = "sidebar_main_nav_links"
modindex_common_prefix = ["toad."]  # ignored prefixes for module index sorting


remove_from_toctrees = [
    "generated/*"
]  # remove generated files from the table of contents, this folder is created by the sphinx-apidoc command


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static", "resources"]
html_css_files = [
    "custom.css",
]

# -> Theme Specific HTML ouptut options
html_sidebars = {"**": ["sidebar_main_nav_links.html", "sidebar_toc.html"]}
html_logo = "resources/toad.png"
html_permalinks_icon = "<span>¶</span>"  # change the default anchor icon, paragraph mark only appears when hovering over the heading

# -- Function for sphinx extention linkcode -------------------------------

# Look at xarray documentation for a more elaborate example.


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object.
    The implementation is based on xarray doc/source/conf.py.
    """
    if domain != "py":
        print(f"\nWARNING: Domain is not 'py'. Object: {domain}")
        return None
    if not info["module"]:
        print(f"\nWARNING: Module is not specified. Object: {info}")
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        print(f"\nWARNING: Could not find module. Object: {submod}")
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            print(f"\nWARNING: Could not find attribute. Object: {obj}")
            return None

    # **Skip properties to avoid the error**
    if isinstance(obj, property):
        return None

    try:
        # Get the source file and line numbers
        sourcefile = inspect.getsourcefile(obj)
        if sourcefile is None:
            print(f"\nWARNING: No source file found. Object: {sourcefile}")
            return None

        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    # identify start and end line number of code in source file
    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    # Adjust for objects imported into __init__.py
    # Use the actual source file instead of relying on the module name
    relpath = os.path.relpath(
        sourcefile, start=os.path.dirname(sys.modules["toad"].__file__)
    )

    # Build the GitHub URL
    return f"https://github.com/tipmip-methods/toad/tree/main/toad/{relpath}{linespec}"