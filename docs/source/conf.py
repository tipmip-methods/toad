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
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    # "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc.typehints",  # to pull types from function definitions
    #'sphinx.ext.viewcode',
    "myst_nb",  # allows to include Jupyter Notebooks and Markdowns
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,  # Exclude special methods like __init__, __repr__, etc.
    "exclude-members": "__weakref__,__dict__,__module__,__doc__,__slots__",  # Exclude problematic attributes
    "show-signature": True,
    "show-signature-with-docstring": False,  # Don't show docstring in signature for dataclasses
}
autosummary_generate = True
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 2  # depth of implicit target for cross references -> needed for git_version_control.rst

nb_execution_mode = "off"  # Prevent myst_nb from executing notebooks

# Autodoc type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_include_ivar_with_doc = False  # Don't show instance variables for dataclasses

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

# Other themes to try in the future maybe:
# html_theme = 'furo'
# html_theme = 'sphinx_book_theme'

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

    # **Skip dataclass instances to avoid the error**
    if hasattr(obj, "__class__") and hasattr(obj.__class__, "__dataclass_fields__"):
        return None

    # **Skip Numba JIT-compiled functions to avoid CPUDispatcher error**
    if hasattr(obj, "__class__") and "CPUDispatcher" in str(type(obj)):
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

    # Check if toad module file exists and is not None
    toad_module = sys.modules.get("toad")
    if toad_module is None or toad_module.__file__ is None:
        print("\nWARNING: Could not find toad module or its __file__ attribute")
        return None

    # Ensure sourcefile is not None before using it
    if sourcefile is None:
        print("\nWARNING: No source file found for object")
        return None

    # Adjust for objects imported into __init__.py
    # Use the actual source file instead of relying on the module name
    relpath = os.path.relpath(sourcefile, start=os.path.dirname(toad_module.__file__))

    # Build the GitHub URL
    return f"https://github.com/tipmip-methods/toad/tree/main/toad/{relpath}{linespec}"
