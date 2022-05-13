# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


here = os.path.dirname(__file__)
repo = os.path.join(
    here,
    "..",
)
sys.path.insert(0, repo)

try:
    import pendsim
except ImportError:
    raise ImportError("check that `pendsim` is available to your system path.")

# -- Project information -----------------------------------------------------

project = "pendsim"
copyright = "2021, Mike Sutherland"
author = "Mike Sutherland"

# The full version, including alpha/beta/rc tags
release = pendsim.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "nb2plots",
    "nbsphinx",
    "nbsphinx_link",
]


autoapi_type = "python"
autoapi_dirs = ["../pendsim/"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_mock_imports = ["scipy", "numpy", "matplotlib", "cvxpy", "filterpy"]
numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

nbsphinx_link_target_root = repo
