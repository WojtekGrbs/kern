"""Sphinx configuration for kern."""

from importlib.metadata import version as package_version

project = "kern"
author = "Wojciech Grabias"
release = package_version("kern")

extensions = [
    "sphinx.ext.autodoc",   
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
