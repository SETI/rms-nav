#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'RMS-NAV'
copyright = '2025, SETI Institute'
author = 'SETI Institute'

# The full version, including alpha/beta/rc tags
try:
    release = importlib.metadata.version('rms-nav')
except importlib.metadata.PackageNotFoundError:
    release = '1.0.0'  # fallback for development

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False
autodoc_typehints_format = 'short'
# Mock PyQt6 so autodoc can import nav.ui modules without display/OpenGL (e.g. in CI).
autodoc_mock_imports = ['PyQt6']

# -- Extension configuration -------------------------------------------------

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
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MyST-Parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Mermaid settings
mermaid_output_format = 'svg'
mermaid_d3_zoom = True
