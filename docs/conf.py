#!/usr/bin/env python3

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'SpinDoctor'
copyright = '2025, SETI Institute'
author = 'SETI Institute'

# The full version, including alpha/beta/rc tags
try:
    release = importlib.metadata.version('rms-spindoctor')
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
# The simulator image galleries carry a NOTES.md regeneration note alongside the
# committed PNG assets; exclude them so Sphinx does not treat them as orphan docs.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/_sim_images/NOTES.md',
    '**/_scene_images/NOTES.md',
]

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Show every section level in the sidebar TOC, with all sub-trees expanded.
html_theme_options = {
    'navigation_depth': -1,
    'collapse_navigation': False,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False
autodoc_typehints_format = 'short'
# Mock PyQt6 and matplotlib Qt backends so autodoc can import spindoctor.ui modules
# without a display or OpenGL context (e.g. in CI).
autodoc_mock_imports = [
    'PyQt6',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_qt',
]

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
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'filecache': ('https://rms-filecache.readthedocs.io/en/latest/', None),
    'pdslogger': ('https://rms-pdslogger.readthedocs.io/en/latest/', None),
}

# Suppress nitpicky warnings for symbols that have no inventory we can link to:
# third-party packages without Sphinx docs (oops), test modules excluded from
# autodoc, sibling packages outside the importable spindoctor API surface, typing
# internals leaked by autodoc, and TypeVars / unqualified type aliases that
# Sphinx does not register as cross-reference targets.
nitpick_ignore_regex = [
    (r'py:.*', r'oops\..*'),
    (r'py:.*', r'tests\..*'),
    (r'py:.*', r'spindoctor\.cli\.backplanes\..*'),
    (r'py:.*', r'spindoctor\.cli\.pds4\..*'),
    (r'py:.*', r'spindoctor\.cli\.reproj\..*'),
    (r'py:.*', r'numpy\._typing\..*'),
    (r'py:.*', r'argparse\._.*'),
    (r'py:.*', r'spindoctor\.support\.types\.NPType'),
    (r'py:.*', r'spindoctor\.ui\.mosaic_viewer\..*'),
]

# MyST-Parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Mermaid settings: use default client-side rendering (no mmdc/Chromium) so docs
# build in headless CI; diagrams render in the browser via mermaid.js.
mermaid_d3_zoom = True
