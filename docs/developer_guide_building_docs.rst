===================================================
Developer Guide: Building the Documentation
===================================================

Building the Documentation
===========================

Prerequisites
-------------

1. Install the required Python packages:

   .. code-block:: bash

      pip install -r requirements.txt

Building HTML Documentation
---------------------------

1. Navigate to the docs directory:

   .. code-block:: bash

      cd docs

2. Build the HTML documentation:

   .. code-block:: bash

      make html

3. The built documentation will be available in ``docs/_build/html``. Open ``index.html`` in your browser to view it.

Building Other Formats
----------------------

PDF (requires LaTeX):

.. code-block:: bash

   make latexpdf

Single HTML page:

.. code-block:: bash

   make singlehtml

EPUB:

.. code-block:: bash

   make epub

Working with Mermaid Diagrams
-----------------------------

Mermaid diagrams are rendered using the sphinxcontrib-mermaid extension. To create or modify diagrams:

1. Edit the Mermaid diagram code in the RST files
2. Run ``make html`` to build the documentation
3. Check the rendered diagram in the HTML output

Example Mermaid diagram syntax:

.. code-block:: rst

   .. mermaid::

      classDiagram
         class NavBase {
             +__init__(config, logger_name)
             +logger
             +config
         }
         class DataSet {
             <<abstract>>
             +__init__(config, logger_name)
             +image_name_valid(name)*
             +yield_image_filenames_from_arguments(args)*
         }
         NavBase <|-- DataSet

Updating API Documentation
--------------------------

The API documentation is automatically generated from docstrings in the code. To update it:

1. Ensure your code has proper docstrings.
2. Run ``make html`` to rebuild the documentation.

If you add new modules, you may need to update ``api_reference.rst`` to include them.

Troubleshooting
---------------

If you encounter issues with the documentation build:

1. Ensure all required packages are installed
2. Check for syntax errors in RST files
3. Look for error messages in the build output
4. Clear the build directory (``rm -rf _build``) and try again

For Mermaid diagram issues:

1. Validate your Mermaid syntax using the online Mermaid Live Editor: https://mermaid.live/
2. Ensure the sphinxcontrib-mermaid extension is properly installed and configured
