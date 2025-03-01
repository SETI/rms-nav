=============
Contributing
=============

Thank you for your interest in contributing to RMS-NAV! This document provides guidelines and instructions for contributing to the project.

Code of Conduct
--------------

We expect all contributors to follow our Code of Conduct, which ensures a welcoming and inclusive environment for everyone.

Getting Started
--------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/rms-nav.git
      cd rms-nav

3. Create a virtual environment and install dependencies:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt
      pip install -r requirements-dev.txt  # Development dependencies

4. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Development Workflow
------------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name
      # or
      git checkout -b bugfix/issue-number

2. Make your changes, following our coding standards
3. Write or update tests as necessary
4. Run the tests to ensure they pass:

   .. code-block:: bash

      pytest

5. Commit your changes with a descriptive message:

   .. code-block:: bash

      git commit -m "Add feature: description of your changes"

6. Push your branch to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. Open a Pull Request on GitHub

Coding Standards
--------------

We follow these standards for all code contributions:

* **Python Style**: Follow PEP 8
* **Type Hints**: Use type hints for all function parameters and return values
* **Docstrings**: Document all classes and methods with docstrings following the Google style
* **Testing**: Include unit tests for new functionality
* **Compatibility**: Ensure compatibility with Python 3.9+

Example of a well-formatted function:

.. code-block:: python

   def calculate_offset(image: NDArrayFloatType, model: NDArrayFloatType) -> Tuple[float, float]:
       """Calculate the offset between an image and a model.

       Args:
           image: The observed image as a NumPy array
           model: The theoretical model as a NumPy array

       Returns:
           A tuple containing the (u, v) offset in pixels
       """
       # Implementation here
       return u_offset, v_offset

Pull Request Process
------------------

1. Ensure all tests pass
2. Update documentation if necessary
3. Add an entry to the CHANGELOG.md file describing your changes
4. Make sure your code is properly formatted and passes all pre-commit checks
5. Request a review from a maintainer
6. Address any feedback from reviewers

The maintainers will merge your PR once it meets all requirements.

Testing
------

We use pytest for testing. To run the tests:

.. code-block:: bash

   pytest

For more verbose output:

.. code-block:: bash

   pytest -v

To run a specific test file:

.. code-block:: bash

   pytest tests/test_specific_file.py

Documentation
-----------

We use Sphinx for documentation. To build the docs:

.. code-block:: bash

   cd docs
   make html

The generated documentation will be in ``docs/_build/html``.

When adding new features, please update the relevant documentation:

* Update docstrings for new functions and classes
* Add examples if appropriate
* Update the user guide or developer guide if necessary

Reporting Issues
--------------

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the GitHub issue tracker
2. If not, create a new issue with:
   * A clear, descriptive title
   * A detailed description of the issue
   * Steps to reproduce (for bugs)
   * Your environment information (Python version, OS, etc.)
   * Any relevant logs or screenshots

Thank you for contributing to RMS-NAV!