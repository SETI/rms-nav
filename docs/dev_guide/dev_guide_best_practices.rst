==============
Best Practices
==============

Project conventions for all new and modified code live in
``.cursor/rules/python.mdc`` (with peer files for
``dependency_management``, ``doc_python``, ``environment``,
``git_workflow``, ``pull_request``, ``python_testing``, and
``security``). That file is the
authoritative standard; this page lists the rules that come up most often.

Code style
----------

* Maximum line length: 100 characters (enforced by Ruff).
* Naming: ``lowercase_with_underscores`` for functions and local variables,
  ``TitleCase`` for classes, ``ALL_CAPS_WITH_UNDERSCORES`` for module-level
  constants. Prefix non-public names with a single underscore.
* Type-annotate every function parameter and return value, including
  ``-> None``. Use modern ``list[str]``, ``dict[str, int]``, ``X | None``
  syntax.
* Keep modules under 1000 lines. Split larger modules into a package.
* Do not introduce compatibility shims for prior versions unless explicitly
  requested; change the code instead.

Imports
-------

* Imports go at the top of the file, in three alphabetically sorted groups
  separated by blank lines: standard library, third party, then this project.
* An import inside a function is permitted only to keep a heavy dependency off
  a path that does not use it. Two such exceptions exist, and each carries a
  comment saying why:

  * The PyQt6 widgets in ``spindoctor.ui``, so that a headless navigation run
    never imports a GUI toolkit.
  * :mod:`spindoctor.results_index.selection` in
    ``spindoctor/dataset/results_filter.py``, imported inside the branch that
    was given a results-index URL. Every navigation run imports
    ``spindoctor.dataset``, and most name no index, so the top-level import
    would put SQLAlchemy on the navigation critical path for all of them. A
    test asserts in a subprocess that importing ``spindoctor.dataset`` imports
    no ``sqlalchemy`` module.

Linting and typing
------------------

* Run ``ruff check src tests`` and ``ruff format --check src tests`` on the
  full codebase after changes.
* Run ``mypy src tests`` and fix every error. Do not add module-level
  ``# mypy: ignore-errors`` or global ``exclude`` entries. A line-level
  ``# type: ignore[error-code]`` is acceptable only with a brief
  justification.

Testing
-------

* Use ``pytest`` with ``pytest-xdist``; the canonical command is
  ``pytest -n auto --dist=loadfile`` (the ``--dist=loadfile`` flag is
  required because PyQt6 workers crash under default xdist scheduling).
* Annotate test function parameters and return types; return ``-> None``.
* One assertion per condition (no ``and`` in assertions). When testing
  exceptions, use ``pytest.raises`` as a context manager and assert on the
  exception message content via ``match=``.
* Target at least 90 % line coverage over the full suite.

Documentation
-------------

* Every module, class, function, and method has a docstring written in
  Google style with ``Parameters:``, ``Returns:``, and ``Raises:`` as
  needed. Wrap docstring text to 90 characters.
* Do not use smart quotes, em-dashes, or arrows in ``.py`` files (they are
  fine in ``.rst`` and ``.md``).
* Update docstrings when the associated code changes, and remove them when
  the code is removed.
* After a code change, run ``sphinx-build -W -b html docs docs/_build`` and
  fix every warning.
