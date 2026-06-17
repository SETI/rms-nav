==============
Best Practices
==============

Project conventions for all new and modified code live in
``.cursor/rules/python.mdc`` (with peer files for
``dependency_management``, ``doc_python``, ``environment``,
``git_workflow``, ``pull_request``, and ``security``). That file is the
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
