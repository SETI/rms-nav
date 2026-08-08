"""Tests that no SQLite-only construct survives in the index query source.

Four spellings work on SQLite and fail, or silently mean something else, on
PostgreSQL: a Python function registered on the connection and then called from
SQL, the ``TOTAL`` aggregate, a script of statements run outside the DDL layer,
and integer arithmetic on a boolean. None of them announces itself; each is
found the first time somebody points the code at a server. These read the source
so that the answer does not depend on which backend a test happened to run
against.

The scan covers every package that holds index queries: the Core layer itself,
and the statistics programs, whose report issues most of the SQL in the system.
A scan aimed at one package while the queries live in another reports green
without reading them.

It reads the source rather than importing it, so a module added to either
package tomorrow is covered without being named here.
"""

import ast
import re
from pathlib import Path

import pytest
import sqlalchemy
from filecache import FCPath

from spindoctor.results_index import METADATA

_SRC = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

_INDEX_ROOT = _SRC / 'results_index'
"""The Core layer: the schema, the opener, and the root bookkeeping."""

_SOURCE_ROOTS = (_INDEX_ROOT, _SRC / 'cli' / 'stats')
"""Every package holding index queries."""

# The connect-time events a PRAGMA may legitimately live inside.  A pragma in a
# query reaches one connection out of however many the pool holds.
_CONNECT_EVENT = 'connect'

_PRAGMA_RE = re.compile(r'\bpragma\b', re.IGNORECASE)

_TOTAL_RE = re.compile(r'\bTOTAL\s*\(', re.IGNORECASE)

# A Markdown table row, which the report emits by the hundred and which is
# never SQL.  One of them is a run-time table headed "total (s)", which the
# aggregate pattern would otherwise read as the SQLite spelling of a sum.
_MARKDOWN_ROW_RE = re.compile(r'^\s*\|')

# Registering a Python callable as a SQL function, in either DBAPI's spelling.
_UDF_REGISTRARS = frozenset({'create_function', 'register_function'})

_SCRIPT_EXECUTORS = frozenset({'executescript'})

_KNOWN_MODULES = frozenset(
    {
        # The Core layer.
        'engine.py',
        'roots.py',
        'schema.py',
        # The statistics programs, where the report's queries live.
        'classify.py',
        'ingest.py',
        'ingest_rows.py',
        'report.py',
        'report_common.py',
        'report_sections.py',
    }
)
"""The modules the scan must reach, whatever else either package grows."""


def _source_files() -> list[FCPath]:
    """Return every Python module of the packages under scan.

    Returns:
        The modules, sorted by path.
    """
    return sorted(path for root in _SOURCE_ROOTS for path in root.rglob('*.py'))


def _module_ids() -> list[str]:
    """Return the module names used as parametrization ids.

    Returns:
        One base name per module under scan.
    """
    return [path.name for path in _source_files()]


def _string_constants(tree: ast.AST) -> list[ast.Constant]:
    """Return every string constant in a parsed module.

    Parameters:
        tree: The parsed module.

    Returns:
        The string constant nodes, which carry line numbers.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _query_strings(tree: ast.AST) -> list[ast.Constant]:
    """Return every string constant that could be part of a statement.

    Markdown table rows are excluded.  The report writes tables as literal
    rows, and a column headed "total (s)" reads exactly like the SQLite
    aggregate; nothing beginning with a table separator is ever SQL.

    Parameters:
        tree: The parsed module.

    Returns:
        The string constant nodes, which carry line numbers.
    """
    return [node for node in _string_constants(tree) if not _MARKDOWN_ROW_RE.match(str(node.value))]


def _call_argument_strings(tree: ast.AST) -> list[ast.Constant]:
    """Return every string constant a parsed module passes to a call.

    A pragma reaches a database only as the argument of an execute call, and the
    word itself is ordinary English: scanning every constant would fail on a
    docstring that merely described one.  Formatted strings are descended into,
    since that is how a pragma carrying a value is written.

    Parameters:
        tree: The parsed module.

    Returns:
        The string constant nodes, which carry line numbers.
    """
    arguments = [
        argument
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
    ]
    return [node for argument in arguments for node in _string_constants(argument)]


def _connect_handler_names(tree: ast.AST) -> set[str]:
    """Return the names of functions registered as connect-time events.

    Both spellings are recognized: ``event.listen(target, 'connect', handler)``
    and the ``@event.listens_for(target, 'connect')`` decorator.

    Parameters:
        tree: The parsed module.

    Returns:
        The handler function names.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _called_name(node) == 'listen'
            and len(node.args) >= 3
            and _is_string(node.args[1], _CONNECT_EVENT)
            and isinstance(node.args[2], ast.Name)
        ):
            names.add(node.args[2].id)
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and _called_name(decorator) == 'listens_for'
                    and len(decorator.args) >= 2
                    and _is_string(decorator.args[1], _CONNECT_EVENT)
                ):
                    names.add(node.name)
    return names


def _called_name(node: ast.Call) -> str | None:
    """Return the final attribute or name a call invokes.

    Parameters:
        node: The call node.

    Returns:
        The name, or None when the call target is an expression.
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _is_string(node: ast.expr, value: str) -> bool:
    """Whether an argument node is exactly the given string literal.

    Parameters:
        node: The argument node.
        value: The literal to compare against.

    Returns:
        True when the node is that literal.
    """
    return isinstance(node, ast.Constant) and node.value == value


def _connect_handler_line_ranges(tree: ast.AST) -> list[range]:
    """Return the line ranges of every connect-time event handler.

    Parameters:
        tree: The parsed module.

    Returns:
        One range per handler, covering its whole body.
    """
    handlers = _connect_handler_names(tree)
    return [
        range(node.lineno, (node.end_lineno or node.lineno) + 1)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in handlers
    ]


def _called_names(tree: ast.AST) -> set[str]:
    """Return every function or method name the module calls.

    Parameters:
        tree: The parsed module.

    Returns:
        The called names.
    """
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [_called_name(node)]
        if name is not None
    }


def _boolean_column_names() -> list[str]:
    """Return every Boolean column name in the schema.

    Returns:
        The names, sorted and de-duplicated.
    """
    return sorted(
        {
            column.name
            for table in METADATA.tables.values()
            for column in table.columns
            if isinstance(column.type, sqlalchemy.Boolean)
        }
    )


@pytest.mark.parametrize('path', _source_files(), ids=_module_ids())
def test_no_module_registers_a_python_function_as_sql(path: FCPath) -> None:
    """A UDF exists only on the connection that registered it.

    Parameters:
        path: The module under scan.
    """
    offenders = sorted(_called_names(ast.parse(path.read_text())) & _UDF_REGISTRARS)
    assert offenders == []


@pytest.mark.parametrize('path', _source_files(), ids=_module_ids())
def test_no_module_uses_the_total_aggregate(path: FCPath) -> None:
    """``TOTAL`` is a SQLite spelling of ``COALESCE(SUM(...), 0)``.

    Parameters:
        path: The module under scan.
    """
    offenders = [
        node.value
        for node in _query_strings(ast.parse(path.read_text()))
        if _TOTAL_RE.search(str(node.value))
    ]
    assert offenders == []


@pytest.mark.parametrize('path', _source_files(), ids=_module_ids())
def test_no_module_runs_a_statement_script(path: FCPath) -> None:
    """DDL comes from the metadata, so no dialect can be baked into a script.

    Parameters:
        path: The module under scan.
    """
    offenders = sorted(_called_names(ast.parse(path.read_text())) & _SCRIPT_EXECUTORS)
    assert offenders == []


@pytest.mark.parametrize('path', _source_files(), ids=_module_ids())
def test_every_pragma_lives_inside_a_connect_event(path: FCPath) -> None:
    """A pragma issued as a query configures one connection out of a pool.

    Parameters:
        path: The module under scan.
    """
    tree = ast.parse(path.read_text())
    allowed = _connect_handler_line_ranges(tree)
    offenders = [
        node.value
        for node in _call_argument_strings(tree)
        if _PRAGMA_RE.search(str(node.value))
        and not any(node.lineno in line_range for line_range in allowed)
    ]
    assert offenders == []


@pytest.mark.parametrize('path', _source_files(), ids=_module_ids())
def test_no_module_compares_a_boolean_column_against_an_integer(path: FCPath) -> None:
    """``spurious = 0`` and ``1 - spurious`` are type errors on PostgreSQL.

    Parameters:
        path: The module under scan.
    """
    patterns = [
        re.compile(rf'\b{re.escape(name)}\b\s*(=|==|<>|!=|<|>|\+|-|\*)\s*\d')
        for name in _boolean_column_names()
    ] + [
        re.compile(rf'\d\s*(=|==|<>|!=|<|>|\+|-|\*)\s*\b{re.escape(name)}\b')
        for name in _boolean_column_names()
    ]
    offenders = [
        node.value
        for node in _query_strings(ast.parse(path.read_text()))
        if any(pattern.search(str(node.value)) for pattern in patterns)
    ]
    assert offenders == []


def test_the_scan_actually_reads_some_modules() -> None:
    """Every assertion above is that a search came back empty.

    A path that names nothing is indistinguishable from a package that is clean,
    so the scan's own reach is asserted.  It is asserted as a floor rather than
    as an inventory, because the scan finds the packages' modules for itself and
    one added tomorrow is meant to be covered without being named here.
    """
    missing = sorted(_KNOWN_MODULES.difference(_module_ids()))
    assert missing == []


def test_the_scan_reaches_the_statistics_programs() -> None:
    """The report issues most of the SQL in the system, so it is scanned too.

    Named as a directory rather than as a module list: a scan that had drifted
    back to the Core layer alone would still find every module it named there.
    """
    reached = {path.as_posix() for path in _source_files()}
    assert any('/cli/stats/' in path for path in reached)


def test_the_markdown_exclusion_leaves_a_statement_alone() -> None:
    """Only a table row is excused, and a statement is not one.

    The exclusion exists for a run-time table headed "total (s)"; if it grew to
    cover statements, every check that reads a string would quietly empty.
    """
    tree = ast.parse("x = ['| total (s) |', 'SELECT TOTAL(n) FROM t']")
    kept = [str(node.value) for node in _query_strings(tree)]
    assert kept == ['SELECT TOTAL(n) FROM t']


def test_the_boolean_scan_knows_which_columns_are_boolean() -> None:
    """The boolean check is driven by the schema, so it cannot silently empty."""
    assert _boolean_column_names() == ['at_edge', 'has_summary_png', 'spurious']


def test_the_pragma_scan_finds_the_connect_handler() -> None:
    """The pragma allowance depends on recognizing the event registration.

    If the recognizer stopped matching, every pragma would read as an offender
    rather than the check quietly passing -- but the reverse (a handler matched
    by accident, allowing pragmas anywhere) is what this pins.
    """
    tree = ast.parse((_INDEX_ROOT / 'engine.py').read_text())
    assert _connect_handler_names(tree) == {'_sqlite_on_connect'}
