"""Which schema of a database the results index's own tables live in.

A server resolves an unqualified table name through a search path that may reach
several schemas, so "where the index is" is a question about a schema rather
than about a database.  The creating open and the drop both have to answer it,
and they have to answer it the same way: what one builds is what the other
removes, and a disagreement between them is a drop aimed at tables nobody
created.

**The stamp names the schema.**  ``schema_meta`` is the one table of the index
whose shape says something.  Two of its columns are the marks: a version column
is what any migration table carries, and a constant-keyed ``singleton`` beside
it is what this one does, so the pair together is idiosyncratic where either
alone is not.  The schema a bare ``schema_meta`` resolves to is therefore the
schema the index lives in, and a database that reaches no such table holds no
index of ours at all.

**Two marks, not the whole column set.**  A database stamped by a schema version
whose columns differ from this one's is exactly what a drop is pointed at, and
requiring today's columns as evidence would withhold the drop from the case it
exists for.  The marks are what a stamp of any version carries.

**A schema a fresh table lands in is a different question**, and only a creating
open asks it: a database with no stamp anywhere has no schema resolved for it,
and the index is then built in the schema an unqualified ``CREATE TABLE`` would
write to.
"""

import sqlalchemy

from spindoctor.results_index.schema import METADATA, SCHEMA_META

__all__ = [
    'INDEX_TABLE_NAMES',
    'STAMP_MARKS',
    'carries_the_stamp_marks',
    'creation_schema',
    'index_tables_in',
    'relations_in',
    'resolved_schema',
]

INDEX_TABLE_NAMES = frozenset(table.name for table in METADATA.sorted_tables)
"""Every table name the index owns, and the whole of what belongs to it.

A schema holding a relation not named here holds something SpinDoctor did not
create, whoever else created it and whatever it is for.
"""

STAMP_MARKS = ('singleton', 'schema_version')
"""Columns whose presence in a ``schema_meta`` say SpinDoctor wrote it.

Two rather than the whole column set, because a stamp left by a schema whose
columns differed from this one's is exactly the database a drop is pointed at,
and requiring today's columns would withhold the drop from it.  Two rather than
one, because the pair is idiosyncratic: a version column is what any migration
table carries, and a constant-keyed ``singleton`` beside it is what this one
does.
"""

_POSTGRES_DIALECT = 'postgresql'
"""Dialect name of every PostgreSQL driver, whichever one the URL asked for."""

_VISIBLE_SCHEMA_SQL = """
SELECT n.nspname
  FROM pg_catalog.pg_class c
  JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relname = :name
   AND c.relkind IN ('r', 'p', 'f')
   AND pg_catalog.pg_table_is_visible(c.oid)
"""
"""Which schema an unqualified table name resolves to on this connection.

PostgreSQL resolves a bare name through the search path, and the catalog is the
only thing that can say where it landed.  ``pg_table_is_visible`` is that rule
itself, so at most one row comes back: the table the server would have reached,
in the schema it lives in.
"""

_CREATION_SCHEMA_SQL = 'SELECT current_schema()'
"""Which schema an unqualified ``CREATE TABLE`` writes to on this connection.

The first schema of the search path that exists.  A path naming none that exist
answers with nothing, which is a connection no table can be created through.
"""


def resolved_schema(connection: sqlalchemy.Connection) -> str | None:
    """Return the schema an unqualified ``schema_meta`` resolves to.

    The same resolution the rest of the system's statements get, asked once and
    then written down, rather than repeated per table where two tables of these
    names in two schemas would answer differently.

    Only a table answers, never a view of that name: what is being looked for is
    the object a ``CREATE TABLE`` of the index's would have found already there.

    Parameters:
        connection: An open connection to the database.

    Returns:
        The schema name, or None when this connection reaches no ``schema_meta``
        at all.
    """
    if connection.dialect.name == _POSTGRES_DIALECT:
        found = connection.execute(
            sqlalchemy.text(_VISIBLE_SCHEMA_SQL), {'name': SCHEMA_META.name}
        ).scalar()
        return None if found is None else str(found)
    # Every other backend this supports has one namespace per database, which
    # the inspector names: "main" for SQLite.  Naming it rather than leaving it
    # implicit is what keeps the two backends on one code path.
    inspector = sqlalchemy.inspect(connection)
    schema = inspector.default_schema_name
    if schema is None or SCHEMA_META.name not in inspector.get_table_names(schema=schema):
        return None
    return schema


def creation_schema(connection: sqlalchemy.Connection) -> str | None:
    """Return the schema a table created without a schema name lands in.

    Asked of a database that reaches no stamp of ours, which is the database an
    index is about to be built in.  Naming that schema is what keeps the six
    tables together in one of them: a name resolved through the search path is
    free to find a table of that name somewhere else and be built over it, and a
    name qualified with the schema a fresh table lands in is not.

    Parameters:
        connection: An open connection to the database.

    Returns:
        The schema name, or None when the connection's search path names no
        schema that exists, which is a connection nothing can be created
        through.
    """
    if connection.dialect.name == _POSTGRES_DIALECT:
        found = connection.execute(sqlalchemy.text(_CREATION_SCHEMA_SQL)).scalar()
        return None if found is None else str(found)
    schema = sqlalchemy.inspect(connection).default_schema_name
    return None if schema is None else str(schema)


def carries_the_stamp_marks(connection: sqlalchemy.Connection, schema: str) -> bool:
    """Report whether a schema's ``schema_meta`` carries SpinDoctor's own marks.

    Parameters:
        connection: An open connection to the database.
        schema: The schema whose ``schema_meta`` is to be examined.

    Returns:
        True when that schema holds a ``schema_meta`` carrying every column of
        :data:`STAMP_MARKS`, whatever else it carries and whatever version it
        is stamped with.  False when it holds no such table, or one carrying
        only some of the marks, which is what any table of that name is free to
        carry.
    """
    inspector = sqlalchemy.inspect(connection)
    if SCHEMA_META.name not in inspector.get_table_names(schema=schema):
        return False
    columns = {column['name'] for column in inspector.get_columns(SCHEMA_META.name, schema=schema)}
    return columns.issuperset(STAMP_MARKS)


def index_tables_in(schema: str) -> dict[str, sqlalchemy.Table]:
    """Return every table the index owns, named in one schema explicitly.

    The schema is rendered into every statement built from these, so nothing
    built from them can resolve through a search path onto a table in another
    schema.  Every table of the answer belongs to one metadata container, which
    is what creates them together and in dependency order.

    Parameters:
        schema: The schema the tables are to be named in.

    Returns:
        The tables, keyed by name.
    """
    metadata = sqlalchemy.MetaData(schema=schema)
    return {table.name: table.to_metadata(metadata) for table in METADATA.sorted_tables}


def relations_in(connection: sqlalchemy.Connection, schema: str) -> tuple[str, ...]:
    """Return the name of every table and view one schema holds.

    Tables and views together, because both are what an unqualified name
    resolves to and what a ``CREATE TABLE`` of that name finds already there.
    The indexes, sequences and constraints the index's own tables bring with
    them are not relations of this kind and are never named here, so a schema
    holding exactly one index answers with exactly its six tables.

    Parameters:
        connection: An open connection to the database.
        schema: The schema to enumerate.

    Returns:
        The names, sorted, and empty for a schema holding neither.
    """
    inspector = sqlalchemy.inspect(connection)
    tables = inspector.get_table_names(schema=schema)
    views = inspector.get_view_names(schema=schema)
    return tuple(sorted(set(tables) | set(views)))
