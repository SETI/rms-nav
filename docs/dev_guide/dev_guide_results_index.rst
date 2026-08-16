=================
The Results Index
=================

Overview
========

The results index is an optional database derived from the navigation results
tree. One ingest pass reads every ``_metadata.json`` document under a results
root and writes one row per image; consumers that need a few fields per image
then read a row instead of downloading a document. It is not authoritative:
the documents are, and the index can be dropped and rebuilt at any time.

:doc:`/user_guide/user_guide_results_index` is the operator's account of it ---
when to build one, how programs are pointed at one, what it promises. This
chapter is about the code: how the layer is put together, what it guarantees
under concurrency, and what changing it costs.

The code divides in two. The library package
:mod:`spindoctor.results_index` owns the schema, the opener and the queries
consumers issue; it is imported by anything that reads an index. The ingest
lives under the command-line package, in
``spindoctor/cli/stats/ingest/``, because writing the index is a program an
operator runs rather than an API a consumer calls.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Responsibility
   * - :mod:`spindoctor.results_index.schema`
     - Every table, its columns and constraints, and ``SCHEMA_VERSION``.
   * - :mod:`spindoctor.results_index.engine`
     - ``open_index``: backend selection, SQLite settings, the version gate.
       ``open_database`` is the one opener that stops before the gate.
   * - :mod:`spindoctor.results_index.scope`
     - Which schema of a database the index lives in, decided from the stamp.
   * - :mod:`spindoctor.results_index.masking`
     - The one rule that hides a password in a connection URL.
   * - :mod:`spindoctor.results_index.roots`
     - Root normalization, and the per-root ingest bookkeeping that makes
       absence of a row readable.
   * - :mod:`spindoctor.results_index.selection`
     - The one query answering the results-based selection filters.
   * - :mod:`spindoctor.results_index.drop`
     - Reading what a database holds, and removing the index's own tables.
   * - ``spindoctor.cli.stats.ingest``
     - The pass itself: walk, select, read, write, prune, complete --- in one
       process or divided into queue tasks.

The Core layer
==============

The index is written against **SQLAlchemy Core**, not the ORM. Tables are
``sqlalchemy.Table`` objects declared in one module against one
``sqlalchemy.MetaData``, and every read is a ``select()`` whose result the
caller consumes as rows. There is no mapped class, no session, no identity map
and no lazy loading.

That choice follows from what the index is for. Each consumer asks one narrow
question per run --- which stubs exist under this root, what does this stub
record --- and the answer is a set or a handful of scalars, never an object
graph to be navigated. An ORM would buy change tracking and relationship
loading that nothing here uses, and cost a mapping layer between the columns
and the rows that ingest writes from documents.

Two properties hold everywhere in the layer, and both are load-bearing:

**Every row is keyed by the pair** ``(root_url, results_path_stub)``. One
database serves several results roots, and two volumes hold images with the
same basename, so neither half identifies a row on its own. A query that
filters on the stub alone passes every test written against a single-root
fixture and answers another root's rows in production. Any new query needs a
fixture holding two roots whose second root *differs* in the value under test.

**Absence of a row is only meaningful once the root is known to be ingested.**
"No row for this stub" means "this image was never navigated" only after a
completed pass over that root; otherwise it means nothing at all.
:func:`~spindoctor.results_index.require_ingested_roots` is what every consumer
asks before it reads absence as an answer, and a completed run is necessary but
not sufficient: a document the ingest refused has a row in ``failed_files`` and
none in ``images``, so both tables are read before absence is reported.

Opening an index
================

:func:`~spindoctor.results_index.open_index` is the opener every reader and
writer goes through. It selects the backend from the URL, applies the SQLite
settings the concurrency model depends on, and refuses a database whose stamped
schema version is not the one the code reads.

Three rules about that opener are worth knowing before changing it:

- **Every refusal is a** :exc:`ValueError` **naming the URL**, including the
  ones a database driver raises, so a caller that reports failures catches one
  type rather than the driver's exception hierarchy.
- **Every message masks the URL's password**, and so does anything the failure
  underneath it quoted back. These messages reach run logs, cloud-task results
  and terminals.
- **The index owns the schema it lives in.** A creating open refuses a schema
  holding any table it did not create, and refuses one holding a table of the
  index's own names that no stamp of SpinDoctor's stands over --- ``images`` is
  among the commonest table names there are, and a stamp written beside
  somebody else's table would make it the index's for every later reading,
  including the drop's. Which schema that is comes from the stamp:
  :mod:`~spindoctor.results_index.scope` resolves ``schema_meta``, identifies
  it by two marks rather than by today's column set, and the drop resolves it
  the same way, so what one builds is what the other removes.

:func:`~spindoctor.results_index.open_database` is the exception that proves
the gate: dropping the tables is the remedy the gate's own message prescribes,
so the drop has to work on a database the gate refuses.

The concurrency model
=====================

Several processes write one index and many read it. What makes that safe is a
small number of decisions, each of which a change can break silently.

**One image is written whole or not at all.** Its ``images`` row and its
``techniques`` and ``feature_sources`` rows are written inside one transaction,
after a delete that cascades to the children, so a concurrent reader never sees
half an image and re-ingesting one replaces its children rather than doubling
them.

**Writes are chunked, not batched into one transaction.** ``sd_stats_ingest``
commits every ``INGEST_COMMIT_CHUNK_SIZE`` images, which bounds both what a
crash costs and how long a writer holds its lock. Retrieval is batched
separately (``INGEST_RETRIEVE_BATCH_SIZE``), because one bounds a download and
the other bounds a transaction. A chunk whose write fails is rewritten one
image at a time, so a single unstorable document costs itself rather than its
chunk.

**On SQLite, several local writers are an ordinary case.** The opener turns on
write-ahead logging and sets ``busy_timeout`` to ``SQLITE_BUSY_TIMEOUT_MS``, so
a competing writer waits rather than failing; with short transactions the
contention is brief. A ``sqlite:`` URL names a local path, which is what
confines this to one machine. Several machines share an index only through
PostgreSQL.

**Only a pass holding a listing of a whole root may delete a row.** Absence of
a row is an answer, so presence has to mean the tree still holds the document,
which is why each pass removes the rows of documents its walk did not find. A
worker handed a share of a root has no evidence about the stubs outside its
share and would delete its peers' rows, so nothing hands a worker a listing and
the prune refuses anything that is not a whole-root listing. In a
queue-divided pass the fan-out is the one step that sees a whole root, and it
prunes before it cuts the shares --- so the prune and the workers' writes
cannot touch the same stub.

**A root is unreadable until its run is stamped.** The run row is written
before the walk and its finish time is left NULL until the pass completes, so
every consumer treats a root that is being ingested as one nobody has ingested.
A pass that cannot list a directory under its root stops rather than
completing, so a run that carries a finish time listed the whole of its root.

The import exception on the navigation path
===========================================

``import spindoctor.dataset`` must not import SQLAlchemy. Every navigation run
imports that package, and most name no index at all, so the database layer has
no business on that path.

:class:`~spindoctor.dataset.results_filter.ResultsFilter` therefore imports
:func:`~spindoctor.results_index.read_result_stubs` **inside** the branch that
has a URL, rather than at the top of the module. The imports-at-the-top rule
permits an inline import only to keep a heavy optional dependency off a path
that does not need it, which is what this is and what the GUI toolkit's inline
imports are; it is not a cycle workaround, and the comment at the import says
which of the two it is.

A test pins it, in a subprocess: by the time any test in the session runs,
something has already imported SQLAlchemy, so the same assertion inside the
process would pass whatever the package did.

Adding a column
===============

There are no migrations. Any change to the column set of any table --- or to
the constraints over it --- means every existing index is rebuilt, and the
version gate is what makes that happen rather than being discovered later.

1. Add the column to its table in
   :mod:`spindoctor.results_index.schema`. Use ``Double`` rather than ``Float``
   for anything that must round-trip a document's value, ``Boolean`` rather
   than an integer flag, and ``JSON`` for a structured value.
2. **Increment** ``SCHEMA_VERSION`` in the same commit. Increment it again for
   a second change in the same branch rather than reusing the bump: an index
   built from the intermediate state would otherwise pass the gate and then
   fail on a column that is not there.
3. Fill it in ``spindoctor.cli.stats.ingest_rows``, reading the document
   through the accessors in ``spindoctor.support.nav_record`` that the
   consumers read it through. The invariant is that a record rebuilt from the
   columns classifies exactly as its document does; a second set of rules in
   the store is a second reader of the record, and the two drift.
4. Read it wherever it is consumed, and extend the reason vocabulary if the
   consumer classifies on it.
5. Update the column-set test in
   ``tests/spindoctor/results_index/test_schema.py``, which pins each table's
   columns, their types, their nullability and their order, and holds the
   schema version the column set belongs to. A version compared only against
   itself agrees with every value it could be given, which is why the number
   is written down beside the columns as well as in the schema.
6. Update the schema tables in
   :doc:`/user_guide/user_guide_statistics`, which document the index for
   somebody writing SQL against it.

Dropping a column is the same list, and the same version bump.

Testing
=======

The suite opens no index it was not handed, and the ``postgres`` tier is
deselected unless a server URL is exported --- so anything pinned only there is
unpinned in practice. See :doc:`dev_guide_testing` for both.

Two failure modes recur in this subsystem specifically, and both have cost real
time:

- **Root-blindness.** A query that filters on the stub and forgets the root.
  Every such defect passes a single-root fixture. Write the two-root fixture.
- **A test that cannot fail.** An exit status a catch-all also produces; an
  absence asserted against output that could not have contained it. When a test
  asserts that something is *not* reported, assert also that the thing it
  should report instead is.

API reference
=============

:doc:`/api_reference/api_results_index`
