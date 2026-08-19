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
     - :func:`~spindoctor.results_index.open_index`: backend selection, SQLite
       settings, the version gate.
       :func:`~spindoctor.results_index.open_database` is the one opener that
       stops before the gate.
   * - :mod:`spindoctor.results_index.scope`
     - Which schema of a database the index lives in, decided from the stamp.
   * - :mod:`spindoctor.results_index.masking`
     - The one rule that hides a password in a connection URL.
   * - :mod:`spindoctor.results_index.roots`
     - The per-root ingest bookkeeping that makes absence of a row readable,
       and the opener that checks it. Root normalization is a rule about
       identity rather than about a database, so it lives in
       :mod:`spindoctor.nav_records` and is re-exported from here.
   * - :mod:`spindoctor.results_index.selection`
     - The one query answering the results-based selection filters.
   * - :mod:`spindoctor.results_index.rebuild`
     - The one correspondence between a row's columns and a record's fields, and
       the rebuild that reads it.
   * - :mod:`spindoctor.results_index.record_source`
     - The record seam's index-backed half, and
       :func:`~spindoctor.results_index.open_record_source`, which decides
       which half a run gets.
   * - :mod:`spindoctor.results_index.drop`
     - Reading what a database holds, and removing the index's own tables.
   * - :mod:`spindoctor.nav_records`
     - Not part of this package: the record seam's database-free half. What a
       record is, what a document is named and where one lives, what a
       selection is, the protocol both storages implement, the implementation
       over the documents themselves, and
       :func:`~spindoctor.nav_records.facts.facts_from_document`, which turns
       one document into the per-image shape both storages answer in. Reading a
       document needs no database, so every reader shares it whether or not its
       program can read an index.
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

One seam for records
====================

Several programs read what a navigation pass wrote, and each of them can be
pointed at either storage. What decides whether they agree is that none of them
reads a storage itself: they all go through
:class:`~spindoctor.nav_records.RecordSource`, and each storage implements it.
``sd_stats_ingest`` is the one that reads only documents, since documents are
what it builds the index out of: it discovers them through the seam's listing
and keeps a reading loop of its own, which owns the refusal vocabulary the
index stores.

Four questions, because the programs ask four things
----------------------------------------------------

:meth:`~spindoctor.nav_records.RecordSource.record` takes one stub and returns
one record. It is the shape a per-image loop asks in --- the backplane and
mosaic stages reading the pointing an image is built with --- so it is the call
an implementation answers in a single round trip. A stub is a key under a root,
so a source holding more than one root refuses it, naming them.

:meth:`~spindoctor.nav_records.RecordSource.records` takes a selection and
yields the records it covers, one at a time. It is what a program that
summarizes or sweeps asks once per run: the kernel writer reading a mission, the
statistics report reading a root. It also takes an explicit list of stubs, which
is what a queue task carries --- a worker must read exactly the files it was
given, and read them in batches, so looping the per-image call would cost it a
round trip apiece on the storage that batching exists for.

:meth:`~spindoctor.nav_records.RecordSource.facts` takes a selection and yields
what every image it covers says about itself --- the image's own values, one
entry per technique that reported, and the aggregated inventory of the features
the models offered. It is what a program reading every field of every image asks
for, because a record cannot carry that: a record is defined as looking like the
document it stands for, so the index rebuilds one out of the columns its consumer
selected and invents no field the document did not have, and no record carries a
per-technique or per-feature row at all. The facts are the whole row --- what
the index column set holds about an image, in the shape both storages hold it
in --- so the columns a consumer named narrow its records and never its facts.
A field of a document that no column holds is in neither storage's facts.

:meth:`~spindoctor.nav_records.RecordSource.listing` takes a selection and
yields what is there --- each stub, where it lives, and the size and
modification time that say whether it has changed --- without opening a single
document. On a cloud root one directory listing returns up to a thousand entries
with their metrics, against one round trip per document, which is what the
ingest's discovery and its unchanged-file skip are both built on. What a listing
cannot do is answer anything a document says, so a selection restricting on a
mission or a span is refused rather than partly honored.

Two more methods hold the run together rather than answering a question:
``describe()`` says which storage answered, for the run log, and ``close()``
releases what the source opened. A stream may hold a connection or a cursor, so
a caller that walks away mid-loop must still release it: a source is a context
manager, and a run writes ``with open_record_source(...) as source:``.

**A stream yields, and returns no list.** A caller that wants one writes
``list(...)`` and owns that decision. Nothing is accumulated on a caller's
behalf, so a program sweeping a mission holds one record at a time rather than
the mission --- and a program wanting many summaries of one stream is forced to
compute them in one pass, which is the constraint that makes the stream worth
having. Nothing promises an order either: a walk cannot know an image's epoch
before it has read the document, and a database sorting text sorts it under the
server's own collation, so each implementation yields in the order it finds
records, says what that order is, and a caller needing a total order calls
``sorted()`` and pays for it knowingly.

**A failure arrives from** ``next()``. A file that could not be read is yielded
into the stream as an :class:`~spindoctor.nav_records.UnreadableFile` rather
than raised, so one of them costs itself and not the rest of the pass. A
refusal that ends a pass --- a
directory nobody can list, an index that stops answering --- surfaces in the
middle of the caller's loop, so a program using the stream finishes its pass
before it writes its output.

Two backends, and why the package is split
------------------------------------------

:class:`~spindoctor.nav_records.TreeRecordSource` answers from the documents:
it walks directory by directory, carrying each entry's metrics out of the
listing, and retrieves documents in batches underneath a stream that yields them
one at a time. :class:`~spindoctor.results_index.IndexRecordSource` answers from
the rows, streamed in server-side chunks. The per-image lookup and the listing
are one query each; a stream of records is two, the images and the files the
ingest refused; and :meth:`~spindoctor.nav_records.RecordSource.facts` is four,
because the per-technique and per-feature rows are merged onto the images stream
by key. That merge is the one place a statement here sorts --- its three
streams order on the key, so each image's child rows arrive next to it --- and
it is safe for the one reason a text sort ever is: the three orders are one
server's, read from one snapshot, and nothing compares them to an order
computed anywhere else.
:func:`~spindoctor.results_index.open_record_source` is what a program calls; it
returns the first when the run names no index and the second when it names one.

**The seam is split along the database line, and the line is not a matter of
taste.** ``import spindoctor.dataset`` must not import SQLAlchemy: every
navigation run imports that package and most of them name no index. The
enumeration is a reader of documents and lives in that package, so the half of
the seam that reads documents has to be reachable from it without acquiring a
database. :mod:`spindoctor.nav_records` --- what a record is, what a document is
named and where one lives, what a selection is, the protocol, the implementation
over the documents, and the per-image shape built from one document ---
therefore imports no database layer, and a subprocess test pins that. The half
that reads rows needs SQLAlchemy by definition, so it lives in
:mod:`spindoctor.results_index` alongside the schema
it reads, and so does the factory that chooses between the two. That is why a
consumer on the navigation path resolves its index half through a branch-local
import rather than at the top of its module.

What a caller asks for
----------------------

:class:`~spindoctor.nav_records.Selection` is the one value every stream is
asked in terms of: which of the source's roots to read, which top-level
directories to descend, which stubs to read outright, one mission, and a span of
exposure midtimes. Every field narrows and the fields combine, and a field left
at its default narrows nothing, so the empty selection covers everything the
source holds. Each backend applies whichever restrictions it can answer
cheaply --- a walk by looking at fewer directories, a query in its ``WHERE``
clause --- and the answer is the same either way.

It is frozen, and everything it carries is checked in its constructor: a subtree
is one directory immediately under a root, a stub is a key rather than a path,
and a time bound is a finite number with the start no later than the stop.
**That is what makes the two backends refuse alike.** A walk and a query cannot
be made to refuse identically by writing the same refusal twice; they refuse
identically because there is one place a selection can be wrong and neither of
them is reached from it. Left to a backend, an unusable value is refused in that
storage's own terms --- one raises out of a query builder in a language its
caller cannot name, another yields nothing, and an inverted range selects
nothing at all, which a run cannot tell from a clean pass over a quiet span.

The rules that hold at the seam
-------------------------------

**One piece of code, because two would be two answers.** A consumer carrying its
own row-to-record rebuild, its own open-and-check ceremony and its own account
of how the two storages differ is a second reader of the record: two rebuilds
agree the day they are written and then each grows a rule the other does not.
The defect that makes the point is an ``ORDER BY`` on text --- correct under
SQLite's default collation, wrong under a PostgreSQL locale collation, and
invisible to a consumer that reads one row at a time.

**A consumer names the columns it reads.** A row is only cheaper than a document
while it carries less, so nobody selects forty columns to read five. The columns
are declared beside the consumer --- ``_ROW_COLUMNS`` for the reprojection and
backplane stages, ``RECORD_COLUMNS`` for the kernel writer --- and a test holds
each list to the fields the rebuild knows a place for, since a column selected
that no field is rebuilt from is paid for on every read and then dropped. The
reprojection list is held to the other direction too: each of its columns is
dropped in turn and what those readers are then given has to differ, since a
column whose absence no reader notices is paid for on every read and then
ignored.

**The rebuild is a table, not a function per consumer.**
:data:`~spindoctor.results_index.rebuild.RECORD_FIELDS` maps each column to its
place in the record. Every column of ``images`` is deliberately one of three
things --- a record field, part of a row's identity, or a value the ingest
computed rather than copied --- and a test asserts the three are a partition of
the table, so a column added to the schema and to the ingest and to nobody's
consumer is caught rather than read as absent by everything.

**A record read from a document carries every field it has; one rebuilt from a
row carries what its consumer selected.** That is the one difference a consumer
has to know about, and it is why the columns are pinned rather than assumed.

**A key is checked where it is written, not where it is read.** A results root
is normalized to one absolute, resolved spelling by
:func:`~spindoctor.nav_records.normalize_root_url` the moment a program is
handed one, and a stub and a subtree are checked as keys by
:class:`~spindoctor.nav_records.Selection` the moment a caller writes one: a
stub names no absolute path, no parent directory and no null byte, and a subtree
is one directory immediately under a root. Everything below that point is a
join, with one answer, so no reader carries a rule of its own about what a join
may have produced --- which is what a walk and a query could not have been made
to agree about by writing the same rule twice.

:func:`~spindoctor.results_index.roots.open_index_for_roots` is where the index
side's opening ceremony lives: it opens an index, refuses a root the index has
no completed ingest of, and disposes of the engine before the refusal leaves it.
Written out per call site, that sequence loses a step at a time --- an engine
left undisposed on a refusal, a root checked after the first query rather than
before it.

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
separately (``RETRIEVE_BATCH_SIZE``), because one bounds a download and
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
share and would delete its peers' rows, so nothing hands a worker a listing. The
licence is a type rather than a check: one function builds the listing the prune
takes, it builds one only for a root it listed entirely, and it carries the root
it listed --- so a share of a root, a partial listing and a prune of one root on
the evidence of another are all unrepresentable rather than refused. In a
queue-divided pass the fan-out is the one step that sees a whole root, and it
prunes before it cuts the shares --- so the prune and the workers' writes
cannot touch the same stub.

**A root is unreadable until its run is stamped.** The run row is written
before the walk and its finish time is left NULL until the pass completes, so
every consumer treats a root that is being ingested as one nobody has ingested.
A pass that cannot list a directory under its root stops rather than
completing, so a run that carries a finish time listed the whole of its root.
That rule belongs to the walk rather than to the ingest: the walk is
:class:`~spindoctor.nav_records.TreeRecordSource`'s, so a kernel-writing or
reporting run over the same tree stops at the same directory. A root that
cannot be listed **at all** raises
:exc:`~spindoctor.nav_records.UnlistableRootError` instead, which the ingest
catches to charge a mistyped root to that root and go on to the next one, and
which every other consumer lets end its run.

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

The same rule is what puts the record seam's database-free half in
:mod:`spindoctor.nav_records` rather than here. The enumeration reads documents
through that seam, so a database layer imported anywhere under it would be
imported by every navigation run --- and an inline import cannot save a package
whose whole surface is on that path. The split is the arrangement instead: the
record types, the document rules, the selection, the protocol, the tree backend
and the per-image shape acquire no database, and the index-backed half and the
factory live on this side, where SQLAlchemy is already a dependency.

Both halves of that are pinned in a subprocess, and have to be: by the time any
test in the session runs, something has already imported SQLAlchemy, so the same
assertion inside the process would pass whatever the packages did. One probe
imports :mod:`spindoctor.dataset` and one imports
:mod:`spindoctor.nav_records`, each asserting that no module of SQLAlchemy
loaded --- and the second asserts as well that the walk itself loaded, since a
guarantee about a package that imported almost nothing is a guarantee about
nothing.

Adding a column
===============

There are no migrations. Any change to the column set of any table --- or to
the constraints over it --- means every existing index is rebuilt.

:data:`~spindoctor.results_index.SCHEMA_VERSION` is pinned at 1 and stays there,
so the version gate does not perform that rebuild for you: an index built from a
different column set carries the same stamp and opens. **Drop and rebuild every
index you hold, by hand, in the same change.** An index left in place reads as
current and then fails on a column that is not in it.

1. Add the column to its table in
   :mod:`spindoctor.results_index.schema`. Use ``Double`` rather than ``Float``
   for anything that must round-trip a document's value, ``Boolean`` rather
   than an integer flag, and JSON for a structured value. Which of the two JSON
   declarations depends on what the value holds: a structure that can hold a
   number takes the one that is plain ``json`` on PostgreSQL, because ``jsonb``
   stores numbers as ``numeric`` and returns a stored negative zero as zero and
   a large-magnitude float as an integer; one holding text alone takes the
   ``jsonb`` variant, whose array and object accessors a direct-SQL query
   reaches inside without a cast.
2. Fill it in ``spindoctor.nav_records.facts``, reading the document through
   the accessors in ``spindoctor.support.nav_record`` that the consumers read
   it through. The invariant is that a record rebuilt from the
   columns classifies exactly as its document does; a second set of rules in
   the store is a second reader of the record, and the two drift.
3. Sort it into one of the three groups in
   :mod:`spindoctor.results_index.rebuild`. A column copied out of one field of
   one document is an entry in ``RECORD_FIELDS`` naming where in a record that
   field sits; one that says where the document is belongs to
   ``IDENTITY_COLUMNS``; one the ingest computes rather than copies --- an epoch
   read from whichever of two fields the document had, a count of a list ---
   belongs to ``DERIVED_COLUMNS``, because no field of a record is what it came
   from. The three are asserted to be a partition of the table, so a column left
   out of all three fails rather than reading as absent from every record.
4. Read it wherever it is consumed, by adding it to that consumer's column list,
   and extend the reason vocabulary if the consumer classifies on it. A consumer
   that does not select a column reads its field as absent.
5. Update the column-set lists in
   ``tests/spindoctor/results_index/test_schema.py``: the per-table list, which
   pins each table's columns, their types, their nullability and their order,
   and --- for a JSON column --- the list of JSON columns, which pins the type
   each backend emits for it and that an absent value is stored as SQL NULL. A
   JSON column missing from the second list fails the test that holds the two
   against the schema. With the stamp pinned, those lists are the whole of what
   a column change has to answer to, which is why they restate every name, type,
   nullability and position instead of reading them off the schema they guard.
6. Update the schema tables in
   :doc:`/user_guide/user_guide_results_index`, which document the index for
   somebody writing SQL against it.

Dropping a column is the same list, and the same rebuild by hand.

The edits, in the order the list gives them:

.. code-block:: python

    # spindoctor/results_index/schema.py
    IMAGES = sqlalchemy.Table(
        'images',
        METADATA,
        ...,
        # What the column carries, and what a NULL in it means.
        sqlalchemy.Column('shutter_mode', sqlalchemy.Text),
    )

    # spindoctor/nav_records/facts.py, inside facts_from_document
    image_row: dict[str, Any] = {
        ...,
        'shutter_mode': _str_or_none(observation.get('shutter_mode')),
    }

    # spindoctor/results_index/rebuild.py, inside RECORD_FIELDS
    RecordField(('shutter_mode',), _OBSERVATION, 'shutter_mode'),

    # The consumer's own column list, which is what decides that it is read
    RECORD_COLUMNS = (
        ...,
        IMAGES.c.shutter_mode,
    )

    # The consumer, reading the field the column was rebuilt into
    mode = record['observation'].get('shutter_mode')

    # tests/spindoctor/results_index/test_schema.py
    IMAGES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
        ...,
        ('shutter_mode', sqlalchemy.Text, True),  # name, type, nullable
    )

    # And, for a JSON column only, the declaration each backend emits
    JSON_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ...,
        ('images.spice_kernels', 'JSON', 'JSONB'),  # column, SQLite, PostgreSQL
    )

The value is read out of the document with the coercion its column's meaning
calls for, and never coerced past what the document said: ``str(None)`` is
``'None'``, which would be stored as a value a document never recorded.

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
