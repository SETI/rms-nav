=================
The Results Index
=================

Overview
========

Navigation writes one ``_metadata.json`` document per image under
``nav_results_root``. Every program downstream of navigation then reads one of
those documents per image it processes. On a local disk that is cheap. On a
cloud root it is one paid round trip per image per program, and a
Cassini-scale run is of the order of 400,000 images.

The fields those programs actually consume are narrow: whether the image was
navigated, what fatal error it recorded, the pixel offset, the corrected
attitude and a handful of quality numbers. The results index is a database
that holds them. One pass reads every document once and writes one row per
image; from then on a consumer answers its questions with a query instead of a
download.

**The documents remain the authoritative record.** The index is derived and
disposable: nothing in it cannot be rebuilt from the tree, and deleting it
costs the time of one ingest and nothing else. Navigation never writes to it,
so nothing in the pipeline that produces results depends on a database being
reachable.

Two chapters divide the subject. This one is about the index as a whole: when
to build one, how programs are pointed at it, and what it does and does not
promise. :doc:`user_guide_statistics` documents ``sd_stats_ingest`` and
``sd_stats_report`` option by option, and holds the table-by-table schema and
the direct-SQL recipes.

Nothing requires an index
=========================

Every program runs with no index at all, and that is the default. A run that
names none reads the results tree exactly as it always has; there is no
fallback path to get wrong, because reading files *is* the ordinary path.

``sd_stats_report`` is the one exception in the other direction: it has no
file-reading mode, and it fails naming ``--results-db`` when no index is
resolved.

**A program becomes index-backed by declaring** ``--results-db``, never by
inheriting an exported ``NAV_RESULTS_DB``. A program whose selection is meant
to read files does not quietly change what it reads because a variable was
exported for another program. A declaring program honors all three levels of
the ladder below; a program that declares nothing reads the tree whatever is
set in the environment.

Naming an index
===============

Every index-backed program resolves its URL the same way, in this order:

1. the ``--results-db URL`` command-line option;
2. the ``environment.results_db`` configuration variable;
3. the ``NAV_RESULTS_DB`` environment variable.

The literal value ``none`` at any level means "no index", and overrides a URL
set at a lower one. That is how a single run is made to read the tree on a
machine that has an index configured:

.. code-block:: bash

    sd_backplanes ... --results-db none

Two URL forms are supported::

    sqlite:////data/nav-offset-results/index.sqlite3
    postgresql+psycopg://user@dbhost/spindoctor

A ``sqlite:`` URL names a **local filesystem path**, spelled with four slashes
for an absolute one. It is the one location in this system that is not
cloud-capable: the SQLite library opens the file directly, so a network
filesystem that cannot honor its locking is refused when the index is opened
rather than corrupted later. The URL carries no query string, since the driver
would then open a file named after the query.

A ``postgresql+psycopg:`` URL names a server, and needs the driver:

.. code-block:: bash

    pip install "rms-spindoctor[postgres]"

An index URL can carry a password, and these URLs are written to run logs,
printed in refusals and returned in cloud-task results. Every message that
names one masks it, so a log records ``postgresql+psycopg://user:***@dbhost``
and the command line it was typed on is masked the same way.

Which programs read it
======================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Program
     - What the index answers
   * - ``sd_stats_ingest``
     - Writes it. The only program that does.
   * - ``sd_stats_ingest_cloud_tasks``
     - Writes one share of it, as a queue worker.
   * - ``sd_stats_report``
     - Reads it and nothing else; the one program an index is not optional for.
   * - ``sd_offset``
     - The results-based selection filters (``--has-offset-file``,
       ``--has-no-offset-file``, the error filters), which otherwise walk the
       tree once per selected volume and read every document an error filter
       looks inside.
   * - ``sd_backplanes``, ``sd_backplanes_cloud_tasks``
     - Each image's recorded status and pointing, which the stage reads before
       it decides there is work to do.
   * - ``sd_mosaic``, ``sd_mosaic_cloud_tasks``
     - The pointing each contributing image is reprojected with.

Every other program reads the tree. Two do so for a reason that will not
change: ``sd_create_bundle`` serializes the whole navigation document into the
PDS4 supplemental product, and ``sd_consolidate_metadata`` copies raw file
bytes. Neither is served by a set of columns.

An index is a snapshot
======================

The index reflects the tree as of the last ingest over the roots it covers.
There is no staleness detection, no re-verification against the tree and no
automatic refresh. A consumer trusts what it holds.

That has two operator-visible consequences, and they run in opposite
directions:

- An image navigated since the last pass is one the index does not hold, so
  ``--has-no-offset-file`` selects it again and a downstream stage reports it
  as an image nothing navigated.
- A result file deleted since the last pass is one the index still holds, so
  ``--has-offset-file`` hands on an image whose document is gone.

Both are answered by running ``sd_stats_ingest`` again, which is cheap over a
tree that has barely changed: a document whose size and modification time
still match is not read at all. Every run that answers from an index reports
when the pass that filled it finished and how long ago that was, so the age of
the answer is in the run log beside the answer.

Where the two disagree for reasons the age does not explain --- a document the
ingest refused, a document rewritten in place --- the cases are enumerated in
:doc:`user_guide_navigation` under the selection filters, and the reason
vocabulary the backplane and reprojection stages report is in
:doc:`user_guide_backplanes` and :doc:`user_guide_reprojection`.

Building one
============

.. code-block:: bash

    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db sqlite:////data/nav-offset-results/index.sqlite3

**Ingest is never automatic.** No batch driver runs it as a side effect; an
operator runs it. A pass over an archive-scale root can also be spread over a
queue of workers, in three steps. Both are documented in
:doc:`user_guide_statistics`.

**The root is the navigation results root**, resolved the way every program
resolves one, and it is half of the key every row is stored under. Ingesting a
subdirectory of a results root produces identifiers no consumer's lookup can
match, so the root identity is part of the contract rather than a convenience.
One index holds as many roots as you ingest into it, and each consumer is
answered only from rows recorded under the root it was pointed at.

Roots are compared in one normalized spelling --- absolute, with any trailing
separator removed --- so a root named relatively on one run and absolutely on
the next is one root.

**A root the index has no completed ingest of is refused, not answered.**
Absence of a row would otherwise read as "this image was never navigated", so
a consumer pointed at an index that has never covered its root fails with a
message naming the root and the roots the index does hold. A pass that is
still running, or one that stopped, leaves its root in exactly that state
until it is completed.

Rebuilding one
==============

The index carries the version of the column set that wrote it. Opening one
stamped with a different version fails, naming both versions: there are no
migrations, because ingest is cheap relative to navigation and entirely
reproducible from the tree.

.. code-block:: bash

    sd_stats_ingest --results-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --drop-index
    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db sqlite:////data/nav-offset-results/index.sqlite3

``--drop-index`` removes the index's own tables and stops, walking no tree. It
is what makes starting over something an operator can reach without
hand-written SQL, on a shared PostgreSQL server as well as on a file. What it
will and will not remove --- and what an index shares a database with --- is
documented in :doc:`user_guide_statistics`.

Sharing one
===========

Several worker processes **on one machine** can write one SQLite index; there
is one file and no merge step. Workers on several machines cannot, because a
``sqlite:`` URL names a local path: a run spread across machines writes to
PostgreSQL.

For reading, the same rule decides. A SQLite index serves every consumer that
can open the file it names; consumers on other machines need either their own
copy of the file or a PostgreSQL server they can all reach.

Where to look next
==================

- :doc:`user_guide_statistics` --- ``sd_stats_ingest`` and ``sd_stats_report``
  in full: every option, the queue-of-workers workflow, the table-by-table
  schema, and direct-SQL recipes in both dialects.
- :doc:`user_guide_navigation` --- the selection filters, and what an
  index-answered selection holds that a tree-walked one does not.
- :doc:`user_guide_backplanes` and :doc:`user_guide_reprojection` --- what each
  stage reads per image, and what it reports when the index cannot answer.
- :doc:`/introduction_configuration` --- the configuration file and the
  environment variables the URL is resolved from.
