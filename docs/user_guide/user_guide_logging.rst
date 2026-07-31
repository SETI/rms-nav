=======
Logging
=======

Every SpinDoctor pipeline program writes two kinds of record, and keeps them
apart. This page covers what goes where, how to change it, and where the files
land.

The two loggers
===============

**The main log** covers one run of one program. It reports what the program is
doing at the top level: which image it is about to process, counts and totals,
elapsed time, and the path of each image log it wrote. There is one for the
life of the run.

**An image log** covers one image inside one processing stage. It carries the
detail of that image's processing -- which models were built, which techniques
ran, what each of them found. A new one is started for each image.

A record belongs to exactly one of them, so the two never repeat each other.
When you want to know what a run did, read the main log; when you want to know
what happened to one image, read that image's log.

Not every program has both. A program that does not process images
individually has only a main log, and the statistics and GUI programs have
neither -- they write to the terminal directly, because both are read as they
run rather than afterwards.

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Program
     - Main log
     - Image log
   * - ``sd_offset``
     - yes
     - ``nav``
   * - ``sd_backplanes``
     - yes
     - ``backplanes``
   * - ``sd_mosaic`` (and ``sd_mosaic_rings`` / ``sd_mosaic_body``)
     - yes
     - ``reproj``
   * - ``sd_create_bundle``
     - yes
     - none
   * - ``sd_consolidate_metadata``
     - yes
     - none
   * - ``sd_offset_cloud_tasks``
     - no
     - ``nav``
   * - ``sd_backplanes_cloud_tasks``
     - no
     - ``backplanes``
   * - ``sd_mosaic_cloud_tasks``
     - no
     - ``reproj``
   * - ``sd_stats_ingest``, ``sd_stats_report``
     - no
     - none
   * - ``sd_create_simulated_image``
     - no
     - none
   * - ``sd_backplane_viewer``, ``sd_mosaic_display``
     - no
     - none

Where the files go
==================

Both kinds live under one log root, named by ``--log-root``, the
``environment.log_root`` configuration variable or the ``NAV_LOG_ROOT``
environment variable, in that order of precedence.

With none of those set, the root is derived: a ``logs`` directory under the
navigation results root. A cloud-task worker is not required to have a
navigation results root, so each falls back to a ``logs`` directory under the
root it does have -- the backplane results root for
``sd_backplanes_cloud_tasks``, and the task's own output directory for
``sd_mosaic_cloud_tasks`` -- rather than dropping its logs for want of a
setting that does not apply to it.

.. code-block:: text

   {log_root}/{program}/main_{timestamp}.log
   {log_root}/{backend}/{results_path_stub}_{timestamp}.log

The main log is filed under the program that wrote it. An image log is filed
under the *stage* rather than the program, so an image's navigation log sits
beside every other navigation log whether an interactive run or a cloud task
produced it. The three stages are ``nav``, ``backplanes`` and ``reproj``.

Every file from one run shares a single timestamp, in UTC and in
``YYYY-MM-DDTHH-MM-SS`` form. UTC rather than local time so the names sort
chronologically and can be compared across machines, which matters when a
batch is spread over workers in different time zones.

.. note::

   The timestamp in the file *name* is UTC; the timestamps on the records
   *inside* are local. A log named ``..._2026-07-31T02-36-04.log`` can open
   with ``2026-07-30 19:36:04``. Match a log to a wall-clock time by its
   contents rather than its name, and glob by name only in UTC terms.

Reprojection logs are keyed by mosaic subject as well, since one image may be
reprojected onto more than one body::

   {log_root}/reproj/{subject}/{results_path_stub}_{timestamp}.log

.. note::

   Reprojection logs live under the log root, not under the mosaic output
   directory. If you have a script or a habit that reads them from
   ``<output-dir>/logs``, it needs updating.

What appears on the terminal
============================

By default the main log goes to both the terminal and a file, and image logs go
to a file only.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Logger
     - Terminal
     - File
   * - Main
     - yes
     - yes
   * - Image
     - no
     - yes

.. note::

   An interactive run therefore shows top-level progress rather than
   per-component detail. The detail is not lost -- it is in the per-image log
   file. Pass ``--log-image-to-console`` to see it on screen as well.

Turning every sink off produces no output at all, rather than falling back to
the terminal.

Command-line options
====================

Every program you run yourself accepts the same options, so what you learn for
one works for the next. A program with no image log accepts only the
main-logger options, and rejects the image ones by name rather than accepting
and ignoring them. The ``_cloud_tasks`` drivers accept none of these and are
configured through the configuration file alone; see `Cloud tasks`_ below.

``--log-root PATH``
    Where this run's log files go.

``--log-level LEVEL``
    The default level for both loggers.

``--log-level MODULE=LEVEL``
    The level for one component. Repeatable, and combines with the bare form.

``--log-level-main LEVEL``, ``--log-level-image LEVEL``
    The level for one logger, taking precedence over a bare ``--log-level``.

``--log-main-to-console`` / ``--no-log-main-to-console``
    Whether the main log reaches the terminal. Default on.

``--log-main-to-file`` / ``--no-log-main-to-file``
    Whether the main log is written to a file. Default on.

``--log-image-to-console`` / ``--no-log-image-to-console``
    Whether image logs reach the terminal. Default off.

``--log-image-to-file`` / ``--no-log-image-to-file``
    Whether image logs are written to files. Default on.

Levels are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL`` and
``NONE``. Both sinks of a logger always share a level, so there is one level to
set per component rather than one per sink.

Worked examples
---------------

Quiet the run but keep one technique verbose, which is the usual shape of
investigating a single technique across many images:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 \
       --log-level WARNING --log-level titan_haze=DEBUG

Watch one image's detail on screen instead of opening its file:

.. code-block:: bash

   sd_offset coiss N1234567890 --log-image-to-console --log-level DEBUG

Keep the terminal clean and the files complete, for a long batch:

.. code-block:: bash

   sd_offset coiss_saturn --no-log-main-to-console --log-level-image DEBUG

Silence one noisy component without lowering anything else:

.. code-block:: bash

   sd_offset coiss_saturn --log-level annotate=NONE

Configuring levels
==================

Anything settable on the command line is settable in the configuration, under
the top-level ``logging`` section:

.. code-block:: yaml

   logging:
     main: INFO            # the run's logger
     image: INFO           # image logs, and any component not named below
     techniques:
       titan_haze: DEBUG   # one technique
       default: WARNING    # every other technique
     models:
       rings: WARNING      # one model family
     other:
       annotate: ERROR
     programs:
       sd_mosaic:          # applies to that program only
         main: WARNING

The most specific setting wins:

.. code-block:: text

   --log-level MODULE=LEVEL
     > a component named in the configuration
     > its category's "default"
     > --log-level-main / --log-level-image
     > --log-level LEVEL
     > logging.main / logging.image
     > INFO

A ``programs`` block applies to that program alone and is merged key by key
with the settings above it, so a program can override one value while
inheriting the rest.

An unrecognized component name, program name or level is rejected when the
configuration loads, naming the offending key. A setting that does nothing is
worse than one that errors, because it looks like it worked.

Component names
---------------

A component is named by the technique or model it is, in snake_case.

**Techniques** -- ``body_blob``, ``body_disc_correlate``, ``body_limb``,
``body_terminator``, ``manual``, ``ring_annulus``, ``ring_edge``,
``star_field_from_catalog``, ``star_refine``, ``star_unique_match``,
``titan_haze``

**Models** -- ``body``, ``rings``, ``stars``, ``titan``. One name covers a
whole family: ``body`` governs every body model regardless of which body it
renders, and a simulated model is named with the model it stands in for.

**Everything else** -- ``annotate``, ``correlate``, ``ensemble``,
``image_derivatives``, ``obs``, ``orchestrator``, ``provenance``

Cloud tasks
===========

The ``_cloud_tasks`` drivers write **nothing** to the terminal. A worker's
console belongs to ``cloud_tasks``, which reports task progress there under its
own configuration, and interleaving per-image navigation detail with it would
make both harder to read.

The per-image logs are written exactly as an interactive run writes them, to
the same ``{log_root}/{backend}/`` tree and at the same levels, so an image's
log reads the same whichever driver produced it. There is no main log: with
many workers writing to one log root, a single shared main log is not something
they can all append to sensibly.

These drivers accept no logging command-line options, because every one of them
configures a logger they do not have or a terminal they must not write to. Set
their levels in the configuration instead -- with a ``programs`` block if they
should differ from an interactive run.

Because a cloud task has no main log, an outcome that an interactive run would
report there is returned in the task result instead. A backplanes task reports
whether the image was processed or skipped, and a reprojection task returns how
many images it completed, skipped and failed.
