"""Read a navigation results tree into the results index.

One pass over a results root reads every ``*_metadata.json`` document under it
and writes one ``images`` row per document, with its per-technique and feature
inventory rows beside it.  Nothing else reads the tree afterwards: a consumer
that needs a few fields per image reads a row.

A pass may also be divided into cloud tasks: one program lists each root and
hands out shares of it, a worker ingests a share, and the shares' tallies are
added up to complete the run.  A worker removes no row, because nothing hands
it the complete listing that alone licenses one.

Public API:

    METADATA_SUFFIX            -- suffix of the per-image navigation document
    SUMMARY_PNG_SUFFIX         -- suffix of the summary PNG written beside it
    INGEST_RETRIEVE_BATCH_SIZE -- metadata files retrieved in one download
    INGEST_COMMIT_CHUNK_SIZE   -- images written per database transaction
    INGEST_TASK_SHARE_SIZE     -- metadata files handed to one cloud task
    IngestCounts               -- what one pass did
    ingest_metadata_files      -- the pass itself, over one or more roots
    FanOut                     -- the tasks one root divides into
    fan_out_ingest_tasks       -- list each root once and divide it up
    ingest_task_share          -- ingest one task's share of a root
    TaskResults                -- what a worker event log holds
    task_results_from_event_log -- read the workers' return values back
    TaskCompletion             -- what adding the shares up did
    complete_ingest_tasks      -- add them up and stamp the runs

The implementation is split by stage, and this module re-exports the whole
surface so a consumer imports from ``spindoctor.cli.stats.ingest`` whichever
stage a name lives in:

* :mod:`~spindoctor.cli.stats.ingest.counts` -- the tally a pass keeps and the
  summary it is read from.
* :mod:`~spindoctor.cli.stats.ingest.walk` -- the single listing of a root,
  which every later stage draws on.
* :mod:`~spindoctor.cli.stats.ingest.store` -- what the index already holds
  about a root, and how rows go back into it.
* :mod:`~spindoctor.cli.stats.ingest.chunks` -- batched retrieval, reading a
  document into rows, and the per-chunk write.
* :mod:`~spindoctor.cli.stats.ingest.runs` -- the record of one pass over one
  root, which is what makes absence of a row readable.
* :mod:`~spindoctor.cli.stats.ingest.driver` -- the pass itself: walk, select,
  ingest, prune, complete.
* :mod:`~spindoctor.cli.stats.ingest.tasks` -- the same pass divided into cloud
  tasks: fan out, ingest a share, add the shares up.
"""

from spindoctor.cli.stats.ingest.chunks import INGEST_RETRIEVE_BATCH_SIZE
from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest.driver import INGEST_COMMIT_CHUNK_SIZE, ingest_metadata_files
from spindoctor.cli.stats.ingest.tasks import (
    INGEST_TASK_SHARE_SIZE,
    FanOut,
    TaskCompletion,
    TaskResults,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_task_share,
    task_results_from_event_log,
)
from spindoctor.cli.stats.ingest.walk import METADATA_SUFFIX, SUMMARY_PNG_SUFFIX

__all__ = [
    'INGEST_COMMIT_CHUNK_SIZE',
    'INGEST_RETRIEVE_BATCH_SIZE',
    'INGEST_TASK_SHARE_SIZE',
    'METADATA_SUFFIX',
    'SUMMARY_PNG_SUFFIX',
    'FanOut',
    'IngestCounts',
    'TaskCompletion',
    'TaskResults',
    'complete_ingest_tasks',
    'fan_out_ingest_tasks',
    'ingest_metadata_files',
    'ingest_task_share',
    'task_results_from_event_log',
]
