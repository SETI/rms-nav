"""Read a navigation results tree into the results index.

One pass over a results root reads every ``*_metadata.json`` document under it
and writes one ``images`` row per document, with its per-technique and feature
inventory rows beside it.  Nothing else reads the tree afterwards: a consumer
that needs a few fields per image reads a row.

Public API:

    METADATA_SUFFIX            -- suffix of the per-image navigation document
    SUMMARY_PNG_SUFFIX         -- suffix of the summary PNG written beside it
    INGEST_RETRIEVE_BATCH_SIZE -- metadata files retrieved in one download
    INGEST_COMMIT_CHUNK_SIZE   -- images written per database transaction
    IngestCounts               -- what one pass did
    ingest_metadata_files      -- the pass itself, over one or more roots

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
"""

from spindoctor.cli.stats.ingest.chunks import INGEST_RETRIEVE_BATCH_SIZE
from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest.driver import INGEST_COMMIT_CHUNK_SIZE, ingest_metadata_files
from spindoctor.cli.stats.ingest.walk import METADATA_SUFFIX, SUMMARY_PNG_SUFFIX

__all__ = [
    'INGEST_COMMIT_CHUNK_SIZE',
    'INGEST_RETRIEVE_BATCH_SIZE',
    'METADATA_SUFFIX',
    'SUMMARY_PNG_SUFFIX',
    'IngestCounts',
    'ingest_metadata_files',
]
