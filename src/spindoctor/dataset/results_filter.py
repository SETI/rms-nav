"""Filter image selections against existing navigation result files.

Implements the ``--has-offset-file`` / ``--has-no-offset-file`` /
``--has-png-file`` / ``--has-no-png-file`` / ``--has-offset-error`` /
``--has-offset-spice-error`` / ``--has-offset-nonspice-error`` image
selection options shared by the PDS3 datasets.

The navigation pipeline writes ``{nav_results_root}/{results_path_stub}_metadata.json``
and ``{results_path_stub}_summary.png`` (see
:func:`spindoctor.navigate_image_files.navigate_image_files`).  Presence
filters are answered by walking the results tree once per selected volume and
collecting the existing result files into sets, so each candidate image costs
no additional cloud round trip.  Absence filters are answered with batched
``FCPath.exists()`` calls (or from the walked sets when a walk already
happened), and the error filters retrieve the metadata JSON files in batches
and inspect their ``status`` / ``status_error`` fields.

Given a results index, every one of those questions is answered instead by one
query over the index, and the tree is not read at all.  The index-backed
implementation lives in :mod:`spindoctor.results_index.selection` and is
imported inside the branch that has a URL, not at the top of this module: this
module is reached by importing :mod:`spindoctor.dataset`, which every navigation
run does, and the top-level import would put SQLAlchemy on that path for the
runs that never name an index.  That module also enumerates, in full, the
answers the index gives differently from the tree.
"""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from filecache import FCPath, FileCache
from pdslogger import PdsLogger

from .dataset import ImageFile

METADATA_SUFFIX = '_metadata.json'
"""Suffix of the per-image offset metadata file under the results root."""

SUMMARY_PNG_SUFFIX = '_summary.png'
"""Suffix of the per-image summary PNG file under the results root."""

RESULTS_FILTER_BATCH_SIZE = 64
"""Number of images checked per batched ``exists()`` / ``retrieve()`` call."""

_SPICE_STATUS_ERROR = 'missing_spice_data'
"""The ``status_error`` value the SPICE error filters tell apart.

The index-backed implementation names the same value rather than sharing this
one, because this module may not import that one at the top of the file, which
is what the branch-local import exists to avoid.
"""


def _elapsed_phrase(seconds: float) -> str:
    """Return an interval in the coarsest unit that has a whole number of it.

    Parameters:
        seconds: The interval.

    Returns:
        The interval named in days, hours or minutes, or as less than a minute.
    """
    for size, unit in ((86400.0, 'day'), (3600.0, 'hour'), (60.0, 'minute')):
        if seconds >= size:
            count = int(seconds // size)
            return f'{count} {unit}' if count == 1 else f'{count} {unit}s'
    return 'less than a minute'


def _snapshot_age(ingested_utc: str | None) -> str:
    """Return how old an index's answer is, in the terms an operator acts on.

    The index answers as of the pass that filled it and detects no change since,
    so the age of that pass is what says whether its answer is the answer the
    tree would give.  Both the stamp and the interval are reported: the stamp
    names the pass to re-run, and the interval is what a reader compares against
    what they know they have navigated.

    Parameters:
        ingested_utc: When the newest pass over the root finished, as the index
            recorded it, or None when it recorded nothing.

    Returns:
        A phrase naming that time and how long ago it was.  A stamp that will
        not parse, or one in the future because two clocks disagree, is reported
        as it stands: a reader can act on a value the index really holds, and an
        interval computed from a stamp nobody can read would be a fiction.
    """
    if not ingested_utc:
        return 'at a time this index does not record'
    try:
        finished = datetime.fromisoformat(ingested_utc)
    except ValueError:
        return ingested_utc
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    elapsed = (datetime.now(UTC) - finished).total_seconds()
    if elapsed < 0:
        return ingested_utc
    return f'{ingested_utc} ({_elapsed_phrase(elapsed)} ago)'


class ResultsFilter:
    """Filters candidate images against their navigation result files.

    Constructed once per enumeration when any of the results-based selection
    flags is active.  Construction validates the flag combination (the flags
    AND together; directly contradictory pairs raise) and then collects what
    the results root holds: from the index in one query when a results-index
    URL is given, and otherwise, when a presence or error filter is active, by
    walking the results tree under each selected volume.

    The filter is applied in two stages:

    - :meth:`passes_presence` is a cheap per-row set-membership test used
      while scanning index rows.
    - :meth:`filter_batch` applies the absence and metadata-content filters
      to a batch of already-accepted images with one batched ``exists()``
      and/or ``retrieve()`` call, preserving input order.  Answered from a
      results index, every filter is settled in the first stage and this one
      does nothing.
    """

    def __init__(
        self,
        volumes: Iterable[str],
        nav_results_root: str | Path | FCPath,
        *,
        has_offset_file: bool = False,
        has_no_offset_file: bool = False,
        has_png_file: bool = False,
        has_no_png_file: bool = False,
        has_offset_error: bool = False,
        has_offset_spice_error: bool = False,
        has_offset_nonspice_error: bool = False,
        results_db_url: str | None = None,
        logger: PdsLogger,
    ) -> None:
        """Validates the flag combination and collects what the results root holds.

        Parameters:
            volumes: Volume names selected by the other constraints; only these
                subdirectories of the results root are walked, and only images
                under them are read from a results index.
            nav_results_root: Root of the navigation results tree; may be a
                cloud URL.  A ``str`` or ``Path`` is normalized to an
                :class:`FCPath` at construction; an existing :class:`FCPath` is
                used as given so its file cache is preserved.
            has_offset_file: Only keep images whose offset metadata file exists.
            has_no_offset_file: Only keep images whose offset metadata file does
                not exist.
            has_png_file: Only keep images whose summary PNG file exists.
            has_no_png_file: Only keep images whose summary PNG file does not
                exist.
            has_offset_error: Only keep images whose offset metadata file
                indicates a fatal error (``status == 'error'``).
            has_offset_spice_error: Only keep images whose offset metadata file
                indicates a fatal error from missing SPICE data.
            has_offset_nonspice_error: Only keep images whose offset metadata
                file indicates a fatal error other than missing SPICE data.
            results_db_url: Connection URL of a results index to answer every
                filter from, or None to read the results tree.  A URL that
                cannot be used is an error rather than a reason to fall back
                to the tree.
            logger: Logger for scan statistics and unreadable-metadata warnings.

        Raises:
            ValueError: If the flag combination is contradictory, or if the
                results index cannot be opened or holds no completed ingest of
                this results root.
        """
        if has_offset_file and has_no_offset_file:
            raise ValueError('has_offset_file and has_no_offset_file are mutually exclusive')
        if has_png_file and has_no_png_file:
            raise ValueError('has_png_file and has_no_png_file are mutually exclusive')
        if has_offset_spice_error and has_offset_nonspice_error:
            raise ValueError(
                'has_offset_spice_error and has_offset_nonspice_error are mutually exclusive'
            )
        needs_metadata_read = (
            has_offset_error or has_offset_spice_error or has_offset_nonspice_error
        )
        if needs_metadata_read and has_no_offset_file:
            raise ValueError(
                'has_no_offset_file contradicts the offset-error filters, which '
                'require the offset metadata file to exist'
            )

        self._has_no_offset_file = has_no_offset_file
        self._has_no_png_file = has_no_png_file
        self._has_offset_spice_error = has_offset_spice_error
        self._has_offset_nonspice_error = has_offset_nonspice_error
        # The error filters read the metadata file, so it must exist; fold them
        # into the presence filter so the walked set prunes candidates first.
        self._needs_offset_presence = has_offset_file or needs_metadata_read
        self._needs_png_presence = has_png_file
        self._needs_metadata_read = needs_metadata_read
        if isinstance(nav_results_root, FCPath):
            self._nav_results_root = nav_results_root
        else:
            # Results are not shared with other processes and may change between
            # runs, so use a private temporary cache like the writers do.
            self._nav_results_root = FileCache(None).new_path(nav_results_root)
        self._logger = logger
        self._offset_rel_paths: set[str] = set()
        self._png_rel_paths: set[str] = set()
        self._error_stubs: frozenset[str] = frozenset()
        self._from_index = results_db_url is not None
        # The index answers every filter, including the absence filters, which
        # a tree read answers only when a walk happened for another reason.
        self._have_result_sets = (
            self._from_index or self._needs_offset_presence or self._needs_png_presence
        )
        if results_db_url is not None:
            self._read_index(
                results_db_url,
                volumes,
                has_offset_error=has_offset_error,
                has_offset_spice_error=has_offset_spice_error,
                has_offset_nonspice_error=has_offset_nonspice_error,
            )
        elif self._have_result_sets:
            self._scan_volumes(volumes)

    @property
    def needs_batch_filtering(self) -> bool:
        """True when :meth:`filter_batch` performs any work.

        The caller uses this to decide whether to buffer accepted images into
        batches (amortizing the batched ``exists()`` / ``retrieve()`` round
        trips) or to yield them immediately.  When the results tree was walked,
        the absence filters are answered from the walked sets in
        :meth:`passes_presence` instead and cost nothing here.  When a results
        index answered the enumeration, so is every other filter, and nothing
        is left to do per batch.
        """
        if self._from_index:
            return False
        if self._needs_metadata_read:
            return True
        return not self._have_result_sets and (self._has_no_offset_file or self._has_no_png_file)

    def _read_index(
        self,
        results_db_url: str,
        volumes: Iterable[str],
        *,
        has_offset_error: bool,
        has_offset_spice_error: bool,
        has_offset_nonspice_error: bool,
    ) -> None:
        """Reads what the results root holds from the results index.

        Parameters:
            results_db_url: Connection URL of the results index.
            volumes: Volume names to read results for.
            has_offset_error: Whether any fatal error is wanted.
            has_offset_spice_error: Whether only a missing-SPICE-data error is
                wanted.
            has_offset_nonspice_error: Whether only a fatal error other than
                missing SPICE data is wanted.

        Raises:
            ValueError: If the index cannot be opened or holds no completed
                ingest of this results root.
        """
        # Imported here rather than at the top of the module, on the same
        # grounds as the GUI imports elsewhere in the package: this module is
        # reached by importing spindoctor.dataset, which every navigation run
        # does, and SQLAlchemy has no business on that path when no index was
        # named.
        from spindoctor.results_index.selection import read_result_stubs

        stubs = read_result_stubs(
            results_db_url,
            self._nav_results_root,
            volumes,
            has_offset_error=has_offset_error,
            has_offset_spice_error=has_offset_spice_error,
            has_offset_nonspice_error=has_offset_nonspice_error,
        )
        self._offset_rel_paths = {stub + METADATA_SUFFIX for stub in stubs.with_metadata}
        self._png_rel_paths = {stub + SUMMARY_PNG_SUFFIX for stub in stubs.with_summary_png}
        self._error_stubs = stubs.matching_error
        self._logger.info(
            '*** Results index holds %d offset metadata and %d summary PNG files under %s, '
            'ingested %s',
            len(self._offset_rel_paths),
            len(self._png_rel_paths),
            self._nav_results_root,
            _snapshot_age(stubs.ingested_utc),
        )
        if stubs.directories_missed:
            # The absence filters read "no row" as "this image was never
            # navigated", and under a directory nobody listed that reading is
            # simply false. The run completed all the same, so this count is
            # the only place the gap shows.
            self._logger.warning(
                'The last ingest of %s did not list %d director%s: an image under one of '
                'them is absent from the index whether or not it was navigated',
                self._nav_results_root,
                stubs.directories_missed,
                'y' if stubs.directories_missed == 1 else 'ies',
            )

    def _scan_volumes(self, volumes: Iterable[str]) -> None:
        """Walks the results tree under each volume, collecting result files.

        One directory walk per volume, restricted to the selected volumes so
        unrelated results are never listed.  Both result-file suffixes are
        collected in the single walk.  A volume with no results directory is
        treated as having no result files.

        Parameters:
            volumes: Volume names to walk under the results root.
        """
        root_prefix = self._nav_results_root.as_posix().rstrip('/') + '/'
        for volume in volumes:
            volume_dir = self._nav_results_root / volume
            try:
                for dir_path, _dir_names, file_names in volume_dir.walk():
                    dir_posix = dir_path.as_posix()
                    if not dir_posix.startswith(root_prefix):
                        continue
                    rel_dir = dir_posix[len(root_prefix) :]
                    for file_name in file_names:
                        rel_path = f'{rel_dir}/{file_name}'
                        if file_name.endswith(METADATA_SUFFIX):
                            self._offset_rel_paths.add(rel_path)
                        elif file_name.endswith(SUMMARY_PNG_SUFFIX):
                            self._png_rel_paths.add(rel_path)
            except (FileNotFoundError, NotADirectoryError):
                # A volume with no results directory (or whose results path is
                # not a directory) simply has no result files. Any other OSError
                # (permission denied, network or cloud-backend failure) is a real
                # scan failure that would silently corrupt the filter result, so
                # it is allowed to propagate.
                continue
        self._logger.info(
            '*** Results scan found %d offset metadata and %d summary PNG files under %s',
            len(self._offset_rel_paths),
            len(self._png_rel_paths),
            self._nav_results_root,
        )

    def passes_presence(self, results_path_stub: str) -> bool:
        """True if the image passes the filters answerable from the collected sets.

        Covers the presence filters and, when the results tree was walked
        anyway, the absence filters too (a set lookup instead of a per-file
        ``exists()`` round trip).  When the sets came from a results index they
        cover the error filters as well, since the query already read what the
        tree path has to open each metadata file for.

        Parameters:
            results_path_stub: The image's results path stub (relative to the
                results root, no suffix).

        Returns:
            True if every active filter answerable from the collected sets is
            satisfied.
        """
        if not self._have_result_sets:
            return True
        metadata_rel_path = results_path_stub + METADATA_SUFFIX
        png_rel_path = results_path_stub + SUMMARY_PNG_SUFFIX
        if self._needs_offset_presence and metadata_rel_path not in self._offset_rel_paths:
            return False
        if self._needs_png_presence and png_rel_path not in self._png_rel_paths:
            return False
        if self._has_no_offset_file and metadata_rel_path in self._offset_rel_paths:
            return False
        if self._has_no_png_file and png_rel_path in self._png_rel_paths:
            return False
        if not self._from_index or not self._needs_metadata_read:
            return True
        return results_path_stub in self._error_stubs

    def filter_batch(self, image_files: list[ImageFile]) -> list[ImageFile]:
        """Applies the absence and metadata-content filters to a batch.

        Input order is preserved.  Absence filters (when the results tree was
        not walked) are answered with one batched ``exists()`` call covering
        every active absence suffix.  The error filters retrieve all metadata
        files in one batched call and inspect their ``status`` /
        ``status_error`` fields.  A filter answered from a results index has
        nothing left to apply here and returns the batch as it was given.

        Parameters:
            image_files: Batch of images that already passed the cheap filters.

        Returns:
            The images that also pass the absence and error filters, in input
            order.
        """
        keep = image_files
        if not keep or not self.needs_batch_filtering:
            return keep

        absence_suffixes: list[str] = []
        if not self._have_result_sets:
            if self._has_no_offset_file:
                absence_suffixes.append(METADATA_SUFFIX)
            if self._has_no_png_file:
                absence_suffixes.append(SUMMARY_PNG_SUFFIX)
        if absence_suffixes:
            sub_paths: list[str | Path] = [
                f.results_path_stub + suffix for f in keep for suffix in absence_suffixes
            ]
            found = cast(list[bool], self._nav_results_root.exists(sub_paths))
            n_suffixes = len(absence_suffixes)
            keep = [
                f
                for i, f in enumerate(keep)
                if not any(found[i * n_suffixes : (i + 1) * n_suffixes])
            ]

        if self._needs_metadata_read and keep:
            metadata_sub_paths: list[str | Path] = [
                f.results_path_stub + METADATA_SUFFIX for f in keep
            ]
            local_paths = cast(
                list[Path | Exception],
                self._nav_results_root.retrieve(metadata_sub_paths, exception_on_fail=False),
            )
            keep = [
                f
                for f, local_path in zip(keep, local_paths, strict=True)
                if not isinstance(local_path, BaseException)
                and self._metadata_matches(f, local_path)
            ]

        return keep

    def _metadata_matches(self, image_file: ImageFile, local_path: Path) -> bool:
        """True if the image's metadata file satisfies the error filters.

        A metadata file that cannot be read, cannot be decoded as UTF-8, does
        not parse as JSON, or does not parse to a JSON object excludes its image
        with a logged warning rather than aborting the enumeration.

        Parameters:
            image_file: The candidate image (for the warning message only).
            local_path: Local path of the retrieved metadata JSON file.
        """
        try:
            parsed: Any = json.loads(local_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._logger.warning(
                'Excluding %s: unreadable metadata file: %s',
                image_file.results_path_stub,
                exc,
            )
            return False
        if not isinstance(parsed, dict):
            self._logger.warning(
                'Excluding %s: metadata JSON is not an object',
                image_file.results_path_stub,
            )
            return False
        metadata: dict[str, Any] = parsed
        if metadata.get('status') != 'error':
            return False
        status_error = metadata.get('status_error')
        if self._has_offset_spice_error and status_error != _SPICE_STATUS_ERROR:
            return False
        return not (self._has_offset_nonspice_error and status_error == _SPICE_STATUS_ERROR)
