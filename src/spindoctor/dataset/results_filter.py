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
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from pdslogger import PdsLogger

from .dataset import ImageFile

METADATA_SUFFIX = '_metadata.json'
"""Suffix of the per-image offset metadata file under the results root."""

SUMMARY_PNG_SUFFIX = '_summary.png'
"""Suffix of the per-image summary PNG file under the results root."""

RESULTS_FILTER_BATCH_SIZE = 64
"""Number of images checked per batched ``exists()`` / ``retrieve()`` call."""

_SPICE_STATUS_ERROR = 'missing_spice_data'


class ResultsFilter:
    """Filters candidate images against their navigation result files.

    Constructed once per enumeration when any of the results-based selection
    flags is active.  Construction validates the flag combination (the flags
    AND together; directly contradictory pairs raise) and, when a presence or
    error filter is active, walks the results tree under each selected volume
    to collect the existing result files.

    The filter is applied in two stages:

    - :meth:`passes_presence` is a cheap per-row set-membership test used
      while scanning index rows.
    - :meth:`filter_batch` applies the absence and metadata-content filters
      to a batch of already-accepted images with one batched ``exists()``
      and/or ``retrieve()`` call, preserving input order.
    """

    def __init__(
        self,
        volumes: Iterable[str],
        nav_results_root: FCPath,
        *,
        has_offset_file: bool = False,
        has_no_offset_file: bool = False,
        has_png_file: bool = False,
        has_no_png_file: bool = False,
        has_offset_error: bool = False,
        has_offset_spice_error: bool = False,
        has_offset_nonspice_error: bool = False,
        logger: PdsLogger,
    ) -> None:
        """Validates the flag combination and scans the results tree if needed.

        Parameters:
            volumes: Volume names selected by the other constraints; only these
                subdirectories of the results root are walked.
            nav_results_root: Root of the navigation results tree; may be a
                cloud URL.
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
            logger: Logger for scan statistics and unreadable-metadata warnings.

        Raises:
            ValueError: If the flag combination is contradictory.
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
        self._nav_results_root = nav_results_root
        self._logger = logger
        self._offset_rel_paths: set[str] = set()
        self._png_rel_paths: set[str] = set()
        self._walked = self._needs_offset_presence or self._needs_png_presence
        if self._walked:
            self._scan_volumes(volumes)

    @property
    def needs_batch_filtering(self) -> bool:
        """True when :meth:`filter_batch` performs any work.

        The caller uses this to decide whether to buffer accepted images into
        batches (amortizing the batched ``exists()`` / ``retrieve()`` round
        trips) or to yield them immediately.  When the results tree was walked,
        the absence filters are answered from the walked sets in
        :meth:`passes_presence` instead and cost nothing here.
        """
        if self._needs_metadata_read:
            return True
        return not self._walked and (self._has_no_offset_file or self._has_no_png_file)

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
            except OSError:
                continue
        self._logger.info(
            f'*** Results scan found {len(self._offset_rel_paths)} offset metadata and '
            f'{len(self._png_rel_paths)} summary PNG files under {self._nav_results_root}'
        )

    def passes_presence(self, results_path_stub: str) -> bool:
        """True if the image passes the filters answerable from the walked sets.

        Covers the presence filters and, when the results tree was walked
        anyway, the absence filters too (a set lookup instead of a per-file
        ``exists()`` round trip).

        Parameters:
            results_path_stub: The image's results path stub (relative to the
                results root, no suffix).

        Returns:
            True if every active filter answerable from the walked sets is
            satisfied.
        """
        if not self._walked:
            return True
        metadata_rel_path = results_path_stub + METADATA_SUFFIX
        png_rel_path = results_path_stub + SUMMARY_PNG_SUFFIX
        if self._needs_offset_presence and metadata_rel_path not in self._offset_rel_paths:
            return False
        if self._needs_png_presence and png_rel_path not in self._png_rel_paths:
            return False
        if self._has_no_offset_file and metadata_rel_path in self._offset_rel_paths:
            return False
        return not (self._has_no_png_file and png_rel_path in self._png_rel_paths)

    def filter_batch(self, image_files: list[ImageFile]) -> list[ImageFile]:
        """Applies the absence and metadata-content filters to a batch.

        Input order is preserved.  Absence filters (when the results tree was
        not walked) are answered with one batched ``exists()`` call covering
        every active absence suffix.  The error filters retrieve all metadata
        files in one batched call and inspect their ``status`` /
        ``status_error`` fields.

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
        if not self._walked:
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

        A metadata file that cannot be read or parsed excludes its image with
        a logged warning rather than aborting the enumeration.

        Parameters:
            image_file: The candidate image (for the warning message only).
            local_path: Local path of the retrieved metadata JSON file.
        """
        try:
            metadata: dict[str, Any] = json.loads(local_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning(
                f'Excluding {image_file.results_path_stub}: unreadable metadata file: {exc}'
            )
            return False
        if metadata.get('status') != 'error':
            return False
        status_error = metadata.get('status_error')
        if self._has_offset_spice_error and status_error != _SPICE_STATUS_ERROR:
            return False
        return not (self._has_offset_nonspice_error and status_error == _SPICE_STATUS_ERROR)
