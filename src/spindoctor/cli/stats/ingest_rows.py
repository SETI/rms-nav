"""Turn one navigation metadata document into the rows the results index holds.

The column mapping lives here, apart from the walk and the writer, because it
is the part that has to be read against the document the navigator writes.

Two mappings are easy to get wrong and are called out where they are made.  The
offset comes from the document's top-level ``offset`` and is stored as written:
``navigation_result.offset_px`` is the same offset rounded for display, and the
index carries the value consumers apply rather than the one an operator reads.
``status_error`` and ``status_reason`` are different vocabularies -- one is
matched verbatim by a selection filter, the other explains a non-success
outcome -- so they are kept in different columns rather than merged into one.

A file that carries neither an observation image name nor an observation
instrument is not a per-image navigation document at all.  A results tree holds
such files, so this refuses them by raising :class:`MetadataDocumentError`,
which carries the reason apart from the file name; the caller counts them and
goes on.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

from spindoctor.cli.stats.classify import date_from_image_et, image_number_from_name

__all__ = [
    'ImageRows',
    'MetadataDocumentError',
    'MetadataSource',
    'rows_from_metadata',
]


class MetadataDocumentError(ValueError):
    """A file ingest cannot read as a current-schema navigation document.

    Carries the reason apart from the file name so a run that meets several
    hundred such files can tally them by cause rather than printing several
    hundred lines that each have to be read individually.
    """

    def __init__(self, reason: str, *, source_file: str) -> None:
        """Record the reason and the file it applies to.

        Parameters:
            reason: What is wrong with the document, with nothing in it that
                identifies the individual file.
            source_file: Path or URL of the file, for the message.
        """
        super().__init__(f'{source_file}: {reason}')
        self.reason = reason


@dataclass(frozen=True)
class MetadataSource:
    """Where one metadata document came from and what the walk saw of it.

    Parameters:
        root_url: Normalized URL of the results root the document lives under.
        results_path_stub: The document's path under that root, without the
            ``_metadata.json`` suffix.
        source_file: Full path or URL of the document, recorded as provenance.
        mtime_ns: Modification time from the listing entry, or None when the
            backend's listing does not report one.
        size_bytes: Size from the listing entry, or None when the backend's
            listing does not report one.
        has_summary_png: Whether the walk saw a summary PNG beside it.
    """

    root_url: str
    results_path_stub: str
    source_file: str
    mtime_ns: int | None
    size_bytes: int | None
    has_summary_png: bool


@dataclass(frozen=True)
class ImageRows:
    """The rows one metadata document becomes.

    Parameters:
        image: The ``images`` row.
        techniques: The ``techniques`` rows, one per technique that reported.
        feature_sources: The ``feature_sources`` rows, one per feature type and
            source.
    """

    image: dict[str, Any]
    techniques: list[dict[str, Any]]
    feature_sources: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# One document into rows
# ---------------------------------------------------------------------------


def _finite_or_none(value: Any) -> float | None:
    """Coerce a JSON value to a finite float, or None.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The float, or None when the value is absent, is not a number, is a
        boolean, or is not finite.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    """Coerce a JSON value to an integer, or None.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The integer, or None when the value is absent or is not an integer.
        A boolean is refused: it is an ``int`` in Python and an identifier
        nowhere.

    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _str_or_none(value: Any) -> str | None:
    """Coerce a JSON value to a non-empty string, or None.

    An empty string becomes None so that the reason columns hold either a
    reason or nothing.  A ``COALESCE`` over them then falls through in exactly
    the cases the reader expects it to.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The string, or None when it is absent, empty, or not a string.
    """
    if isinstance(value, str) and value:
        return value
    return None


def _pair(value: Any) -> tuple[float | None, float | None]:
    """Split a two-element JSON list into a pair of finite floats.

    Parameters:
        value: The value as it was parsed, or None.

    Returns:
        The two values, each None when absent or unusable.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    return _finite_or_none(value[0]), _finite_or_none(value[1])


def _image_shape(value: Any) -> tuple[int | None, int | None]:
    """Per-axis ``(v, u)`` pixel dimensions from a metadata image_shape list.

    Parameters:
        value: The recorded ``observation.image_shape``.

    Returns:
        The two dimensions, each None when the value is not a usable pair.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None, None


def _covariance_block(covariance: Any) -> tuple[float | None, float | None, float | None]:
    """The 2x2 offset block of a recorded covariance matrix.

    A twist-fitted result records a 3x3 matrix whose third row and column
    describe the rotation.  Only the offset block is indexed; the rotation's
    uncertainty is carried by ``sigma_rotation_deg`` alone.

    Parameters:
        covariance: The recorded ``covariance_px2``.

    Returns:
        ``(vv, vu, uu)``, each None when the matrix is absent or unusable.
    """
    if not isinstance(covariance, list) or len(covariance) < 2:
        return None, None, None
    try:
        return (
            _finite_or_none(covariance[0][0]),
            _finite_or_none(covariance[0][1]),
            _finite_or_none(covariance[1][1]),
        )
    except (TypeError, IndexError, KeyError):
        return None, None, None


def _sigma_from_covariance(covariance: Any) -> tuple[float | None, float | None]:
    """Per-axis 1-sigma pair from a curated covariance matrix (or None pair).

    Parameters:
        covariance: The recorded ``covariance_px2`` of one technique.

    Returns:
        ``(sigma_dv, sigma_du)``, each None when the variance is absent or
        negative.
    """
    if not isinstance(covariance, list) or len(covariance) < 2:
        return None, None
    try:
        var_dv = float(covariance[0][0])
        var_du = float(covariance[1][1])
    except (TypeError, ValueError, IndexError):
        return None, None
    sigma_dv = math.sqrt(var_dv) if var_dv >= 0.0 else None
    sigma_du = math.sqrt(var_du) if var_du >= 0.0 else None
    return sigma_dv, sigma_du


def _cmatrix_or_none(value: Any) -> list[float] | None:
    """Coerce a recorded rotation matrix to nine floats, or None.

    Parameters:
        value: The recorded matrix, row-major, or None.

    Returns:
        The nine values, or None when the matrix is absent or is not nine
        numbers.
    """
    if not isinstance(value, list) or len(value) != 9:
        return None
    coerced = [_finite_or_none(entry) for entry in value]
    if any(entry is None for entry in coerced):
        return None
    return cast(list[float], coerced)


def _source_names_from_feature_ids(feature_ids: Any) -> list[str]:
    """Body / ring / catalog names parsed from curated feature ids.

    Feature ids follow ``kind:NAME[...]`` (``body_disc:IAPETUS``,
    ``ring_edge:SATURN:feature_135_ieg:IEG``, ``star:UCAC4:10230452``).

    Parameters:
        feature_ids: The technique's recorded feature id list.

    Returns:
        The distinct names, sorted.
    """
    names: set[str] = set()
    if not isinstance(feature_ids, list):
        return []
    for feature_id in feature_ids:
        if not isinstance(feature_id, str):
            continue
        parts = feature_id.split(':')
        if len(parts) >= 2 and parts[1]:
            names.add(parts[1])
    return sorted(names)


def _volume_of(results_path_stub: str) -> str | None:
    """The volume a stub names, or None when it names none.

    Parameters:
        results_path_stub: The stub.

    Returns:
        The first path segment, or None for a stub with no separator -- which
        is what the simulated dataset's bare scene basenames produce.
    """
    volume, separator, _rest = results_path_stub.partition('/')
    return volume if separator else None


def _technique_rows(source: MetadataSource, per_technique: Any) -> list[dict[str, Any]]:
    """Build the ``techniques`` rows of one document.

    Parameters:
        source: Where the document came from.
        per_technique: The recorded ``navigation_result.per_technique`` list.

    Returns:
        One row per technique that reported, in recorded order.
    """
    rows: list[dict[str, Any]] = []
    for entry in per_technique:
        offset_dv, offset_du = _pair(entry.get('offset_px'))
        sigma_dv, sigma_du = _sigma_from_covariance(entry.get('covariance_px2'))
        rows.append(
            {
                'root_url': source.root_url,
                'results_path_stub': source.results_path_stub,
                'technique_name': entry.get('technique_name', 'unknown'),
                'offset_dv': offset_dv,
                'offset_du': offset_du,
                'sigma_dv': sigma_dv,
                'sigma_du': sigma_du,
                'confidence': _finite_or_none(entry.get('confidence')),
                'spurious': bool(entry.get('spurious')),
                'at_edge': bool(entry.get('at_edge')),
                'source_names': _source_names_from_feature_ids(entry.get('feature_ids')),
                'diagnostics': entry.get('diagnostics') or {},
            }
        )
    return rows


def _feature_source_rows(source: MetadataSource, inventory: Any) -> list[dict[str, Any]]:
    """Build the ``feature_sources`` rows of one document.

    The inventory is aggregated by ``(feature_type, source_model, source_name)``;
    per-feature identity and gating detail is not retained.

    Parameters:
        source: Where the document came from.
        inventory: The recorded ``navigation_result.feature_inventory`` list.

    Returns:
        One row per distinct feature type and source, in key order.
    """
    counts: dict[tuple[str, str, str], list[int]] = {}
    for entry in inventory:
        feature_id = str(entry.get('feature_id', ''))
        parts = feature_id.split(':')
        name = parts[1] if len(parts) >= 2 and parts[1] else '(none)'
        key = (
            str(entry.get('feature_type', 'unknown')),
            str(entry.get('source_model', 'unknown')),
            name,
        )
        tally = counts.setdefault(key, [0, 0])
        tally[0] += 1
        tally[1] += int(bool(entry.get('gated')))
    return [
        {
            'root_url': source.root_url,
            'results_path_stub': source.results_path_stub,
            'feature_type': feature_type,
            'source_model': source_model,
            'source_name': source_name,
            'n_features': tally[0],
            'n_gated': tally[1],
        }
        for (feature_type, source_model, source_name), tally in sorted(counts.items())
    ]


def rows_from_metadata(metadata: dict[str, Any], source: MetadataSource) -> ImageRows:
    """Flatten one metadata document into the rows the index holds.

    The offset comes from the document's top-level ``offset`` and is stored as
    written.  ``navigation_result.offset_px`` is the same offset rounded for
    display and is deliberately not what the index carries, because every
    consumer applies the value rather than reading it.

    ``status_error`` and ``status_reason`` are different vocabularies and are
    kept in different columns: ``status_error`` is what a selection filter
    matches verbatim, ``status_reason`` is the navigator's explanation of a
    non-success outcome.

    Parameters:
        metadata: Parsed metadata JSON as written by ``navigate_image_files``.
        source: Where the document came from and what the walk saw of it.

    Returns:
        The image row and its child rows.

    Raises:
        MetadataDocumentError: If the document lacks the observation image name
            or the observation instrument, which is what a file that is not a
            per-image navigation document looks like.
    """
    observation = metadata.get('observation') or {}
    image_name = observation.get('image_name')
    if not image_name or not isinstance(image_name, str):
        raise MetadataDocumentError('no observation.image_name', source_file=source.source_file)
    instrument = observation.get('instrument')
    if not isinstance(instrument, str) or not instrument:
        raise MetadataDocumentError('no observation.instrument', source_file=source.source_file)
    nav = metadata.get('navigation_result') or {}
    provenance = nav.get('provenance') or {}
    classifier = nav.get('image_classifier') or {}
    times = nav.get('times') or {}
    pointing = nav.get('pointing') or {}
    # A navigated image's epoch comes from its observation (provenance); an
    # image that never loaded has no provenance, so the navigator records the
    # epoch it read from the index under ``observation.image_et``.  Either way
    # every image is placed in time.
    image_et = _finite_or_none(provenance.get('image_et'))
    if image_et is None:
        image_et = _finite_or_none(observation.get('image_et'))
    per_technique = nav.get('per_technique') or []
    timing = metadata.get('timing') or {}
    shape_v, shape_u = _image_shape(observation.get('image_shape'))
    offset_dv, offset_du = _pair(metadata.get('offset'))
    sigma_dv, sigma_du = _pair(nav.get('sigma_px'))
    covariance_vv, covariance_vu, covariance_uu = _covariance_block(nav.get('covariance_px2'))

    image_row: dict[str, Any] = {
        'root_url': source.root_url,
        'results_path_stub': source.results_path_stub,
        'volume': _volume_of(source.results_path_stub),
        'image_name': image_name,
        'instrument': instrument,
        # Present whenever the dataset index supplied it, including for an
        # image that never loaded; NULL only when the dataset has no camera
        # column at all.  It is never inferred from the image name.
        'camera': _str_or_none(observation.get('camera')),
        'image_path': observation.get('image_path'),
        'image_et': image_et,
        'image_date': date_from_image_et(image_et),
        'status': str(metadata.get('status') or nav.get('status') or 'unknown'),
        'status_error': _str_or_none(metadata.get('status_error')),
        'status_reason': _str_or_none(nav.get('status_reason')),
        'offset_dv': offset_dv,
        'offset_du': offset_du,
        'sigma_dv': sigma_dv,
        'sigma_du': sigma_du,
        'covariance_vv': covariance_vv,
        'covariance_vu': covariance_vu,
        'covariance_uu': covariance_uu,
        'sigma_along_unobservable_px': _finite_or_none(nav.get('sigma_along_unobservable_px')),
        'rotation_deg': _finite_or_none(nav.get('rotation_deg')),
        'sigma_rotation_deg': _finite_or_none(nav.get('sigma_rotation_deg')),
        'confidence': _finite_or_none(metadata.get('confidence')),
        'confidence_rank': _str_or_none(nav.get('confidence_rank')),
        'n_techniques': len(per_technique),
        'excluded_from_consensus': sorted(nav.get('excluded_from_consensus') or []),
        'image_class': _str_or_none(classifier.get('class')),
        'noise_sigma': _finite_or_none(classifier.get('noise_sigma')),
        'image_shape_v': shape_v,
        'image_shape_u': shape_u,
        'run_start': _str_or_none(timing.get('start_iso8601')),
        'run_end': _str_or_none(timing.get('end_iso8601')),
        'elapsed_s': _finite_or_none(timing.get('elapsed_s')),
        'config_hash': _str_or_none(provenance.get('config_hash')),
        'git_sha': _str_or_none(provenance.get('spindoctor_git_sha')),
        'pipeline_run': _str_or_none(provenance.get('pipeline_run_iso8601')),
        'image_number': image_number_from_name(image_name),
        'has_summary_png': source.has_summary_png,
        'start_et': _finite_or_none(times.get('start_et')),
        'stop_et': _finite_or_none(times.get('stop_et')),
        'exposure_s': _finite_or_none(times.get('exposure_s')),
        'sclk_start': _str_or_none(times.get('sclk_start')),
        'sclk_midtime': _str_or_none(times.get('sclk_midtime')),
        'sclk_stop': _str_or_none(times.get('sclk_stop')),
        'camera_frame_id': _int_or_none(pointing.get('camera_frame_id')),
        'ck_frame_id': _int_or_none(pointing.get('ck_frame_id')),
        'cmatrix': _cmatrix_or_none(pointing.get('cmatrix')),
        'cmatrix_original': _cmatrix_or_none(pointing.get('cmatrix_original')),
        'source_file': source.source_file,
        'mtime_ns': source.mtime_ns,
        'size_bytes': source.size_bytes,
    }
    return ImageRows(
        image=image_row,
        techniques=_technique_rows(source, per_technique),
        feature_sources=_feature_source_rows(source, nav.get('feature_inventory') or []),
    )
