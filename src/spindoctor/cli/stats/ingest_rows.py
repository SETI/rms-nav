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

Every container the document schema declares is checked before it is read.  A
file whose ``observation`` is a string, or whose ``per_technique`` holds
numbers, is a document of some other shape rather than a navigation result, and
the refusal names which field said so.  Reading it without the check would raise
an ``AttributeError`` out of the middle of a run and cost every other file in
the tree, so the shape is part of what "current-schema" means here.
"""

import math
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from spindoctor.cli.stats.classify import date_from_image_et, image_number_from_name

__all__ = [
    'ImageRows',
    'MetadataDocumentError',
    'MetadataSource',
    'rows_from_metadata',
]

NOT_A_NAVIGATION_DOCUMENT = 'not a current-schema navigation document'
"""Opening of every refusal of a file that is some other kind of document.

A results tree holds hundreds of ``*_metadata.json`` files that were never
navigation results.  The tally an operator reads has to say that in as many
words, because "no observation.instrument" on its own reads as a navigation
result that failed to ingest.
"""


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
# Shapes the document schema declares
# ---------------------------------------------------------------------------


def _refuse(detail: str, source: MetadataSource) -> NoReturn:
    """Refuse a file that is not a current-schema navigation document.

    Parameters:
        detail: Which field said so, with nothing file-specific in it.
        source: Where the document came from.

    Raises:
        MetadataDocumentError: Always.
    """
    raise MetadataDocumentError(
        f'{NOT_A_NAVIGATION_DOCUMENT} ({detail})', source_file=source.source_file
    )


def _object(value: Any, name: str, source: MetadataSource) -> dict[str, Any]:
    """Read a JSON object the schema declares, or an empty one when it is absent.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The object, or an empty one when the field is absent.

    Raises:
        MetadataDocumentError: If the field is present and is not an object.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        _refuse(f'{name} is not an object', source)
    return value


def _array(value: Any, name: str, source: MetadataSource) -> list[Any]:
    """Read a JSON array the schema declares, or an empty one when it is absent.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The array, or an empty one when the field is absent.

    Raises:
        MetadataDocumentError: If the field is present and is not an array.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        _refuse(f'{name} is not a list', source)
    return value


def _array_of_objects(value: Any, name: str, source: MetadataSource) -> list[dict[str, Any]]:
    """Read a JSON array of objects the schema declares.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The entries, or an empty list when the field is absent.

    Raises:
        MetadataDocumentError: If the field is not an array, or holds an entry
            that is not an object.
    """
    entries = _array(value, name, source)
    if not all(isinstance(entry, dict) for entry in entries):
        _refuse(f'{name} holds an entry that is not an object', source)
    return cast(list[dict[str, Any]], entries)


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
    except (TypeError, ValueError, IndexError, KeyError):
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


def _technique_rows(
    source: MetadataSource, per_technique: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the ``techniques`` rows of one document.

    Parameters:
        source: Where the document came from.
        per_technique: The recorded ``navigation_result.per_technique`` list.

    Returns:
        One row per technique that reported, in recorded order.

    Raises:
        MetadataDocumentError: If an entry's ``diagnostics`` is not an object.
    """
    rows: list[dict[str, Any]] = []
    for entry in per_technique:
        offset_dv, offset_du = _pair(entry.get('offset_px'))
        sigma_dv, sigma_du = _sigma_from_covariance(entry.get('covariance_px2'))
        rows.append(
            {
                'root_url': source.root_url,
                'results_path_stub': source.results_path_stub,
                'technique_name': _str_or_none(entry.get('technique_name')) or 'unknown',
                'offset_dv': offset_dv,
                'offset_du': offset_du,
                'sigma_dv': sigma_dv,
                'sigma_du': sigma_du,
                'confidence': _finite_or_none(entry.get('confidence')),
                'spurious': bool(entry.get('spurious')),
                'at_edge': bool(entry.get('at_edge')),
                # An empty list is a statement: this technique named no source.
                'source_names': _source_names_from_feature_ids(entry.get('feature_ids')),
                # As is an empty object: it reported no diagnostics.
                'diagnostics': _object(
                    entry.get('diagnostics'),
                    'navigation_result.per_technique[].diagnostics',
                    source,
                ),
            }
        )
    return rows


def _feature_source_rows(
    source: MetadataSource, inventory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
            or the observation instrument, or if any container the schema
            declares holds something of another shape.  That is what a file
            which is not a per-image navigation document looks like.
    """
    observation = _object(metadata.get('observation'), 'observation', source)
    image_name = _str_or_none(observation.get('image_name'))
    if image_name is None:
        _refuse('no observation.image_name', source)
    instrument = _str_or_none(observation.get('instrument'))
    if instrument is None:
        _refuse('no observation.instrument', source)
    nav = _object(metadata.get('navigation_result'), 'navigation_result', source)
    provenance = _object(nav.get('provenance'), 'navigation_result.provenance', source)
    classifier = _object(nav.get('image_classifier'), 'navigation_result.image_classifier', source)
    times = _object(nav.get('times'), 'navigation_result.times', source)
    pointing = _object(nav.get('pointing'), 'navigation_result.pointing', source)
    # A navigated image's epoch comes from its observation (provenance); an
    # image that never loaded has no provenance, so the navigator records the
    # epoch it read from the index under ``observation.image_et``.  Either way
    # every image is placed in time.
    image_et = _finite_or_none(provenance.get('image_et'))
    if image_et is None:
        image_et = _finite_or_none(observation.get('image_et'))
    per_technique = _array_of_objects(
        nav.get('per_technique'), 'navigation_result.per_technique', source
    )
    inventory = _array_of_objects(
        nav.get('feature_inventory'), 'navigation_result.feature_inventory', source
    )
    excluded = _array(
        nav.get('excluded_from_consensus'), 'navigation_result.excluded_from_consensus', source
    )
    if not all(isinstance(name, str) for name in excluded):
        _refuse(
            'navigation_result.excluded_from_consensus holds a name that is not a string', source
        )
    timing = _object(metadata.get('timing'), 'timing', source)
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
        'image_path': _str_or_none(observation.get('image_path')),
        'image_et': image_et,
        'image_date': date_from_image_et(image_et),
        'status': _str_or_none(metadata.get('status'))
        or _str_or_none(nav.get('status'))
        or 'unknown',
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
        # An empty list is a statement: the ensemble excluded nothing.
        'excluded_from_consensus': sorted(excluded),
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
        feature_sources=_feature_source_rows(source, inventory),
    )
