"""The eight navigation documents the statistics fixture tree holds.

The statistics ingest and the report regression both run over
``data/results_tree``, and the frozen report output under ``data/golden`` is
what that tree produces.  A document there therefore has to be one the
pipeline could have written: anything else lets the ingest and every report
section be measured against key sets, vocabularies and value shapes no writer
emits, and the frozen output then freezes whatever those produced.

So the documents are built here, through the writer itself.  Seven of them are
the return value of
:func:`~spindoctor.navigate_image_files.build_metadata_from_result` over a real
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`; the eighth, whose
image never loads, is the return value of
:func:`~spindoctor.navigate_image_files.navigate_image_files` itself.  All eight
are serialized by the writer's own
:func:`~spindoctor.support.file.json_as_string`.  What is chosen here is only
what a navigation run reads from its image and its configuration: the geometry,
the scores, the epochs, the clock.

Two values a run takes from the wall clock are pinned instead, since a stored
document cannot hold one: the ``timing`` block is built by the writer's own
:func:`~spindoctor.navigate_image_files.build_timing_section` from fixed
moments, and ``provenance.pipeline_run_iso8601`` carries a fixed stamp in the
spelling the orchestrator produces.

Run this module as a script to write the tree again::

    PYTHONPATH=src python tests/spindoctor/cli/stats/results_tree_documents.py

The stored tree is then what the writer emits, and the frozen report output has
to be re-ratified against it.  ``test_results_tree_documents.py`` holds the two
against each other, so a writer change is reported here rather than being
absorbed silently by a tree nothing checks.

What the tree covers
--------------------

Every document earns its place, and regenerating one must not cost the report a
section or a column:

- **Three instruments**: two with SPICE camera frames (``coiss``, ``vgiss``)
  and the simulated scene, which is the one host that correctly records no
  attitude and no exposure times.
- **Two subtrees and a bare stub**: ``COISS_2001`` and ``VGISS_5101`` name a
  subtree; the simulated scene's basename names none, which is the case a stub
  with no separator produces.
- **Three outcomes**: five successes, two failed navigations, and one image
  whose load failed before an observation existed.  The last records no epoch,
  no image shape and no navigation result at all, so its date cells are empty
  and its reason is a ``status_error`` rather than a ``status_reason``.
- **Two failure reasons over two instruments**, each consistent with its own
  inventory: every feature gated on the Cassini failure, no feature at all on
  the Voyager one.
- **Four feature sources**: a body, a second body, a ring system and a star
  catalog, with gated features under two of them.
- **Four camera and image-size groups**, so the offset tables have more than
  one row to order.
- **A BOTSIM pair**: ``N1294561202`` and ``W1294561202`` share a shutter,
  a spacecraft clock and an epoch, and both carry ``shutter_mode`` of
  ``BOTSIM``.  The other Cassini images carry ``NACONLY``; Voyager and the
  simulated scene carry none, as their hosts read none.
- **A suspect offset**: one Cassini image's fused offset reaches the search
  limit for its size, and one Voyager size has no configured limit at all.
- **A spurious technique and an ensemble exclusion**, on separate techniques,
  since the ensemble drops a spurious result before consensus selection and can
  never report one as excluded.
- **Images with two contributing techniques**, which is what the
  cross-technique agreement and confidence-calibration sections measure.
- **A distinct elapsed time per image**, including the shortest on the image
  that failed to load.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.ensemble import derive_confidence_rank
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.navigate_image_files import (
    build_metadata_from_result,
    build_timing_section,
    navigate_image_files,
)
from spindoctor.obs import ObsCassiniISS
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution
from spindoctor.support.file import json_as_string
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'RESULTS_TREE',
    'results_tree_documents',
    'stored_documents',
    'write_results_tree',
]

RESULTS_TREE = Path(__file__).resolve().parent / 'data' / 'results_tree'
"""Where the stored tree lives."""

_COISS_SUBTREE = 'COISS_2001/data/1294561143_1295221348'
"""The Cassini volume and observation directory the Cassini images sit under."""

_VGISS_SUBTREE = 'VGISS_5101/data/C13854XX'
"""The Voyager volume and image directory the Voyager images sit under."""


# ---------------------------------------------------------------------------
# What every document of one run shares
# ---------------------------------------------------------------------------

_VERSION = '0.0.0'
"""Package version the run recorded."""

_GIT_SHA = '719cde5'
"""Short git SHA the run recorded."""

_CONFIG_HASH = '3ca76ec39b1fb875a86bed2793adc4430785242e07d705f2d65581963040a6b6'
"""Digest of the fully resolved configuration the run used."""

_PIPELINE_RUN = '2026-08-08T16:46:29Z'
"""When the run began, in the spelling the orchestrator stamps.

Seconds precision and a ``Z`` designator, which is what
``datetime.isoformat(timespec='seconds')`` produces for a UTC moment.
"""

_TECHNIQUE_NAMES = (
    'BodyBlobNav',
    'BodyDiscCorrelateNav',
    'BodyLimbNav',
    'BodyTerminatorNav',
    'RingAnnulusNav',
    'RingEdgeNav',
    'StarFieldFromCatalogNav',
    'StarRefineNav',
    'StarUniqueMatchNav',
    'TitanHazeNav',
)
"""Every technique the run had registered.

Written out rather than read from the registry: what a stored document holds is
what one run recorded, and a tree that changed shape whenever a technique was
added would move the frozen report for a reason the report is not about.
"""

_STATIC_DATA_HASHES = {
    'config_220_body_shape.yaml': (
        'ac10e82c9c141c0e449dcfc92d8c4f341400ffa51976f53c94c98eaabac7a52a'
    ),
    'config_310_saturn_rings.yaml': (
        '5f5f0b8f9a3b4d8c6e2a1c7d9b0e4f3a2d6c8b1e7f0a9d3c5b2e8f1a4d7c0b63'
    ),
    'config_400_inst_coiss.yaml': (
        '8c20d352ed0b5b690f7fc573f505f062551966c3305a01ae0e6fba63a8400f17'
    ),
}
"""Digests of the shipped static data the run hashed."""

_STAR_CATALOGS = {
    'tycho2': '/resources/SPICE/Stars',
    'ucac4': '/star-catalogs/UCAC4',
    'ybsc': '/star-catalogs/YBSC',
}
"""Where the run resolved each configured star catalog."""

_COISS_KERNELS = (
    '05138_05159ra.bc',
    'cas00172.tsc',
    'cpck15Dec2017.tpc',
    'naif0012.tls',
    'sat428.bsp',
)
"""Kernels loaded for the Cassini images."""

_VGISS_KERNELS = (
    'naif0012.tls',
    'vg100019.tsc',
    'vg1_saturn.bsp',
    'vg1_super.bc',
)
"""Kernels loaded for the Voyager images."""

_SIM_KERNELS: tuple[str, ...] = ()
"""Kernels loaded for the simulated scene, of which there are none.

An empty list is a statement about the run rather than an absent value, and a
simulated scene is the run that makes it.
"""

_CASSINI_OOPS_FROM_SPICE: NDArrayFloatType = np.diag([-1.0, -1.0, 1.0])
"""The constant rotation between the oops and SPICE Cassini ISS camera frames."""

_VOYAGER_OOPS_FROM_SPICE: NDArrayFloatType = np.eye(3)
"""The constant rotation between the oops and SPICE Voyager ISS camera frames."""


def _rotation(z_deg: float, y_deg: float, x_deg: float) -> NDArrayFloatType:
    """Return the proper rotation ``Rz . Ry . Rx`` for three angles.

    Parameters:
        z_deg: Rotation about the third axis, in degrees.
        y_deg: Rotation about the second axis, in degrees.
        x_deg: Rotation about the first axis, in degrees.

    Returns:
        The 3x3 rotation, orthonormal to float64 precision.
    """
    z, y, x = np.radians(np.array([z_deg, y_deg, x_deg], dtype=np.float64))
    rz = np.array(
        [[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(x), -np.sin(x)], [0.0, np.sin(x), np.cos(x)]],
        dtype=np.float64,
    )
    product: NDArrayFloatType = rz @ ry @ rx
    return product


def _provenance(
    *, image_et: float, kernels: tuple[str, ...], extractors: tuple[str, ...]
) -> Provenance:
    """Return the reproducibility envelope one image's run recorded.

    Parameters:
        image_et: The observation midtime, which is the image's epoch.
        kernels: Kernel basenames the run had loaded.
        extractors: Names of the models built for the observation.

    Returns:
        The envelope.
    """
    return Provenance(
        spindoctor_version=_VERSION,
        image_et=image_et,
        pipeline_run_iso8601=_PIPELINE_RUN,
        spindoctor_git_sha=_GIT_SHA,
        spice_kernels=kernels,
        static_data_hashes=_STATIC_DATA_HASHES,
        technique_names=_TECHNIQUE_NAMES,
        extractor_names=extractors,
        config_hash=_CONFIG_HASH,
        config_overrides=(),
        star_catalogs=_STAR_CATALOGS,
    )


def _classifier(
    *, noise_sigma: float, max_dn: float, gradient_score: float | None
) -> NavImageClassifierResult:
    """Return a clean-image classifier verdict.

    Parameters:
        noise_sigma: The MAD-based noise sigma the classifier measured.
        max_dn: The largest DN in the image.
        gradient_score: The background-gradient score, or None for an image
            whose downsample is perfectly flat.

    Returns:
        The verdict.
    """
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=noise_sigma,
        max_dn=max_dn,
        background_gradient_score=gradient_score,
        flags=[],
    )


def _pointing(
    *,
    camera_frame: str,
    camera_frame_id: int,
    ck_frame_id: int,
    oops_from_spice: NDArrayFloatType,
    midtime_et: float,
    exposure_s: float,
    sclk: tuple[str, str, str],
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
) -> PointingSolution:
    """Return the attitude solution the orchestrator stamps onto a result.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        camera_frame_id: SPICE id of that frame.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        oops_from_spice: Constant rotation between the two frame conventions.
        midtime_et: Exposure midtime, which is also the image's epoch.
        exposure_s: Exposure duration in seconds.
        sclk: Spacecraft clock strings at start, midtime and stop.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result that produced
            no offset and therefore no correction.

    Returns:
        The solution.
    """
    baseline = AttitudeBaseline(
        cmatrix_original=original,
        oops_from_spice=oops_from_spice,
        camera_frame=camera_frame,
        camera_frame_id=camera_frame_id,
        ck_frame_id=ck_frame_id,
        start_et=midtime_et - exposure_s / 2.0,
        stop_et=midtime_et + exposure_s / 2.0,
        midtime_et=midtime_et,
        exposure_s=exposure_s,
        sclk_start=sclk[0],
        sclk_midtime=sclk[1],
        sclk_stop=sclk[2],
    )
    return PointingSolution(baseline=baseline, cmatrix=corrected)


def _star(
    unique_number: int, *, reliability: float, snr_score: float, bbox: tuple[int, int, int, int]
) -> NavFeatureSummary:
    """Return one ungated catalog-star inventory entry.

    Parameters:
        unique_number: The star's UCAC4 identifier.
        reliability: The self-assessed score.
        snr_score: The detection component of that score.
        bbox: The star's extfov bounding box.

    Returns:
        The entry.
    """
    return NavFeatureSummary(
        feature_id=f'star:UCAC4:{unique_number}',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        reliability=reliability,
        gated=False,
        gate_reason=None,
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            predicted_snr=snr_score,
            in_body_silhouette=False,
            in_saturation_or_cosmic=False,
            smear_length_ok=True,
        ),
    )


def _faint_star(unique_number: int, *, bbox: tuple[int, int, int, int]) -> NavFeatureSummary:
    """Return one star gated out below the STAR reliability threshold.

    Parameters:
        unique_number: The star's UCAC4 identifier.
        bbox: The star's extfov bounding box.

    Returns:
        The entry, carrying the reason the gate writes.
    """
    return NavFeatureSummary(
        feature_id=f'star:UCAC4:{unique_number}',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        reliability=0.12,
        gated=True,
        gate_reason='reliability_0.120_below_threshold_0.200',
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            predicted_snr=0.12,
            in_body_silhouette=False,
            in_saturation_or_cosmic=False,
            smear_length_ok=True,
        ),
    )


def _ring_edge(
    ring_key: str,
    edge_label: str,
    *,
    reliability: float,
    gated: bool,
    gate_reason: str | None,
    bbox: tuple[int, int, int, int],
) -> NavFeatureSummary:
    """Return one Saturn ring-edge inventory entry.

    Parameters:
        ring_key: The catalog key of the ring feature the edge belongs to.
        edge_label: ``IEG`` or ``OEG`` for the two edges of a gap.
        reliability: The self-assessed score.
        gated: Whether the gate dropped it.
        gate_reason: Why, when it did.
        bbox: The edge's extfov bounding box.

    Returns:
        The entry.
    """
    return NavFeatureSummary(
        feature_id=f'ring_edge:SATURN:{ring_key}:{edge_label}',
        feature_type=NavFeatureType.RING_EDGE,
        source_model='rings:SATURN',
        reliability=reliability,
        gated=gated,
        gate_reason=gate_reason,
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=1.0 if not gated else 0.21,
            shadow_occluded_fraction=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# One document per image
# ---------------------------------------------------------------------------


def _navigated(
    result: NavResult,
    *,
    image_name: str,
    instrument: str,
    camera: str,
    shutter_mode: str | None,
    image_shape: tuple[int, int],
    start: datetime,
    elapsed_s: float,
) -> dict[str, Any]:
    """Return the document the writer builds for one navigated image.

    Parameters:
        result: The navigation result to curate.
        image_name: Basename of the source image.
        instrument: Registered instrument name of the observation class.
        camera: The camera that took the image.
        shutter_mode: The shutter mode the label recorded, or None for a host
            whose labels carry none.
        image_shape: The loaded image's ``(v, u)`` pixel dimensions.
        start: When this image's run began.
        elapsed_s: How long it took.

    Returns:
        The document, as the writer assembles it.
    """
    return build_metadata_from_result(
        result,
        Path(f'/holdings/{image_name}'),
        image_name,
        instrument=instrument,
        camera=camera,
        shutter_mode=shutter_mode,
        image_shape=image_shape,
        timing=build_timing_section(start, start + timedelta(seconds=elapsed_s)),
    )


def _cassini_star_and_limb() -> dict[str, Any]:
    """The NAC half of the BOTSIM pair: stars and a body limb, both contributing.

    Two techniques agree closely, which is what the cross-technique agreement
    and confidence-calibration sections measure, and the fused offset is ten
    times its WAC partner's, which is what a consistent BOTSIM pair looks like.

    Returns:
        The document.
    """
    stars = [
        _star(100000, reliability=0.91, snr_score=0.91, bbox=(142, 611, 153, 622)),
        _star(100001, reliability=0.84, snr_score=0.84, bbox=(268, 344, 279, 355)),
        _star(100002, reliability=0.77, snr_score=0.77, bbox=(401, 802, 412, 813)),
        _star(100003, reliability=0.69, snr_score=0.69, bbox=(556, 178, 567, 189)),
        _star(100004, reliability=0.55, snr_score=0.55, bbox=(704, 923, 715, 934)),
        _star(100005, reliability=0.41, snr_score=0.41, bbox=(881, 466, 892, 477)),
    ]
    inventory = [
        *stars,
        _faint_star(100099, bbox=(933, 705, 944, 716)),
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.62,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(388, 402, 636, 650),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.62, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.814,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(388, 402, 636, 650),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=1.0, incidence_factor=0.93
            ),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='StarFieldFromCatalogNav',
            feature_ids=tuple(star.feature_id for star in stars),
            offset_px=(3.3, -1.45),
            covariance_px2=np.diag([0.1444, 0.1156]),
            confidence=0.93,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(
                n_inliers=6,
                median_residual_px=0.184,
                n_detected_sources=11,
                n_catalog_predicted=7,
                n_triplets_evaluated=35,
                rotation_below_separability_floor=False,
                wide_offset_lock=False,
                wide_offset_false_lock_expectation=0.002,
            ),
        ),
        NavTechniqueResult(
            technique_name='BodyLimbNav',
            feature_ids=('limb_arc:IAPETUS',),
            offset_px=(3.1, -1.62),
            covariance_px2=np.diag([0.2916, 0.2401]),
            confidence=0.84,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(
                visible_limb_arc_fraction=0.986,
                visible_arc_px=277.0,
                dt_fit_rms_px=0.278,
                lm_iterations=12,
                tukey_inlier_count=269,
                lm_converged=True,
                polarity_rejection_fraction=0.0,
                coarse_peak_fraction=0.625,
            ),
        ),
    ]
    covariance = np.diag([0.0961, 0.0784])
    result = NavResult.success(
        offset_px=(3.25, -1.5),
        covariance_px2=covariance,
        confidence=0.91,
        confidence_rank=derive_confidence_rank(confidence=0.91, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=0.42, gradient_score=1.732),
        provenance=_provenance(
            image_et=170000000.0,
            kernels=_COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        consensus_techniques=['StarFieldFromCatalogNav', 'BodyLimbNav'],
    )
    result = _with_pointing(
        result,
        camera='NAC',
        midtime_et=170000000.0,
        sclk=('1/1294561202.077', '1/1294561202.089', '1/1294561202.102'),
        corrected=_rotation(41.208, -12.664, 87.311),
        original=_rotation(41.204, -12.661, 87.309),
    )
    return _navigated(
        result,
        image_name='N1294561202_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='BOTSIM',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 25, 933806, tzinfo=UTC),
        elapsed_s=12.5,
    )


def _cassini_all_features_gated() -> dict[str, Any]:
    """A Cassini failure whose two body features both fell below the gate.

    The reason and the inventory agree: every feature gated is what
    ``all_features_gated`` means, and it leaves the scene classified by content
    as a single body all the same.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.21,
            gated=True,
            gate_reason='reliability_0.210_below_threshold_0.300',
            bbox_extfov_vu=(470, 486, 552, 568),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.21, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.24,
            gated=True,
            gate_reason='reliability_0.240_below_threshold_0.300',
            bbox_extfov_vu=(470, 486, 552, 568),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=0.24, incidence_factor=0.31
            ),
        ),
    ]
    result = NavResult.failed(
        status_reason=NavStatusReason.ALL_FEATURES_GATED,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=0.18, gradient_score=2.104),
        provenance=_provenance(
            image_et=170000800.0,
            kernels=_COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        feature_inventory=inventory,
    )
    result = _with_pointing(
        result,
        camera='NAC',
        midtime_et=170000800.0,
        sclk=('1/1294562000.055', '1/1294562000.062', '1/1294562000.070'),
        corrected=None,
        original=_rotation(43.917, -11.902, 87.664),
    )
    return _navigated(
        result,
        image_name='N1294562000_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 38, 532110, tzinfo=UTC),
        elapsed_s=8.25,
    )


_LOAD_ERROR_IMAGE_NAME = 'N1294563000_1_CALIB.IMG'
"""Basename of the image whose load fails."""

_LOAD_ERROR_STUB = f'{_COISS_SUBTREE}/N1294563000_1_CALIB'
"""Where the failed image's document sits under the results root."""

_LOAD_ERROR_MESSAGE = (
    'SPICE(SPKINSUFFDATA) -- Insufficient ephemeris data has been loaded to compute '
    'the state of body -82 relative to body 699 at the ephemeris epoch '
    '2005 MAY 22 02:26:36.000.'
)
"""What SPICE says when the kernels cannot place the spacecraft.

The driver classifies a load failure by the hints in this text, so a document
whose ``status_error`` reads ``missing_spice_data`` has to carry a message that
says so.
"""


@contextlib.contextmanager
def _load_always_failing(message: str) -> Iterator[None]:
    """Make the Cassini host's loader raise, and put it back afterwards.

    The driver decides the instrument by exact identity against the registry,
    so the failing loader has to be the registered class's own rather than a
    stand-in subclass.

    Parameters:
        message: What the loader raises.

    Yields:
        Nothing; the loader is replaced for the body of the block.
    """

    def raising(cls: type, /, path: Any, **kwargs: Any) -> Any:
        """Fail the load the way an image with no kernel coverage fails.

        Parameters:
            cls: The observation class; unread.
            path: The image URL the driver resolved; unread.
            kwargs: Further loader options; unread.

        Raises:
            RuntimeError: always, carrying the SPICE coverage text.
        """
        raise RuntimeError(message)

    original = ObsCassiniISS.__dict__.get('from_file')
    setattr(ObsCassiniISS, 'from_file', classmethod(raising))  # noqa: B010
    try:
        yield
    finally:
        if original is None:
            delattr(ObsCassiniISS, 'from_file')
        else:
            setattr(ObsCassiniISS, 'from_file', original)  # noqa: B010


def _cassini_load_error() -> dict[str, Any]:
    """The document the driver returns for an image with no kernel coverage.

    Built by the driver rather than by the metadata writer, because that is
    what writes this shape: no navigation result at all, no image shape, no
    epoch, and a ``status_error`` in place of a ``status_reason``.  The camera
    survives, since the dataset index named it without opening the image.

    Returns:
        The document, with its wall-clock timing replaced by a fixed one.
    """
    image_path = Path('/holdings') / _LOAD_ERROR_IMAGE_NAME
    with TemporaryDirectory() as scratch, _load_always_failing(_LOAD_ERROR_MESSAGE):
        entry = ImageFile(
            image_file_url=FCPath(image_path.as_posix()),
            label_file_url=FCPath(image_path.with_suffix('.LBL').as_posix()),
            results_path_stub=_LOAD_ERROR_STUB,
            camera='NAC',
            # The local path the driver would have downloaded to.  Supplied
            # rather than fetched: the load is the step under test and it
            # fails before any byte of the file is read.
            _image_file_path=image_path,
        )
        _success, document = navigate_image_files(
            ObsCassiniISS,
            ImageFiles(image_files=[entry]),
            FCPath(Path(scratch) / 'results'),
            write_output_files=False,
        )
    start = datetime(2026, 8, 8, 16, 46, 47, 104881, tzinfo=UTC)
    # The driver stamps the moments it ran at, which a stored document cannot
    # hold.  Rebuilt through the writer's own section builder from fixed ones.
    document['timing'] = build_timing_section(start, start + timedelta(seconds=1.5))
    return document


def _cassini_suspect_offset() -> dict[str, Any]:
    """A Cassini success at the search limit, with an outlier and a spurious result.

    Three techniques reported.  The disc correlation is the consensus; the blob
    centroid is viable and sits far enough away to be rejected as an outlier,
    which is what ``excluded_from_consensus`` records; the single-star match
    self-flagged spurious and was dropped before consensus selection, which is
    why it is not in that list.  The fused offset reaches the configured search
    margin for this image size, so the frame is reported as suspect.

    Returns:
        The document.
    """
    stars = [
        _star(200000, reliability=0.58, snr_score=0.58, bbox=(96, 233, 107, 244)),
        _star(200001, reliability=0.52, snr_score=0.52, bbox=(214, 655, 225, 666)),
        _star(200002, reliability=0.47, snr_score=0.47, bbox=(377, 91, 388, 102)),
        _star(200003, reliability=0.39, snr_score=0.39, bbox=(512, 744, 523, 755)),
        _star(200004, reliability=0.31, snr_score=0.31, bbox=(690, 318, 701, 329)),
        _star(200005, reliability=0.24, snr_score=0.24, bbox=(842, 587, 853, 598)),
    ]
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.55,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.55, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='body_blob:IAPETUS',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model='body:IAPETUS',
            reliability=0.38,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(blob_snr=0.61, blob_extent_px=108.0),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.68,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=0.74, incidence_factor=0.88
            ),
        ),
        *stars,
        _faint_star(200099, bbox=(908, 122, 919, 133)),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='BodyDiscCorrelateNav',
            feature_ids=('body_disc:IAPETUS',),
            offset_px=(59.5, -12.0),
            covariance_px2=np.diag([6.8142, 5.2015]),
            confidence=0.42,
            spurious=False,
            at_edge=False,
            diagnostics=BodyDiscDiagnostics(
                ncc_peak=0.412,
                peak_to_runner_up_ratio=1.18,
                consistency_px=2.85,
                consistency_ratio=0.94,
                used_gradient=True,
                body_count=1,
            ),
        ),
        NavTechniqueResult(
            technique_name='BodyBlobNav',
            feature_ids=('body_blob:IAPETUS',),
            offset_px=(34.5, -3.2),
            covariance_px2=np.diag([182.25, 156.25]),
            confidence=0.24,
            spurious=False,
            at_edge=False,
            diagnostics=BodyBlobDiagnostics(
                body_snr_inside_predicted_bbox=41.9,
                body_extent_px=108.037,
                blob_count=1,
                residual_px=26.503,
                max_phase_angle_deg=78.412,
                max_phase_irregularity_factor=0.114,
            ),
        ),
        NavTechniqueResult(
            technique_name='StarUniqueMatchNav',
            feature_ids=('star:UCAC4:200001',),
            offset_px=(-8.0, 44.0),
            covariance_px2=np.diag([0.64, 0.64]),
            confidence=0.12,
            spurious=True,
            at_edge=True,
            diagnostics=StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=6.4,
                brightness_margin_mag=0.35,
                residual_px=3.9,
                detection_peak_ratio=1.05,
            ),
        ),
    ]
    result = NavResult.success(
        offset_px=(59.5, -12.0),
        covariance_px2=np.diag([6.8142, 5.2015]),
        confidence=0.42,
        confidence_rank=derive_confidence_rank(confidence=0.42, sigma_px=(2.6104, 2.2807)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=1.06, gradient_score=3.298),
        provenance=_provenance(
            image_et=170002800.0,
            kernels=_COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        excluded_from_consensus=['BodyBlobNav'],
        consensus_techniques=['BodyDiscCorrelateNav'],
    )
    result = _with_pointing(
        result,
        camera='NAC',
        midtime_et=170002800.0,
        sclk=('1/1294564000.011', '1/1294564000.030', '1/1294564000.049'),
        corrected=_rotation(46.882, -10.117, 88.204),
        original=_rotation(46.861, -10.104, 88.196),
    )
    return _navigated(
        result,
        image_name='N1294564000_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 48, 921574, tzinfo=UTC),
        elapsed_s=31.75,
    )


def _cassini_ring_edges() -> dict[str, Any]:
    """The WAC half of the BOTSIM pair: one Encke gap edge navigated the frame.

    It shares its partner's spacecraft clock and epoch, since a BOTSIM pair is
    one shutter, and its offset is a tenth of the NAC image's, since one WAC
    pixel is ten NAC pixels.

    Returns:
        The document.
    """
    inventory = [
        _ring_edge(
            'encke_gap',
            'IEG',
            reliability=0.72,
            gated=False,
            gate_reason=None,
            bbox=(118, 40, 402, 472),
        ),
        _ring_edge(
            'encke_gap',
            'OEG',
            reliability=0.18,
            gated=True,
            gate_reason='reliability_0.180_below_threshold_0.300',
            bbox=(120, 41, 405, 474),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='RingEdgeNav',
            feature_ids=('ring_edge:SATURN:encke_gap:IEG',),
            offset_px=(0.36, -0.11),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.74,
            spurious=False,
            at_edge=False,
            diagnostics=RingEdgeDiagnostics(
                total_edge_length_px=431.0,
                per_edge_dt_rms_summed=0.216,
                per_edge_dt_rms_mean=0.216,
                per_edge_dt_median_max=0.184,
                edge_count=1,
                is_rank_1=False,
                lm_converged=True,
                coarse_peak_fraction=0.812,
                sigma_orbit_radial_px=0.031,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(0.35, -0.12),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.74,
        confidence_rank=derive_confidence_rank(confidence=0.74, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=0.88, gradient_score=1.417),
        provenance=_provenance(
            image_et=170000000.0,
            kernels=_COISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
        consensus_techniques=['RingEdgeNav'],
    )
    result = _with_pointing(
        result,
        camera='WAC',
        midtime_et=170000000.0,
        sclk=('1/1294561202.077', '1/1294561202.089', '1/1294561202.102'),
        corrected=_rotation(41.211, -12.669, 87.315),
        original=_rotation(41.204, -12.661, 87.309),
    )
    return _navigated(
        result,
        image_name='W1294561202_1_CALIB.IMG',
        instrument='coiss',
        camera='WAC',
        shutter_mode='BOTSIM',
        image_shape=(512, 512),
        start=datetime(2026, 8, 8, 16, 47, 21, 8443, tzinfo=UTC),
        elapsed_s=12.5,
    )


def _voyager_ring_edges() -> dict[str, Any]:
    """A Voyager success on a Huygens gap edge, at an image size with no search limit.

    Its size has no configured extfov margin, which is what makes the suspect
    offset section report that a limit could not be resolved for it.

    Returns:
        The document.
    """
    inventory = [
        _ring_edge(
            'huygens_gap',
            'IEG',
            reliability=0.64,
            gated=False,
            gate_reason=None,
            bbox=(212, 96, 588, 704),
        ),
        _ring_edge(
            'huygens_gap',
            'OEG',
            reliability=0.19,
            gated=True,
            gate_reason='reliability_0.190_below_threshold_0.300',
            bbox=(215, 98, 592, 708),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='RingEdgeNav',
            feature_ids=('ring_edge:SATURN:huygens_gap:IEG',),
            offset_px=(-2.7, 4.55),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.66,
            spurious=False,
            at_edge=False,
            diagnostics=RingEdgeDiagnostics(
                total_edge_length_px=612.0,
                per_edge_dt_rms_summed=0.341,
                per_edge_dt_rms_mean=0.341,
                per_edge_dt_median_max=0.288,
                edge_count=1,
                is_rank_1=False,
                lm_converged=True,
                coarse_peak_fraction=0.703,
                sigma_orbit_radial_px=0.118,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(-2.75, 4.5),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.66,
        confidence_rank=derive_confidence_rank(confidence=0.66, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=212.0, gradient_score=0.914),
        provenance=_provenance(
            image_et=-660000000.0,
            kernels=_VGISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
        consensus_techniques=['RingEdgeNav'],
    )
    result = _with_pointing(
        result,
        camera='NAC',
        instrument='vgiss',
        midtime_et=-660000000.0,
        sclk=('1/13854:55:001', '1/13854:55:001', '1/13854:55:002'),
        corrected=_rotation(112.447, 3.918, -64.220),
        original=_rotation(112.438, 3.911, -64.213),
    )
    return _navigated(
        result,
        image_name='C1385455_GEOMED.IMG',
        instrument='vgiss',
        camera='NAC',
        shutter_mode=None,
        image_shape=(800, 800),
        start=datetime(2026, 8, 8, 16, 47, 33, 799265, tzinfo=UTC),
        elapsed_s=12.5,
    )


def _voyager_no_features() -> dict[str, Any]:
    """A Voyager failure in which no extractor produced a feature at all.

    An empty inventory is what ``no_features_extracted`` means, and it is the
    scene the failure taxonomy classifies as holding no features.

    Returns:
        The document.
    """
    result = NavResult.failed(
        status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=96.0, gradient_score=0.622),
        provenance=_provenance(
            image_et=-659999000.0,
            kernels=_VGISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
    )
    result = _with_pointing(
        result,
        camera='WAC',
        instrument='vgiss',
        midtime_et=-659999000.0,
        sclk=('1/13854:60:001', '1/13854:60:001', '1/13854:60:002'),
        corrected=None,
        original=_rotation(113.005, 4.212, -63.884),
    )
    return _navigated(
        result,
        image_name='C1385460_GEOMED.IMG',
        instrument='vgiss',
        camera='WAC',
        shutter_mode=None,
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 47, 46, 612907, tzinfo=UTC),
        elapsed_s=8.25,
    )


def _simulated_scene() -> dict[str, Any]:
    """A simulated scene navigated on one body limb.

    The one host with no spacecraft and no furnished camera frame, so it
    correctly records no attitude and no exposure times, no shutter mode, and
    no loaded kernels.  Its results path stub names no subtree.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='limb_arc:MIMAS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:MIMAS',
            reliability=0.88,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(84, 92, 172, 180),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=1.0, incidence_factor=0.97
            ),
        )
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='BodyLimbNav',
            feature_ids=('limb_arc:MIMAS',),
            offset_px=(1.5, 0.5),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.88,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(
                visible_limb_arc_fraction=1.0,
                visible_arc_px=188.0,
                dt_fit_rms_px=0.121,
                lm_iterations=7,
                tukey_inlier_count=186,
                lm_converged=True,
                polarity_rejection_fraction=0.0,
                coarse_peak_fraction=0.914,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(1.5, 0.5),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.88,
        confidence_rank=derive_confidence_rank(confidence=0.88, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=_classifier(noise_sigma=0.75, max_dn=0.64, gradient_score=None),
        provenance=_provenance(image_et=100.0, kernels=_SIM_KERNELS, extractors=('body:MIMAS',)),
        consensus_techniques=['BodyLimbNav'],
    )
    return _navigated(
        result,
        image_name='sim_scene_000042.img',
        instrument='sim',
        camera='SIM',
        shutter_mode=None,
        image_shape=(256, 256),
        start=datetime(2026, 8, 8, 16, 47, 55, 180332, tzinfo=UTC),
        elapsed_s=12.5,
    )


_CASSINI_CAMERA_FRAME_IDS = {'NAC': -82360, 'WAC': -82361}
"""SPICE frame id of each Cassini ISS camera frame."""

_VOYAGER_CAMERA_FRAME_IDS = {'NAC': -31101, 'WAC': -31102}
"""SPICE frame id of each Voyager 1 ISS camera frame."""

_CASSINI_EXPOSURE_S = 0.46
"""Exposure the Cassini images were taken with."""

_VOYAGER_EXPOSURE_S = 1.44
"""Exposure the Voyager images were taken with."""


def _with_pointing(
    result: NavResult,
    *,
    camera: str,
    midtime_et: float,
    sclk: tuple[str, str, str],
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
    instrument: str = 'coiss',
) -> NavResult:
    """Stamp an attitude solution onto a result, as the orchestrator does.

    Parameters:
        result: The result to stamp.
        camera: The camera that took the image, which names its frame.
        midtime_et: Exposure midtime, which is also the image's epoch.
        sclk: Spacecraft clock strings at start, midtime and stop.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result with no offset.
        instrument: Which host's frames to use.

    Returns:
        The same result, carrying the solution.
    """
    if instrument == 'coiss':
        solution = _pointing(
            camera_frame=f'CASSINI_ISS_{camera}',
            camera_frame_id=_CASSINI_CAMERA_FRAME_IDS[camera],
            ck_frame_id=-82000,
            oops_from_spice=_CASSINI_OOPS_FROM_SPICE,
            midtime_et=midtime_et,
            exposure_s=_CASSINI_EXPOSURE_S,
            sclk=sclk,
            original=original,
            corrected=corrected,
        )
    else:
        solution = _pointing(
            camera_frame=f'VG1_ISS{camera[0]}A',
            camera_frame_id=_VOYAGER_CAMERA_FRAME_IDS[camera],
            ck_frame_id=-31100,
            oops_from_spice=_VOYAGER_OOPS_FROM_SPICE,
            midtime_et=midtime_et,
            exposure_s=_VOYAGER_EXPOSURE_S,
            sclk=sclk,
            original=original,
            corrected=corrected,
        )
    return dataclasses.replace(result, pointing=solution)


def results_tree_documents() -> dict[str, dict[str, Any]]:
    """Return every document of the fixture tree, keyed by its results path stub.

    Returns:
        Stub to document, in the order the tree is written.
    """
    return {
        f'{_COISS_SUBTREE}/N1294561202_1_CALIB': _cassini_star_and_limb(),
        f'{_COISS_SUBTREE}/N1294562000_1_CALIB': _cassini_all_features_gated(),
        _LOAD_ERROR_STUB: _cassini_load_error(),
        f'{_COISS_SUBTREE}/N1294564000_1_CALIB': _cassini_suspect_offset(),
        f'{_COISS_SUBTREE}/W1294561202_1_CALIB': _cassini_ring_edges(),
        f'{_VGISS_SUBTREE}/C1385455_GEOMED': _voyager_ring_edges(),
        f'{_VGISS_SUBTREE}/C1385460_GEOMED': _voyager_no_features(),
        'sim_scene_000042': _simulated_scene(),
    }


def write_results_tree(root: Path) -> list[Path]:
    """Write every document into a results root, as the writer serializes them.

    Parameters:
        root: The results root to write under.

    Returns:
        The paths written, in tree order.
    """
    written: list[Path] = []
    for stub, document in results_tree_documents().items():
        path = root / f'{stub}_metadata.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_as_string(document), encoding='utf-8')
        written.append(path)
    return written


def stored_documents() -> dict[str, dict[str, Any]]:
    """Return every document the stored tree holds, keyed by its stub.

    Returns:
        Stub to parsed document.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_TREE.rglob('*_metadata.json')):
        stub = path.relative_to(RESULTS_TREE).as_posix().removesuffix('_metadata.json')
        parsed = json.loads(path.read_text(encoding='utf-8'))
        assert isinstance(parsed, dict)
        found[stub] = parsed
    return found


if __name__ == '__main__':
    for written_path in write_results_tree(RESULTS_TREE):
        print(written_path)
