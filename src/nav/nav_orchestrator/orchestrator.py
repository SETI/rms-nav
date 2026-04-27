"""NavOrchestrator — top-level driver for autonomous navigation.

The orchestrator turns one observation into one ``NavResult``:

1. Build a ``NavContext`` (image, masks, classifier verdict, provenance).
2. Iterate registered ``NavModel`` instances and gather features and
   annotations from each.
3. Apply the ``FeatureReliabilityGate`` to drop bad-data features.
4. Run every feasible prior-free technique on the surviving features.
5. Combine pass-1 results via the ``ensemble`` function to derive a prior.
6. Run prior-required techniques against the derived prior.
7. Combine the union of pass-1 and pass-2 results via ``ensemble``.

Glob-pattern filters at construction time let an operator restrict which
models or techniques run for debugging (``only_models='body:MIMAS'``,
``only_techniques='!StarFieldFromCatalogNav'``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from nav.annotation import Annotations
from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.reliability import FeatureReliabilityGate, GatedFeatureRecord
from nav.nav_model import NavModel
from nav.nav_orchestrator.ensemble import EnsembleConfig, ensemble
from nav.nav_orchestrator.feature_summary import NavFeatureSummary
from nav.nav_orchestrator.image_classifier import (
    ImageQualityThresholds,
    NavImageClassifier,
)
from nav.nav_orchestrator.image_classifier_result import (
    ImageClass,
    NavImageClassifierResult,
)
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_orchestrator.nav_result import NavResult
from nav.nav_orchestrator.provenance import Provenance
from nav.nav_technique.nav_technique import NavTechnique, filter_technique_names
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.image_quality import cosmic_ray_mask, saturation_mask
from nav.support.nav_base import NavBase
from nav.support.noise_estimate import estimate_image_noise_sigma
from nav.support.status_reason import NavStatusReason

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.obs import ObsSnapshotInst

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'NavOrchestrator',
]


_HARD_FAILURE_TO_REASON: dict[ImageClass, NavStatusReason] = {
    'blank': NavStatusReason.NO_SIGNAL_IN_IMAGE,
    'fully_overexposed': NavStatusReason.IMAGE_OVEREXPOSED,
    'mostly_missing_data': NavStatusReason.MISSING_DATA_DOMINANT,
    'corrupt': NavStatusReason.IMAGE_CORRUPT,
}
"""Image-classifier classes that short-circuit before any technique runs.

Maps each hard-failure ``ImageClass`` to the matching ``NavStatusReason``
returned by the orchestrator's preflight.  Reading from this dict is the
sole admission test for the hard-failure short-circuit.
"""


DEFAULT_FULL_WELL_DN_12_BIT: float = 4095.0
"""Default saturation DN for a 12-bit camera readout.

Used by ``_instrument_full_well_dn`` when no per-instrument override is
configured.  The value is a generous upper bound: any pixel at the value
is fully saturated on every supported 12-bit instrument (Cassini ISS,
Voyager ISS, Galileo SSI, New Horizons LORRI).
"""


@dataclass
class _ModelRegistry:
    """Registry of NavModel instances built per-image by the orchestrator.

    Concrete NavModel subclasses do not auto-register at import time because
    they require an observation; the orchestrator instantiates them per
    image and registers them here.

    Parameters:
        models: List of constructed NavModel instances for the current image.
    """

    models: list[NavModel] = field(default_factory=list)

    def filter_by_glob(self, patterns: str | list[str]) -> list[NavModel]:
        """Return models whose ``name`` matches the glob pattern list."""
        names = [m.name for m in self.models]
        kept = set(filter_technique_names(names, patterns))
        return [m for m in self.models if m.name in kept]


class NavOrchestrator(NavBase):
    """Top-level driver for autonomous navigation.

    Parameters:
        models: List of constructed NavModel instances for one observation.
            Caller builds these per-image (since each NavModel binds to an
            ``obs``).
        config: Optional ``Config`` override.
        only_models: Glob-pattern string or list selecting which models run.
            Default ``'*'`` runs every supplied model.
        only_techniques: Glob-pattern string or list selecting which
            techniques run.  Default ``'*'`` runs every registered
            technique.
        ensemble_config: Optional ``EnsembleConfig`` override.
        image_quality_thresholds: Optional thresholds for the image-quality
            classifier.
        rms_nav_version: Version string written into provenance.
    """

    def __init__(
        self,
        models: list[NavModel],
        *,
        config: Config | None = None,
        only_models: str | list[str] = '*',
        only_techniques: str | list[str] = '*',
        ensemble_config: EnsembleConfig | None = None,
        image_quality_thresholds: ImageQualityThresholds | None = None,
        rms_nav_version: str = '0.0.0',
    ) -> None:
        super().__init__(config=config)
        self._registry = _ModelRegistry(
            models=_ModelRegistry(models=models).filter_by_glob(only_models)
        )
        self._only_techniques = only_techniques
        self._ensemble_config = ensemble_config or EnsembleConfig()
        self._image_classifier = NavImageClassifier(
            thresholds=image_quality_thresholds or ImageQualityThresholds()
        )
        self._gate = FeatureReliabilityGate()
        self._rms_nav_version = rms_nav_version

    def navigate(self, obs: ObsSnapshotInst) -> NavResult:
        """Run the full pipeline on one observation.

        Parameters:
            obs: The observation snapshot to navigate.

        Returns:
            A single ``NavResult`` summarizing the navigation outcome.
        """
        provenance = self._make_provenance(obs)
        context, image_classifier = self._make_context(obs, provenance)
        if image_classifier.image_class in _HARD_FAILURE_TO_REASON:
            return NavResult.failed(
                status_reason=_HARD_FAILURE_TO_REASON[image_classifier.image_class],
                image_classifier=image_classifier,
                provenance=provenance,
            )
        for model in self._registry.models:
            model.create_model()
        all_features = self._extract_features(context)
        kept, gated = self._gate.apply(all_features)
        feature_inventory = self._build_inventory(kept, gated)
        model_metadata = self._collect_model_metadata()
        annotations = self._collect_annotations(context)
        if not all_features:
            return NavResult.failed(
                status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        if not kept:
            return NavResult.failed(
                status_reason=NavStatusReason.ALL_FEATURES_GATED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        # Pass 1 — prior-free techniques.
        pass1_results = self._run_pass(kept, context, requires_prior=False)
        if not pass1_results:
            return NavResult.failed(
                status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        pass1_ensemble = ensemble(
            pass1_results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            config=self._ensemble_config,
            model_metadata=model_metadata,
            annotations=annotations,
        )
        if pass1_ensemble.status == 'failed':
            return pass1_ensemble
        # Pass 2 — prior-required techniques refine on the pass-1 prior.
        if pass1_ensemble.offset_px is not None and pass1_ensemble.covariance_px2 is not None:
            pass2_context = context.with_prior(
                offset_px=pass1_ensemble.offset_px,
                covariance_px2=pass1_ensemble.covariance_px2,
            )
        else:
            pass2_context = context
        pass2_results = self._run_pass(kept, pass2_context, requires_prior=True)
        # Final ensemble over the union of both passes' results.
        return ensemble(
            pass1_results + pass2_results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            config=self._ensemble_config,
            model_metadata=model_metadata,
            annotations=annotations,
        )

    def _collect_model_metadata(self) -> dict[str, dict[str, Any]]:
        """Snapshot ``model.metadata`` from every registered NavModel."""
        out: dict[str, dict[str, Any]] = {}
        for model in self._registry.models:
            out[model.name] = dict(model.metadata)
        return out

    def _collect_annotations(self, context: NavContext) -> Annotations:
        """Merge per-NavModel annotation collections into one.

        Each model's ``to_annotations`` is invoked; failures are logged and
        treated as if the model emitted an empty collection so a misbehaving
        model never blocks the rest of the pipeline.
        """
        merged = Annotations()
        for model in self._registry.models:
            try:
                model_annotations = model.to_annotations(context)
            except Exception:  # plugin sandbox; mirrors _extract_features
                self._logger.exception(
                    'NavModel %s.to_annotations raised; skipping its annotations',
                    model.name,
                )
                continue
            merged.add_annotations(model_annotations)
        return merged

    def _extract_features(self, context: NavContext) -> list[NavFeature]:
        """Iterate registered models and gather their features.

        A misbehaving NavModel is logged with a full traceback and treated
        as if it emitted zero features.  Catching every exception is
        intentional: the orchestrator must never raise through to its
        caller — failures surface on the returned ``NavResult`` instead.
        Specific exceptions cannot be enumerated because every NavModel
        plugin has its own failure modes.
        """
        all_features: list[NavFeature] = []
        for model in self._registry.models:
            try:
                emitted = model.to_features(context)
            except Exception:  # plugin sandbox; see docstring
                self._logger.exception(
                    'NavModel %s.to_features raised; treating as no features',
                    model.name,
                )
                emitted = []
            all_features.extend(emitted)
        return all_features

    def _run_pass(
        self,
        features: list[NavFeature],
        context: NavContext,
        *,
        requires_prior: bool,
    ) -> list[NavTechniqueResult]:
        """Run every feasible technique whose ``requires_prior`` matches.

        A misbehaving NavTechnique is logged with a full traceback and
        treated as if it produced no result.  Catching every exception is
        intentional for the same reason as ``_extract_features``: the
        orchestrator never raises through to its caller, failures land on
        the returned ``NavResult``.
        """
        results: list[NavTechniqueResult] = []
        names = [cls.name for cls in NavTechnique._registry if cls.requires_prior == requires_prior]
        kept_names = set(filter_technique_names(names, self._only_techniques))
        for cls in NavTechnique._registry:
            if cls.requires_prior != requires_prior:
                continue
            if cls.name not in kept_names:
                continue
            available_types: set[NavFeatureType] = {f.feature_type for f in features}
            if not (cls.accepts_feature_types & available_types):
                continue
            technique = cls(config=self.config)
            feasibility = technique.is_feasible(features)
            if not feasibility.feasible:
                continue
            subset = [f for f in features if f.feature_type in cls.accepts_feature_types]
            try:
                results.append(technique.navigate(subset, context))
            except Exception:  # plugin sandbox; see docstring
                self._logger.exception(
                    'NavTechnique %s.navigate raised; treating as no result',
                    cls.name,
                )
        return results

    def _make_context(
        self, obs: ObsSnapshotInst, provenance: Provenance
    ) -> tuple[NavContext, NavImageClassifierResult]:
        """Build a NavContext from an observation."""
        image = obs.data.astype('float64')
        sensor_mask = obs.extfov_data_sensor_mask()
        classifier_result = self._image_classifier.classify(image, sensor_mask)
        full_well = self._instrument_full_well_dn(obs)
        sat_mask = saturation_mask(image, full_well_dn=full_well)
        # Cosmic-ray detection requires a strictly positive noise sigma;
        # the classifier supplies one already, but a near-zero estimate is
        # clamped to a tiny value so the mask is well-defined even on
        # near-blank inputs.
        cr_noise_sigma = max(classifier_result.noise_sigma, 1e-6)
        cr_mask = cosmic_ray_mask(image, image_noise_sigma=cr_noise_sigma)
        noise_sigma = (
            classifier_result.noise_sigma
            if classifier_result.noise_sigma > 0.0
            else estimate_image_noise_sigma(image, sensor_mask)
        )
        context = NavContext(
            obs=obs,
            image_ext=image,
            sensor_mask_ext=sensor_mask,
            image_noise_sigma=noise_sigma,
            saturation_mask_ext=sat_mask,
            cosmic_ray_mask_ext=cr_mask,
            image_classifier=classifier_result,
            provenance=provenance,
        )
        return context, classifier_result

    def _make_provenance(self, obs: ObsSnapshotInst) -> Provenance:
        """Build the per-image Provenance envelope."""
        timestamp = datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')
        return Provenance(
            rms_nav_version=self._rms_nav_version,
            image_et=float(obs.midtime),
            pipeline_run_iso8601=timestamp,
            technique_names=tuple(sorted(cls.name for cls in NavTechnique._registry)),
            extractor_names=tuple(sorted(m.name for m in self._registry.models)),
        )

    def _instrument_full_well_dn(self, obs: ObsSnapshotInst) -> float:
        """Return the saturation DN used for the per-image saturation mask.

        Returns the 12-bit camera default ``DEFAULT_FULL_WELL_DN_12_BIT``
        for now; per-instrument values will replace this lookup once the
        per-instrument noise YAML blocks ship.  ``obs`` is accepted so the
        per-instrument path can branch on it without a signature change.
        """
        del obs
        return DEFAULT_FULL_WELL_DN_12_BIT

    @staticmethod
    def _build_inventory(
        kept: list[NavFeature], gated: list[GatedFeatureRecord]
    ) -> list[NavFeatureSummary]:
        """Build the feature inventory consumed by the curator."""
        out: list[NavFeatureSummary] = []
        for f in kept:
            out.append(_summary_from_feature(f, gated=False, gate_reason=None))
        for record in gated:
            out.append(_summary_from_feature(record.feature, gated=True, gate_reason=record.reason))
        return out


def _summary_from_feature(
    feature: NavFeature, *, gated: bool, gate_reason: str | None
) -> NavFeatureSummary:
    """Project a NavFeature down to a NavFeatureSummary."""
    bbox = _bbox_from_geometry(feature)
    return NavFeatureSummary(
        feature_id=feature.feature_id,
        feature_type=feature.feature_type,
        source_model=feature.source_model,
        reliability=feature.reliability,
        gated=gated,
        gate_reason=gate_reason,
        bbox_extfov_vu=bbox,
    )


def _bbox_from_geometry(feature: NavFeature) -> tuple[int, int, int, int]:
    """Return the ``bbox_extfov_vu`` tuple for a feature's geometry payload.

    Every ``NavFeatureGeometry`` variant carries ``bbox_extfov_vu``;
    direct attribute access is safe because the union excludes any
    payload that lacks it.
    """
    bbox = feature.geometry.bbox_extfov_vu
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
