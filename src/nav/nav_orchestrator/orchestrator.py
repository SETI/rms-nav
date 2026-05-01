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

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from nav.annotation import Annotations
from nav.config import IMAGE_LOGGER, Config
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
from nav.nav_orchestrator.image_derivatives import (
    ImageDerivativesConfig,
    compute_all_image_derivatives,
)
from nav.nav_orchestrator.instrument_config import (
    InstrumentSettings,
    instrument_settings_from_obs,
)
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_orchestrator.nav_result import NavResult
from nav.nav_orchestrator.provenance import (
    Provenance,
    collect_provenance_metadata,
)
from nav.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE
from nav.nav_technique.nav_technique import NavTechnique, filter_technique_names
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.filters import NavFilterKind, NavFilterSpec, apply_filter
from nav.support.image_quality import cosmic_ray_mask, saturation_mask
from nav.support.nav_base import NavBase
from nav.support.noise_estimate import estimate_image_noise_sigma
from nav.support.status_reason import NavStatusReason

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.obs import ObsSnapshotInst

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
        """Return models whose ``name`` matches the glob pattern list.

        Model names follow the ``prefix:VALUE`` convention
        (``rings:SATURN``, ``body:DIONE``, plain ``stars`` for the
        catalog-driven star model).  This helper normalizes user
        patterns to that convention so the common shorthand forms
        match without requiring explicit globs:

        - ``"rings"`` is expanded to ``"rings:*"`` so a pattern
          missing a colon matches every namespaced model under that
          prefix.  ``"stars"`` (which has no colon in its model name)
          continues to match itself directly.
        - The value part of ``"prefix:VALUE"`` is normalized to
          uppercase so ``"body:saturn"`` matches ``"body:SATURN"``.

        The leading-``!`` exclusion convention is preserved.
        """
        names = [m.name for m in self.models]
        normalized = _normalize_model_patterns(patterns, names)
        kept = set(filter_technique_names(names, normalized))
        return [m for m in self.models if m.name in kept]


def _normalize_model_patterns(patterns: str | list[str], names: list[str]) -> list[str]:
    """Normalize model glob patterns to the ``prefix:VALUE`` convention.

    See :meth:`_ModelRegistry.filter_by_glob` for the rules.

    Parameters:
        patterns: Single pattern or list of patterns from the user.
        names: Available model names; used to detect whether the
            ``prefix:*`` expansion makes sense (any name contains
            ``:``).

    Returns:
        A list of normalized patterns suitable for
        :func:`filter_technique_names`.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    has_namespaced_models = any(':' in n for n in names)
    out: list[str] = []
    for raw in patterns:
        if raw.startswith('!'):
            prefix, body = '!', raw[1:]
        else:
            prefix, body = '', raw
        if ':' in body:
            head, _, tail = body.partition(':')
            normalized = f'{head}:{tail.upper()}'
        elif has_namespaced_models and body and not any(c in body for c in '*?['):
            # Plain prefix-only token (e.g. "rings"): expand to "rings:*"
            # so it matches every namespaced model under that prefix.
            # Names that have no namespace (like "stars") still match
            # via the unmodified pattern below.
            normalized = f'{body}:*'
            out.append(prefix + body)  # also keep the literal for "stars"
        else:
            normalized = body
        out.append(prefix + normalized)
    return out


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
        image_derivatives_config: ImageDerivativesConfig | None = None,
        rms_nav_version: str = '0.0.0',
    ) -> None:
        super().__init__(config=config)
        self._registry = _ModelRegistry(
            models=_ModelRegistry(models=models).filter_by_glob(only_models)
        )
        self._only_techniques = only_techniques
        self._ensemble_config = ensemble_config or EnsembleConfig()
        # ``image_quality_thresholds`` overrides the per-instrument config
        # block when supplied; otherwise the orchestrator builds the
        # thresholds from ``obs.inst_config`` per image.
        self._explicit_thresholds = image_quality_thresholds
        self._image_derivatives_config = image_derivatives_config or ImageDerivativesConfig()
        self._gate = FeatureReliabilityGate()
        self._rms_nav_version = rms_nav_version

    def prepare(
        self,
        obs: ObsSnapshotInst,
        *,
        apply_gate: bool = True,
    ) -> tuple[NavContext, list[NavFeature]]:
        """Build per-image state without running any technique.

        Runs the same pre-technique pipeline as :meth:`navigate`: builds
        the provenance envelope and the :class:`NavContext`, calls every
        registered NavModel's ``create_model``, and extracts features.
        Hard-failure image classes are logged but **not** short-circuited
        — the caller (manual-nav dialog, debugger) decides what to do on
        a blank or saturated frame.

        Parameters:
            obs: The observation snapshot to prepare.
            apply_gate: When ``True`` (the default) the reliability gate
                runs and only the kept features are returned.  When
                ``False`` every emitted feature is returned, gated or
                not — useful for the manual-nav dialog, where the
                operator overrides the autonomous reliability decision.

        Returns:
            ``(context, features)``.  ``features`` is the gated subset
            when ``apply_gate=True``, otherwise the full set.
        """
        provenance = self._make_provenance(obs)
        context, image_classifier = self._make_context(obs, provenance)
        self._log_classifier_verdict(image_classifier)
        built_models = self._build_models()
        all_features = self._extract_features(context, built_models)
        if not apply_gate:
            self._logger.info(
                'Reliability gate skipped (apply_gate=False); returning %d feature(s)',
                len(all_features),
            )
            return context, all_features
        kept, _gated = self._extract_and_gate(all_features)
        return context, kept

    def navigate(self, obs: ObsSnapshotInst) -> NavResult:
        """Run the full pipeline on one observation.

        Parameters:
            obs: The observation snapshot to navigate.

        Returns:
            A single ``NavResult`` summarizing the navigation outcome.
        """
        provenance = self._make_provenance(obs)
        context, image_classifier = self._make_context(obs, provenance)
        self._log_classifier_verdict(image_classifier)
        if image_classifier.image_class in _HARD_FAILURE_TO_REASON:
            return self._fail(
                status_reason=_HARD_FAILURE_TO_REASON[image_classifier.image_class],
                image_classifier=image_classifier,
                provenance=provenance,
            )
        built_models = self._build_models()
        all_features = self._extract_features(context, built_models)
        kept, gated = self._extract_and_gate(all_features)
        if gated:
            gated_kinds: dict[str, int] = {}
            for record in gated:
                key = record.feature.feature_type.name
                gated_kinds[key] = gated_kinds.get(key, 0) + 1
            self._logger.debug('Gated breakdown: %s', gated_kinds)
        feature_inventory = self._build_inventory(kept, gated)
        model_metadata = self._collect_model_metadata(built_models)
        annotations = self._collect_annotations(context, built_models)
        if not all_features:
            return self._fail(
                status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        if not kept:
            return self._fail(
                status_reason=NavStatusReason.ALL_FEATURES_GATED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        # Pass 1 — prior-free techniques.
        self._logger.info('Pass 1: running prior-free techniques')
        pass1_results = self._run_pass(kept, context, requires_prior=False)
        self._logger.info('Pass 1: %d technique result(s) produced', len(pass1_results))
        if not pass1_results:
            return self._fail(
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
            self._log_status_reason(pass1_ensemble.status_reason)
            return pass1_ensemble
        if pass1_ensemble.offset_px is not None:
            self._logger.info(
                'Pass 1 prior: offset (dv, du) = (%.4f, %.4f) px, confidence %.3f',
                pass1_ensemble.offset_px[0],
                pass1_ensemble.offset_px[1],
                pass1_ensemble.confidence,
            )
        # Pass 2 — prior-required techniques refine on the pass-1 prior.
        if pass1_ensemble.offset_px is not None and pass1_ensemble.covariance_px2 is not None:
            pass2_context = context.with_prior(
                offset_px=pass1_ensemble.offset_px,
                covariance_px2=pass1_ensemble.covariance_px2,
            )
        else:
            pass2_context = context
        self._logger.info('Pass 2: running prior-required techniques')
        pass2_results = self._run_pass(kept, pass2_context, requires_prior=True)
        self._logger.info('Pass 2: %d technique result(s) produced', len(pass2_results))
        # Final ensemble over the union of both passes' results.
        final = ensemble(
            pass1_results + pass2_results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            config=self._ensemble_config,
            model_metadata=model_metadata,
            annotations=annotations,
        )
        self._log_final_result(final)
        self._log_status_reason(final.status_reason)
        return final

    def _log_classifier_verdict(self, classifier: NavImageClassifierResult) -> None:
        """Emit the per-image image-quality classifier verdict at INFO."""
        self._logger.info(
            'Image classifier: class=%s, max_dn=%.4g, noise_sigma=%.3g, '
            'saturation_frac=%.3f, missing_frac=%.3f',
            classifier.image_class,
            classifier.max_dn,
            classifier.noise_sigma,
            classifier.saturation_frac,
            classifier.missing_frac,
        )
        if classifier.flags:
            self._logger.info('Image classifier flags: %s', list(classifier.flags))

    def _log_final_result(self, result: NavResult) -> None:
        """Emit the final offset / confidence / per-technique summary."""
        if result.offset_px is None:
            self._logger.info('Final result: no offset (status=%s)', result.status)
            return
        sigma_dv = result.sigma_px[0] if result.sigma_px is not None else float('nan')
        sigma_du = result.sigma_px[1] if result.sigma_px is not None else float('nan')
        self._logger.info(
            'Final offset (dv, du) = (%.4f, %.4f) px; sigma (dv, du) = (%.4f, %.4f) px',
            result.offset_px[0],
            result.offset_px[1],
            sigma_dv,
            sigma_du,
        )
        self._logger.info(
            'Final status = %s, confidence = %.3f (rank=%s); %d technique result(s) fused',
            result.status,
            result.confidence,
            result.confidence_rank,
            len(result.per_technique),
        )
        if result.per_technique:
            for tr in result.per_technique:
                self._logger.debug(
                    'Per-technique: %s -> offset (%.4f, %.4f) px, confidence %.3f, '
                    'spurious=%s, at_edge=%s',
                    tr.technique_name,
                    tr.offset_px[0],
                    tr.offset_px[1],
                    tr.confidence,
                    tr.spurious,
                    tr.at_edge,
                )

    def _build_models(self) -> list[NavModel]:
        """Call ``create_model`` on every registered NavModel.

        Wrapped so :meth:`prepare` and :meth:`navigate` share the same
        log line and exception-sandbox behavior.  Models whose
        ``create_model`` raises are dropped from the returned list so
        downstream feature / annotation / metadata collection skips
        them entirely (rather than processing partially-built state).
        """
        self._logger.info(
            'Building %d NavModel(s): %s',
            len(self._registry.models),
            ', '.join(m.name for m in self._registry.models) or '(none)',
        )
        built_models: list[NavModel] = []
        for model in self._registry.models:
            try:
                model.create_model()
            except Exception:  # plugin sandbox; mirrors _extract_features
                self._logger.exception(
                    'NavModel %s.create_model raised; skipping its features and annotations',
                    model.name,
                )
                continue
            built_models.append(model)
        return built_models

    def _extract_and_gate(
        self, all_features: list[NavFeature]
    ) -> tuple[list[NavFeature], list[GatedFeatureRecord]]:
        """Apply the reliability gate and emit the standard summary log line."""
        kept, gated = self._gate.apply(all_features)
        self._logger.info(
            'Reliability gate: %d feature(s) kept, %d gated (out of %d emitted)',
            len(kept),
            len(gated),
            len(all_features),
        )
        return kept, gated

    def _fail(
        self,
        *,
        status_reason: NavStatusReason,
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
        feature_inventory: list[NavFeatureSummary] | None = None,
        model_metadata: dict[str, dict[str, Any]] | None = None,
        annotations: Annotations | None = None,
    ) -> NavResult:
        """Emit the operator-readable INFO lines and return a failed NavResult.

        Wraps ``NavResult.failed`` so every short-circuit and gate emits
        the per-status-reason INFO log lines defined in
        :data:`STATUS_REASON_INFO_TEMPLATE`.
        """
        self._log_status_reason(status_reason)
        return NavResult.failed(
            status_reason=status_reason,
            image_classifier=image_classifier,
            provenance=provenance,
            feature_inventory=feature_inventory or [],
            model_metadata=model_metadata or {},
            annotations=annotations or Annotations(),
        )

    def _log_status_reason(self, status_reason: NavStatusReason) -> None:
        """Emit the per-status-reason INFO log lines."""
        for line in STATUS_REASON_INFO_TEMPLATE.get(status_reason, ()):
            self._logger.info(line)

    def _collect_model_metadata(self, models: list[NavModel]) -> dict[str, dict[str, Any]]:
        """Snapshot ``model.metadata`` from every successfully-built NavModel."""
        out: dict[str, dict[str, Any]] = {}
        for model in models:
            out[model.name] = dict(model.metadata)
        return out

    def _collect_annotations(self, context: NavContext, models: list[NavModel]) -> Annotations:
        """Merge per-NavModel annotation collections into one.

        Each model's ``to_annotations`` is invoked; failures are logged and
        treated as if the model emitted an empty collection so a misbehaving
        model never blocks the rest of the pipeline.
        """
        merged = Annotations()
        for model in models:
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

    def _extract_features(self, context: NavContext, models: list[NavModel]) -> list[NavFeature]:
        """Iterate built models and gather their features.

        A misbehaving NavModel is logged with a full traceback and treated
        as if it emitted zero features.  Catching every exception is
        intentional: the orchestrator must never raise through to its
        caller — failures surface on the returned ``NavResult`` instead.
        Specific exceptions cannot be enumerated because every NavModel
        plugin has its own failure modes.
        """
        all_features: list[NavFeature] = []
        for model in models:
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
        """Build a NavContext from an observation.

        The image, sensor mask, saturation mask, and cosmic-ray mask all
        live on the *extended FOV* (zero-padded around the original sensor
        rectangle).  ``obs.extdata`` is the canonical source for the
        extfov-shaped image and matches the extfov sensor mask shape
        regardless of the per-instrument ``extfov_margin_vu``.
        """
        settings = instrument_settings_from_obs(obs)
        raw_image = obs.extdata.astype('float64')
        sensor_mask = obs.extfov_data_sensor_mask()
        image, pre_filter = self._apply_source_image_filter(raw_image, obs)
        # The classifier reads the image *after* the source-image filter
        # so blank/saturation/missing fractions match what downstream
        # extractors will see.
        classifier_thresholds = (
            self._explicit_thresholds
            if self._explicit_thresholds is not None
            else settings.thresholds
        )
        classifier = NavImageClassifier(thresholds=classifier_thresholds)
        classifier_result = classifier.classify(image, sensor_mask)
        sat_mask = self._build_saturation_mask(image, sensor_mask, settings)
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
        # Single-pass derivative computation: one gaussian + sobel pair
        # produces gradient magnitude, edge DT, and the signed gradient
        # vector image together rather than doing the heavy smoothing
        # twice for separate calls.
        gradient_ext, edge_dt_ext, gradient_vu_ext = compute_all_image_derivatives(
            image,
            noise_sigma,
            config=self._image_derivatives_config,
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
            image_gradient_ext=gradient_ext,
            image_gradient_vu_ext=gradient_vu_ext,
            image_edge_dt_ext=edge_dt_ext,
            pre_filter_applied=pre_filter,
            fit_camera_rotation=settings.fit_camera_rotation,
            max_rotation_deg=settings.max_rotation_deg,
        )
        return context, classifier_result

    @staticmethod
    def _build_saturation_mask(
        image: np.ndarray,
        sensor_mask: np.ndarray,
        settings: InstrumentSettings,
    ) -> np.ndarray:
        """Construct the per-image saturation mask.

        For raw-DN instruments the mask is ``image >= saturation_dn``.
        For calibrated-IF instruments the saturation_dn is unavailable
        and the orchestrator emits an empty mask plus a one-line WARNING.
        """
        if settings.saturation_dn is not None:
            return saturation_mask(image, full_well_dn=settings.saturation_dn)
        if settings.data_units == 'calibrated_if':
            IMAGE_LOGGER.warning(
                'no saturation_dn configured for calibrated_if instrument; '
                'overexposure cannot be detected — saturation mask left empty'
            )
        return np.zeros(image.shape, dtype=bool)

    def _apply_source_image_filter(
        self, image: np.ndarray, obs: ObsSnapshotInst
    ) -> tuple[np.ndarray, NavFilterSpec | None]:
        """Apply the per-instrument source-image filter (when enabled).

        Returns the filtered image and the ``NavFilterSpec`` that produced
        it (so :class:`NavContext.pre_filter_applied` records what ran).
        When the per-instrument ``source_image_filter`` block is missing,
        disabled, or set to ``NONE`` the image is returned unchanged and
        the spec is ``None``.
        """
        inst_config = getattr(obs, 'inst_config', None)
        if inst_config is None:
            return image, None
        block = inst_config.get('source_image_filter')
        if not block or not bool(block.get('enabled', False)):
            return image, None
        kind_str = str(block.get('kind', 'NONE')).upper()
        try:
            kind = NavFilterKind[kind_str]
        except KeyError:
            self._logger.warning(
                'unknown source_image_filter.kind %r; skipping pre-filter', kind_str
            )
            return image, None
        if kind is NavFilterKind.NONE:
            return image, None
        if kind is NavFilterKind.BANDPASS_DOG:
            lo = float(block.get('lo_sigma_px', 0.0))
            hi = float(block.get('hi_sigma_px', 0.0))
            spec = NavFilterSpec(kind=kind, bandpass_cutoffs_px=(lo, hi))
        else:
            sigma = float(block.get('sigma_px', 1.0))
            spec = NavFilterSpec(kind=kind, sigma_xy=(sigma, sigma))
        filtered = apply_filter(image, spec)
        return filtered, spec

    def _make_provenance(self, obs: ObsSnapshotInst) -> Provenance:
        """Build the per-image Provenance envelope.

        Reads the runtime-derived fields (git SHA, loaded SPICE kernels,
        static-data hashes) once per ``navigate`` call via
        :func:`collect_provenance_metadata`.
        """
        timestamp = datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')
        meta = collect_provenance_metadata()
        return Provenance(
            rms_nav_version=self._rms_nav_version,
            image_et=float(obs.midtime),
            pipeline_run_iso8601=timestamp,
            rms_nav_git_sha=meta.git_sha,
            spice_kernels=meta.spice_kernels,
            static_data_hashes=meta.static_data_hashes,
            technique_names=tuple(sorted(cls.name for cls in NavTechnique._registry)),
            extractor_names=tuple(sorted(m.name for m in self._registry.models)),
        )

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
