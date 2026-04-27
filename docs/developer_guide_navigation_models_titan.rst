=====
Titan
=====

:class:`~nav.nav_model.nav_model_titan.NavModelTitan` is a registered
placeholder.  Atmospheric-body navigation requires a haze-aware limb-fit
algorithm with per-filter haze profiles; that algorithm is not in the
current implementation.

The model registers with the orchestrator's auto-discovery registry like
any other concrete NavModel but contributes nothing on a real image:
``create_model`` records ``self._metadata['stub'] = True``,
``to_features`` returns an empty list, and ``to_annotations`` returns an
empty :class:`~nav.annotation.annotations.Annotations` collection.  The
orchestrator falls through to other features (stars, rings) on Titan
scenes.

The real Titan algorithm will replace the ``to_features`` and
``to_annotations`` implementations when it lands; no API changes are
needed in the orchestrator.
