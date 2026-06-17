==========================================================
Simulated Star Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated` is
the simulated-image counterpart of
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`. It is a thin subclass
of the catalog-driven model: it sources its star list from the simulated renderer's
output rather than from a catalog reduction, and inherits the parent's
:data:`~nav.feature.feature_type.NavFeatureType.STAR` feature emission, CRLB
covariance, reliability gate, and annotations unchanged. A simulated star field is
therefore navigated by exactly the same
:class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
:class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`, and
:class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` code a real frame
is.

Unlike :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`, this model *does*
override :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`: it builds one
instance for a simulated observation that rendered at least one star, and the parent
declines simulated observations, so the autonomous registry routes simulated frames to
this subclass and real frames to the parent.

Theory
======

The simulated path is the calibration regime for the star techniques: a developer can
probe star matching with a field whose true offset, photometry, and (planted) camera
roll are known by construction.

The renderer (:func:`~nav.sim.render.render_stars`) builds each star at its *unshifted*
predicted ``(v, u)`` and draws it into the image shifted by the scene's planted offset
and camera roll. This model adopts that unshifted star list as its prediction, so a
technique that detects the shifted peak recovers the planted transform -- the same
prediction / observation split a real navigation has, which is why the recovery
transfers.

Two rendering details make the simulated field faithful to a real one:

- **Pixel-centre convention.** ``psfmodel.eval_rect`` measures its sub-pixel offset
  from the pixel's lower edge (``offset=0`` centres the PSF half a pixel low), whereas
  the navigator's detection centroid and this model's predicted position both treat
  integer index ``i`` as the pixel centre. :func:`~nav.sim.render.render_stars` adds
  0.5 to the eval offset so a star the model predicts at ``(v, u)`` lands there in the
  image, with no half-pixel bias in the recovered offset.
- **Camera roll about the boresight.** A planted ``offset_rotation_deg`` rotates each
  star about the image centre before the translation offset, while the star record
  keeps its unrolled ``(v, u)``; the similarity fit in
  :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` then
  recovers the roll. See :doc:`dev_guide_rotation`.

Restrictions and assumptions
----------------------------

- A simulated star field is clean by construction: there are no body- or ring-occlusion
  conflicts to mark, and the per-image smear vector is zero (the sim renders no attitude
  rate). The reliability gate's occlusion and saturation terms still apply, computed
  from the per-image masks the same way the parent computes them.
- The synthesised effective SNR follows the parent's magnitude-margin formula against
  ``obs.star_max_usable_vmag()``; on a simulated observation that limit is generous, so
  reliability saturates toward 1.0 and the recovered *offset* -- not the calibrated
  confidence -- is the quantity the simulated star tests assert.

Sources of uncertainty
----------------------

Per-feature uncertainty is the parent model's anisotropic CRLB covariance, derived from
the predicted SNR and the (zero) smear vector. The simulated star positions themselves
are exact, so any recovered-offset error reflects detection-centroid noise under the
rendered detector noise, not a prediction error.

Configuration
=============

The model consumes no YAML configuration of its own; it reads the rendered star list
the observation carries on ``sim_star_list``. The per-star geometry comes from the
scene's ``stars.list`` entries (see :doc:`/user_guide/user_guide_simulated_images`), and
the per-image PSF sigma comes from the selected instrument's ``star_psf_sigma`` via the
renderer.

Implementation
==============

Source file: ``src/nav/nav_model/stars/nav_model_stars_simulated.py`` --
:class:`~nav.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`, base
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`. The subclass self-registers
via ``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated.instances_for_obs`
  -- returns one instance for a simulated observation carrying a non-empty
  ``sim_star_list``; an empty list for a real observation or a simulated one with no
  rendered stars.
- :meth:`~nav.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated.create_model`
  -- adopts ``obs.sim_star_list`` as the star list, sets a zero smear vector, and
  populates the same metadata fields the parent records.
- ``to_features`` / ``to_annotations`` -- inherited unchanged from
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`.

Examples
========

A clean planted-offset star field plus the camera-roll fixture live under
``tests/integration/sim_scenes/algorithmic_invariants/`` (``planted_offset_star_field``,
``planted_rotation_star_field``); ``tests/integration/test_sim_algorithmic_invariants.py``
asserts the planted offset and roll are recovered. See
:doc:`/user_guide/user_guide_simulated_images` for the scene-catalog workflow.
