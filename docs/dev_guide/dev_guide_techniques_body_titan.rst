===================
Titan (unsupported)
===================

Titan is a body, but its opaque haze hides the surface: the visible "limb" is the haze top,
which varies with wavelength, and at high phase Titan is not even a circle. Ellipsoid limb /
terminator / disc navigation is therefore systematically wrong rather than merely noisy, so
no autonomous technique navigates Titan today.

:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` renders no navigable feature for
Titan; it records a no-result instead. A frame whose only content is Titan fails with
:attr:`~spindoctor.support.status_reason.NavStatusReason.TITAN_UNSUPPORTED` rather than a
silent empty failure, while a Titan-bearing frame that also holds a resolved moon, a ring, or
stars navigates on that other content.

Planned support
===============

A haze-aware limb-fit technique with per-filter haze profiles is planned but not yet
implemented. When it is added it will replace the recorded no-result with a real Titan
navigation contribution. The model documentation is
:doc:`dev_guide_navigation_models_titan`.
