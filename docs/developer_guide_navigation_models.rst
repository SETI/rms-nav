====================
Navigation models
====================

:class:`~nav.nav_model.nav_model.NavModel` is the abstract base class for
synthetic model generators. Each subclass implements ``create_model(...)`` to
populate image arrays, masks, optional quality metadata, and annotations that
support navigation techniques (for example correlation against the observation).

Concrete implementations cover stars, planetary bodies (including a simulated
variant), Saturn's rings (real and simulated), Titan, and a combined model that
merges contributions from multiple sources. The API surface is summarized under
:doc:`api_reference/api_nav_model`.

The following sections describe design and behavior for each model family. Only
the rings chapter is filled in today; the others are placeholders for future
expansion.

.. toctree::
   :maxdepth: 1

   developer_guide_navigation_models_stars
   developer_guide_navigation_models_bodies
   developer_guide_navigation_models_rings
   developer_guide_navigation_models_titan
