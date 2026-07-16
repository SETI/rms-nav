"""The simulator's image side: the forward model that renders the scene.

This package owns everything that turns a validated ``sim_params`` scene
into the image the navigator is handed: scene-radiance composition, optics,
the detector model, and telemetry effects, run as an ordered pipeline of
stages over a :class:`~spindoctor.sim.forward.stages.SimFrame`.

The navigator-side simulated NavModels never import from this package; they
consume only the filtered idealized view (``obs.nav_params``) built at the
``ObsSim`` boundary.  Geometry conventions shared with the navigator side
live in the neutral ``spindoctor.sim.*_geometry`` modules.

Entry point: :func:`spindoctor.sim.render.render_combined_model`, a thin
driver over :func:`spindoctor.sim.forward.pipeline.run_pipeline`.
"""
