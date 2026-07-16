"""Per-instrument artifact defaults for the forward model.

Placeholder module: the catalog is deliberately empty, since no
per-instrument artifact incidences are defined.  Its intended contents are
per-instrument defaults (gain-state tables, PSF kernels, structured-loss
incidences, reseau geometry) keyed by sim instrument name, with scenes
opting in via ``artifacts: {instrument_defaults: true}`` -- naming an
instrument selects a geometry and a detector, never a set of defects.
"""

__all__: list[str] = []
