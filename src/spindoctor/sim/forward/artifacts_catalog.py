"""Per-instrument artifact defaults for the forward model.

Placeholder module: no current renderer has per-instrument artifact
incidences to port.  Phases B and C populate this catalog (gain-state
tables, PSF kernels, structured-loss incidences, reseau geometry) keyed by
sim instrument name; scenes opt in via ``artifacts: {instrument_defaults:
true}`` -- naming an instrument selects a geometry and a detector, never a
set of defects.
"""

__all__: list[str] = []
