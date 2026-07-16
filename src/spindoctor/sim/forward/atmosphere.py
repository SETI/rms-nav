"""Image-side atmosphere rendering for haze-limb (Titan-class) bodies.

Placeholder module: no current renderer has atmospheric logic to port, so
this exists to reserve the architectural slot.  Phase G adds the exponential
haze layer (scale height, tangent optical depth, forward-scattering phase
behavior, detached shell) as part of body radiance composition; a body
without an ``atmosphere`` block renders hard-limbed exactly as today.
"""

__all__: list[str] = []
