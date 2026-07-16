"""Ghost reflections for the optics stage.

A ghost is a faint internal reflection of the focal-plane image: a displaced,
defocused, low-amplitude copy of the scene added back onto it.  Each ghost
copies the pre-ghost signal (so ghosts do not reflect one another), shifts it
by its offset, blurs it by its defocus, scales it by its amplitude, and adds
it in.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from scipy import ndimage

from spindoctor.sim.forward.stages import SimFrame

__all__ = ['apply_ghosts']


def apply_ghosts(
    frame: SimFrame,
    *,
    ghosts: Sequence[Mapping[str, Any]],
    oversample: int,
) -> None:
    """Add displaced, defocused, scaled copies of the scene in place.

    Parameters:
        frame: The frame whose signal plane receives the ghosts.
        ghosts: The scene ``optics.ghosts`` list of ghost specifications.
        oversample: The render-grid oversampling factor (offsets and defocus
            are in detector pixels and scale to the render grid).
    """
    if not ghosts:
        return
    pre_ghost = frame.signal.copy()
    for ghost in ghosts:
        amplitude = float(ghost.get('amplitude', 0.0))
        if amplitude == 0.0:
            continue
        dv = float(ghost.get('dv_px', 0.0)) * oversample
        du = float(ghost.get('du_px', 0.0)) * oversample
        defocus_sigma = float(ghost.get('defocus_sigma', 0.0)) * oversample
        copy = ndimage.shift(pre_ghost, (dv, du), order=1, mode='constant', cval=0.0)
        if defocus_sigma > 0.0:
            copy = ndimage.gaussian_filter(copy, defocus_sigma, mode='constant')
        frame.signal += amplitude * copy
