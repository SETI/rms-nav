"""Refusing a kernel pool in which two frame kernels define the same frame.

The reproduction test asks the furnished pool for the attitude an image
navigated against, and it asks through the image's camera frame.  That frame is
defined by a frame kernel, and a frame kernel is a text kernel: when two of them
assign the same variable, the one furnished last wins and nothing is reported.
Two versions of a mission's frame kernel differ in exactly those assignments --
the fixed rotation from the spacecraft to the camera is one of them -- so a
corpus navigated across a frame kernel upgrade would be reproduced entirely
against whichever version happened to be furnished last.

Every image navigated under the other version then fails to reproduce, the run
writes nothing, and it reports the whole corpus as having no reproducing
baseline: the drift verdict, delivered for a pool the run built itself.  That is
the misdiagnosis the assignment step's missing-frame check exists to prevent,
arriving by a different route, so it is refused the same way -- before any
candidate baseline is tried, naming the two kernels and the frame they
disagree about.

The check is deliberately over the frames the run's images actually name rather
than over every frame a kernel defines.  A mission furnishes several frame
kernels at once by design -- Cassini's dynamic, rocks and status kernels beside
its main one -- and they overlap in ways that are none of this run's business.
What is this run's business is the camera frame each image was navigated
through and the frame naming the object its correction targets.
"""

from collections.abc import Collection, Mapping

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.pool import furnished

# The extensions a frame kernel is stored under in the holdings.
FK_SUFFIXES = frozenset({'.tf', '.tk'})


def camera_frame_is_defined(camera_frame: str) -> bool:
    """Report whether the furnished kernels define one camera frame by name.

    Parameters:
        camera_frame: SPICE name of the frame, for example
            ``'CASSINI_ISS_NAC'``.

    Returns:
        True when the pool resolves that name to a frame id.
    """
    try:
        cspyce.namfrm(camera_frame)
    except LookupError:
        return False
    return True


def object_frame_is_defined(ck_frame_id: int) -> bool:
    """Report whether the furnished kernels name the frame of one CK object.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.

    Returns:
        True when the pool gives that object a frame name.
    """
    try:
        cspyce.frmnam(ck_frame_id)
    except LookupError:
        return False
    return True


def frames_defined_by(
    path: FCPath, camera_frames: Collection[str], ck_frame_ids: Collection[int]
) -> frozenset[str]:
    """Return which of the frames a run needs one frame kernel defines.

    Parameters:
        path: The frame kernel to probe, local or remote.
        camera_frames: The camera frames the run's images name.
        ck_frame_ids: The CK objects the run's corrections target.

    Returns:
        The needed frames it defines, camera frames by their own name and CK
        objects as ``'object <id>'``.

    Raises:
        OSError: if the kernel cannot be furnished.
    """
    with furnished(path):
        return frozenset(_defined_frame_labels(camera_frames, ck_frame_ids))


def _defined_frame_labels(
    camera_frames: Collection[str], ck_frame_ids: Collection[int]
) -> list[str]:
    """Label the needed frames the currently furnished kernels define.

    Camera frames are labeled by their own name and CK objects as
    ``'object <id>'``.  Both callers compare their answers against each
    other's, so the one spelling lives here.

    Parameters:
        camera_frames: The camera frames the run's images name.
        ck_frame_ids: The CK objects the run's corrections target.

    Returns:
        The labels of the frames the pool defines, in no particular order.
    """
    return [frame for frame in camera_frames if camera_frame_is_defined(frame)] + [
        f'object {ck_frame_id}'
        for ck_frame_id in ck_frame_ids
        if object_frame_is_defined(ck_frame_id)
    ]


def require_one_frame_kernel_per_frame(
    candidates: Mapping[str, FCPath],
    *,
    camera_frames: Collection[str],
    ck_frame_ids: Collection[int],
) -> None:
    """Refuse a pool in which two frame kernels define one frame the run needs.

    Each candidate is furnished on its own and asked which of the run's frames
    it defines, so the answer is a property of that kernel rather than of the
    order the pool was built in.

    Parameters:
        candidates: The frame kernels the run's images name and its kernel
            directories resolve, keyed by basename.
        camera_frames: The camera frames the run's images name.
        ck_frame_ids: The CK objects the run's corrections target.

    Raises:
        ValueError: if one of those frames is already defined by a furnished
            kernel, which would make every candidate look like its author; or
            if two candidates define the same one, since the pool would then
            answer with whichever was furnished last and every image navigated
            through the other would be reported as having no baseline.
        OSError: if a candidate cannot be furnished for the probe.
    """
    already = sorted(_defined_frame_labels(camera_frames, ck_frame_ids))
    if len(already) > 0:
        raise ValueError(
            f'the frame(s) {already} are already defined by a furnished kernel; which frame kernel '
            f'defines a frame cannot be established while another already answers for it'
        )
    definers: dict[str, list[str]] = {}
    for basename in sorted(candidates):
        for frame in frames_defined_by(candidates[basename], camera_frames, ck_frame_ids):
            definers.setdefault(frame, []).append(basename)
    contested = {frame: names for frame, names in definers.items() if len(names) > 1}
    if len(contested) > 0:
        detail = '; '.join(f'{frame} by {names}' for frame, names in sorted(contested.items()))
        raise ValueError(
            f'more than one frame kernel defines a frame this run needs: {detail}. The pool would '
            f'answer with whichever was furnished last, and every image navigated through the '
            f'other would be reported as having no reproducing baseline'
        )
