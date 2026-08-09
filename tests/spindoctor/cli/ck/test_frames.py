"""Hermetic tests for ``spindoctor.cli.ck.frames``.

Two versions of a mission's frame kernel differ in the fixed rotation from the
spacecraft to the camera, and a text kernel's last assignment wins, so a pool
holding both answers the reproduction test with whichever was furnished last.
These tests write two such versions and check that the run refuses rather than
reproducing half its corpus against the wrong one -- and that a mission's
several unrelated frame kernels, which is the normal case, are not refused.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import cspyce
import pytest
from filecache import FCPath

from spindoctor.cli.ck.frames import (
    camera_frame_is_defined,
    frames_defined_by,
    object_frame_is_defined,
    require_one_frame_kernel_per_frame,
)

# The CK object under test and the camera frame fixed to it.  Both ids are
# real, since the rest of the writer refuses any other.
CK_FRAME_ID = -82000
CAMERA_FRAME = 'SD_TEST_NAC'
OTHER_CAMERA_FRAME = 'SD_TEST_WAC'

_BUS_AND_CAMERA = """KPL/FK

A frame kernel defining the bus and one camera fixed to it.  The camera's
angles are a parameter so that two versions can disagree about them, which is
what two versions of a real frame kernel do.

\\begindata
FRAME_SD_TEST_BUS         = -82000
FRAME_-82000_NAME         = 'SD_TEST_BUS'
FRAME_-82000_CLASS        = 3
FRAME_-82000_CLASS_ID     = -82000
FRAME_-82000_CENTER       = -82
CK_-82000_SCLK            = -82
CK_-82000_SPK             = -82

FRAME_{camera}         = {camera_id}
FRAME_{camera_id}_NAME         = '{camera}'
FRAME_{camera_id}_CLASS        = 4
FRAME_{camera_id}_CLASS_ID     = {camera_id}
FRAME_{camera_id}_CENTER       = -82
TKFRAME_{camera_id}_SPEC       = 'ANGLES'
TKFRAME_{camera_id}_RELATIVE   = 'SD_TEST_BUS'
TKFRAME_{camera_id}_ANGLES     = ( {angles} )
TKFRAME_{camera_id}_AXES       = ( 3, 1, 3 )
TKFRAME_{camera_id}_UNITS      = 'DEGREES'
\\begintext
"""

_UNRELATED = """KPL/FK

A frame kernel of the kind a mission furnishes beside its main one: it defines
a body frame and nothing this run needs.

\\begindata
FRAME_SD_TEST_ROCK        = 1999999
FRAME_1999999_NAME        = 'SD_TEST_ROCK'
FRAME_1999999_CLASS       = 4
FRAME_1999999_CLASS_ID    = 1999999
FRAME_1999999_CENTER      = 599
TKFRAME_1999999_SPEC      = 'ANGLES'
TKFRAME_1999999_RELATIVE  = 'J2000'
TKFRAME_1999999_ANGLES    = ( 0.0, 0.0, 0.0 )
TKFRAME_1999999_AXES      = ( 3, 1, 3 )
TKFRAME_1999999_UNITS     = 'DEGREES'
\\begintext
"""


@pytest.fixture
def frame_root(tmp_path: Path) -> Iterator[Path]:
    """Yield a directory for candidate frame kernels, with nothing furnished.

    The check refuses to run while a frame it is looking for is already
    defined, so this fixture furnishes nothing at all and unloads anything a
    test leaves behind.
    """
    loaded_before = {str(cspyce.kdata(at, 'ALL')[0]) for at in range(int(cspyce.ktotal('ALL')))}
    try:
        yield tmp_path
    finally:
        for at in reversed(range(int(cspyce.ktotal('ALL')))):
            path = str(cspyce.kdata(at, 'ALL')[0])
            if path not in loaded_before:
                cspyce.unload(path)


def _fk(
    root: Path,
    name: str,
    *,
    camera: str = CAMERA_FRAME,
    camera_id: int = -82361,
    angles: str = '-90.0, 60.0, 30.0',
) -> FCPath:
    """Write one frame kernel defining the bus and a camera, and return its path.

    Parameters:
        root: Directory to write into.
        name: Basename of the kernel.
        camera: Name of the camera frame it defines.
        camera_id: Id of that frame.
        angles: The fixed rotation it assigns, so two versions can disagree.

    Returns:
        The kernel's path.
    """
    path = root / name
    path.write_text(_BUS_AND_CAMERA.format(camera=camera, camera_id=camera_id, angles=angles))
    return FCPath(str(path))


def _unrelated(root: Path, name: str) -> FCPath:
    """Write a frame kernel defining nothing this run needs.

    Parameters:
        root: Directory to write into.
        name: Basename of the kernel.

    Returns:
        The kernel's path.
    """
    path = root / name
    path.write_text(_UNRELATED)
    return FCPath(str(path))


def test_one_frame_kernel_defining_the_camera_frame_is_accepted(frame_root: Path) -> None:
    """The ordinary case: one kernel defines the frames, and it is allowed."""
    require_one_frame_kernel_per_frame(
        {'main.tf': _fk(frame_root, 'main.tf')},
        camera_frames=[CAMERA_FRAME],
        ck_frame_ids=[CK_FRAME_ID],
    )


def test_unrelated_frame_kernels_beside_it_are_accepted(frame_root: Path) -> None:
    """A mission furnishes several by design; only the run's own frames count."""
    require_one_frame_kernel_per_frame(
        {
            'main.tf': _fk(frame_root, 'main.tf'),
            'rocks.tf': _unrelated(frame_root, 'rocks.tf'),
        },
        camera_frames=[CAMERA_FRAME],
        ck_frame_ids=[CK_FRAME_ID],
    )


def test_two_versions_defining_the_camera_frame_are_refused(frame_root: Path) -> None:
    """Whichever was furnished last would silently define the camera."""
    candidates = {
        'v40.tf': _fk(frame_root, 'v40.tf', angles='-90.0, 60.0, 30.0'),
        'v43.tf': _fk(frame_root, 'v43.tf', angles='-90.0, 60.0, 30.5'),
    }
    with pytest.raises(ValueError, match='more than one frame kernel'):
        require_one_frame_kernel_per_frame(
            candidates, camera_frames=[CAMERA_FRAME], ck_frame_ids=[CK_FRAME_ID]
        )


def test_the_refusal_names_the_frame_and_both_kernels(frame_root: Path) -> None:
    """So the operator can see which two versions to choose between."""
    candidates = {
        'v40.tf': _fk(frame_root, 'v40.tf'),
        'v43.tf': _fk(frame_root, 'v43.tf', angles='-90.0, 60.0, 30.5'),
    }
    with pytest.raises(ValueError, match=re.escape(f"{CAMERA_FRAME} by ['v40.tf', 'v43.tf']")):
        require_one_frame_kernel_per_frame(
            candidates, camera_frames=[CAMERA_FRAME], ck_frame_ids=[CK_FRAME_ID]
        )


def test_two_versions_are_refused_for_the_object_frame_too(frame_root: Path) -> None:
    """The frame naming the corrected object is checked the same way.

    Each kernel defines a different camera, so only the bus frame they share
    can be what the refusal is about.
    """
    candidates = {
        'a.tf': _fk(frame_root, 'a.tf', camera=CAMERA_FRAME, camera_id=-82361),
        'b.tf': _fk(frame_root, 'b.tf', camera=OTHER_CAMERA_FRAME, camera_id=-82362),
    }
    with pytest.raises(ValueError, match=re.escape(f'object {CK_FRAME_ID}')):
        require_one_frame_kernel_per_frame(
            candidates, camera_frames=[CAMERA_FRAME], ck_frame_ids=[CK_FRAME_ID]
        )


def test_a_frame_already_defined_is_refused(frame_root: Path) -> None:
    """A kernel furnished earlier would look like the author of every frame."""
    already = _fk(frame_root, 'already.tf')
    cspyce.furnsh(str(already))
    try:
        with pytest.raises(ValueError, match='already defined'):
            require_one_frame_kernel_per_frame(
                {'main.tf': _fk(frame_root, 'main.tf')},
                camera_frames=[CAMERA_FRAME],
                ck_frame_ids=[CK_FRAME_ID],
            )
    finally:
        cspyce.unload(str(already))


@pytest.mark.usefixtures('frame_root')
def test_no_candidate_at_all_is_accepted() -> None:
    """A missing frame kernel is the assignment step's refusal, not this one.

    That refusal names the frame; refusing here would say a run holds too many
    frame kernels when it holds none.
    """
    require_one_frame_kernel_per_frame({}, camera_frames=[CAMERA_FRAME], ck_frame_ids=[CK_FRAME_ID])


def test_the_probe_leaves_the_pool_as_it_found_it(frame_root: Path) -> None:
    """Every candidate is unloaded again, whether or not it defined anything."""
    require_one_frame_kernel_per_frame(
        {'main.tf': _fk(frame_root, 'main.tf')},
        camera_frames=[CAMERA_FRAME],
        ck_frame_ids=[CK_FRAME_ID],
    )
    assert camera_frame_is_defined(CAMERA_FRAME) is False


def test_a_kernel_reports_the_frames_it_defines(frame_root: Path) -> None:
    """Both kinds of needed frame come back named."""
    defined = frames_defined_by(_fk(frame_root, 'main.tf'), [CAMERA_FRAME], [CK_FRAME_ID])
    assert defined == frozenset({CAMERA_FRAME, f'object {CK_FRAME_ID}'})


def test_a_kernel_reports_no_frame_it_does_not_define(frame_root: Path) -> None:
    """An unrelated kernel contributes nothing to the contest."""
    assert frames_defined_by(_unrelated(frame_root, 'rocks.tf'), [CAMERA_FRAME], [CK_FRAME_ID]) == (
        frozenset()
    )


@pytest.mark.usefixtures('frame_root')
def test_an_undefined_object_frame_reads_as_undefined() -> None:
    """The probe answers False rather than raising for an unknown object."""
    assert object_frame_is_defined(-999999) is False
