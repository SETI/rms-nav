#!/usr/bin/env python3
"""nav_mosaic_display -- interactive viewer for ring and body reprojection/mosaic files.

Entry points
------------
nav_mosaic_display         -- dispatches on the first positional argument (``rings`` or ``body``)
nav_mosaic_display_rings   -- equivalent to ``nav_mosaic_display rings ...``
nav_mosaic_display_body    -- equivalent to ``nav_mosaic_display body ...``

Usage
-----
nav_mosaic_display_rings [options] FILE [FILE ...]
nav_mosaic_display_body  [options] FILE [FILE ...]

Multiple files are browsed one at a time with Prev/Next or by choosing a
file in the sidebar list (hover for the full path).
Accepts local paths, ``file://`` URLs, and ``gs://`` paths via FileCache.
"""

import argparse
import os
import sys

# Allow running directly from the source tree:
#   python src/main/nav_mosaic_display.py rings file1.fits
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from PyQt6.QtWidgets import QApplication

from nav.ui.mosaic_viewer.body_window import BodyMosaicWindow
from nav.ui.mosaic_viewer.projections import ProjectionKind
from nav.ui.mosaic_viewer.ring_window import RingMosaicWindow
from reproj_cli.args import add_display_args

_PROJ_CHOICES = {
    'rect': ProjectionKind.RECT,
    'polar_n': ProjectionKind.POLAR_N,
    'polar_s': ProjectionKind.POLAR_S,
    'mollweide': ProjectionKind.MOLLWEIDE,
    'sphere3d': ProjectionKind.SPHERE_3D,
}


def _build_parser(mode: str) -> argparse.ArgumentParser:
    """Build the :class:`argparse.ArgumentParser` for ``nav_mosaic_display``.

    Parameters:
        mode: ``'rings'`` or ``'body'``. Selects description text, program name,
            and whether ``--projection`` is registered (body only).

    Returns:
        Parser with a required ``files`` positional (one or more mosaic/reproj
        paths), display/stretch options from :func:`reproj_cli.args.add_display_args`,
        and for ``mode='body'`` only, ``--projection`` (initial map projection).
    """
    description = {
        'rings': (
            'Display ring reprojection or mosaic files produced by nav_mosaic_rings. '
            'Supports tiled dynamic zoom, stretch, color-by, show-radii, EW plot, '
            'radial-slice, axis ticks, and Save-FOV.'
        ),
        'body': (
            'Display body reprojection or mosaic files produced by nav_mosaic_body. '
            'Supports tiled dynamic zoom (including zooming out to the full '
            '0..360 deg longitude and -90..90 deg latitude canvas around the data), '
            'stretch, color-by, a cursor metadata grid, show-parallels/meridians, '
            'axis ticks, and Save-FOV.'
        ),
    }[mode]
    parser = argparse.ArgumentParser(
        prog=f'nav_mosaic_display_{mode}',
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='+',
        metavar='FILE',
        help='One or more reprojection or mosaic files to display.',
    )
    add_display_args(parser)
    if mode == 'body':
        parser.add_argument(
            '--projection',
            choices=list(_PROJ_CHOICES),
            default='rect',
            metavar='PROJ',
            help=(
                'Initial display projection: rect (rectangular), polar_n '
                '(Polar North Stereographic), polar_s (Polar South Stereographic), '
                'mollweide (Mollweide), sphere3d (3D Sphere). Default: rect.'
            ),
        )
    return parser


def _run_rings(args: argparse.Namespace) -> None:
    """Open a :class:`~nav.ui.mosaic_viewer.ring_window.RingMosaicWindow` and run Qt.

    Ensures a :class:`~PyQt6.QtWidgets.QApplication` exists, constructs the
    window from ``args.files`` and stretch settings ``args.stretch_black`` /
    ``args.stretch_white`` / ``args.stretch_gamma``, enables radius and
    longitude axis ticks, shows the window, and calls ``sys.exit(app.exec())``
    (does not return under normal operation).

    Parameters:
        args: Namespace produced by :func:`_build_parser` with ``mode='rings'``.
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = RingMosaicWindow(
        file_paths=args.files,
        initial_black=args.stretch_black,
        initial_white=args.stretch_white,
        initial_gamma=args.stretch_gamma,
        show_longitude_ticks=True,
        show_radius_ticks=True,
    )
    win.show()
    sys.exit(app.exec())


def _run_body(args: argparse.Namespace) -> None:
    """Open a :class:`~nav.ui.mosaic_viewer.body_window.BodyMosaicWindow` and run Qt.

    Same application bootstrap as :func:`_run_rings`. Maps ``args.projection``
    through :data:`_PROJ_CHOICES` (body mode always defines ``--projection``),
    passes parallel/meridian tick flags and stretch fields from ``args``, then
    ``sys.exit(app.exec())``.

    Parameters:
        args: Namespace from :func:`_build_parser` with ``mode='body'``.
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    proj_kind = _PROJ_CHOICES[args.projection]
    win = BodyMosaicWindow(
        file_paths=args.files,
        initial_black=args.stretch_black,
        initial_white=args.stretch_white,
        initial_gamma=args.stretch_gamma,
        show_parallels=args.show_parallels,
        show_meridians=args.show_meridians,
        show_lat_ticks=True,
        show_lon_ticks=True,
        initial_projection=proj_kind,
    )
    win.show()
    sys.exit(app.exec())


def main() -> None:
    """Dispatch on ``rings`` or ``body`` first positional argument."""
    args_list = sys.argv[1:]
    if ('-h' in args_list or '--help' in args_list) and (
        not args_list or args_list[0] not in ('rings', 'body')
    ):
        top_parser = argparse.ArgumentParser(
            prog='nav_mosaic_display',
            description=(
                'Interactive PyQt6 viewer for ring or body reprojection/mosaic files. '
                'The first argument must be rings or body; remaining arguments are '
                'passed to that mode (see nav_mosaic_display_rings / '
                'nav_mosaic_display_body).'
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=(
                'Usage:\n'
                '  nav_mosaic_display rings [options] FILE [FILE ...]\n'
                '  nav_mosaic_display body  [options] FILE [FILE ...]'
            ),
        )
        top_parser.print_help()
        sys.exit(0)
    if not args_list or args_list[0] not in ('rings', 'body'):
        argparse.ArgumentParser(prog='nav_mosaic_display').error(
            'Usage: nav_mosaic_display <rings|body> [options] FILE [FILE ...]'
        )
    mode = args_list[0]
    parser = _build_parser(mode)
    args = parser.parse_args(args_list[1:])
    if mode == 'rings':
        _run_rings(args)
    else:
        _run_body(args)


def rings_main() -> None:
    """Entry point for ``nav_mosaic_display_rings``; prepends ``rings`` to argv."""
    sys.argv = [sys.argv[0], 'rings', *sys.argv[1:]]
    main()


def body_main() -> None:
    """Entry point for ``nav_mosaic_display_body``; prepends ``body`` to argv."""
    sys.argv = [sys.argv[0], 'body', *sys.argv[1:]]
    main()


if __name__ == '__main__':
    main()
