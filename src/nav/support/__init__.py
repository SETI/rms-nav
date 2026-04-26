"""Shared utilities for navigation code.

This package groups small, reusable helpers. Import from submodules directly, for
example ``from nav.support import image`` or
``from nav.support.image import shift_array``.

Modules:

    ``types``
        NumPy typing aliases (e.g. ``NDArrayFloatType``), ``PathLike``, and protocols
        such as ``MutableStar``.
    ``image``
        Two-dimensional array helpers: shifting, padding, cropping, normalization,
        and FFT-related image operations.
    ``correlate``
        Fourier-domain and template-matching utilities (e.g. normalized
        cross-correlation) built on ``image`` and ``misc``.
    ``misc``
        Miscellaneous helpers, including sky-coordinate formatting and oops-backed
        utilities.
    ``time``
        Wall-clock helpers: ISO strings, timezone-aware datetimes, and Julian
        conversions.
    ``file``
        YAML/JSON serialization helpers and ``clean_obj`` for stripping NumPy scalars
        from nested structures.
    ``constants``
        Common mathematical constants (e.g. ``PI``, ``HALFPI``).
    ``nav_base``
        ``NavBase``, a small base class wiring ``Config`` and ``PdsLogger`` for nav
        objects.
    ``attrdict``
        ``AttrDict``, a ``dict`` subclass that supports attribute-style key access.
    ``flux``
        Legacy flux and filter-convolution experiments; most of the implementation is
        commented out but kept for reference.
"""

__all__ = [
    'attrdict',
    'constants',
    'correlate',
    'file',
    'flux',
    'image',
    'misc',
    'nav_base',
    'time',
    'types',
]
