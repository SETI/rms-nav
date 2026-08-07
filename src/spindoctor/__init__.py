# Version
try:
    from ._version import __version__
except ImportError:
    __version__ = 'Version unspecified'

# Re-exported explicitly so that reading the version is a supported import
# rather than one that depends on how the name got here.
__all__ = ['__version__']
