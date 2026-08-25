"""Hermetic tests for ``spindoctor.support.attrdict``.

The class publishes its own dict as its ``__dict__``, which is what makes
attribute access work and is also its one hazard: any library that sets an
attribute on one of these objects is inserting a key into configuration data.
The tests below pin both halves -- that our own attribute writes still land as
keys, and that a library which honors the opt-out protocol leaves the mapping
alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from spindoctor.support.attrdict import AttrDict


def walk_and_annotate(obj: Any, *, honor_optout: bool) -> bool:
    """Stand in for a library that caches bookkeeping on objects it inspects.

    This mirrors what ``oops.mutable._get_info`` does: it walks whatever it can
    reach and writes its own verdict back onto each object, skipping any object
    that advertises ``_IS_IMMUTABLE``.

    Parameters:
        obj: The object to inspect.
        honor_optout: True to skip objects carrying the opt-out marker, as the
            protocol requires; False to write unconditionally.

    Returns:
        True if bookkeeping was written.
    """
    if honor_optout and hasattr(obj, '_IS_IMMUTABLE'):
        return False
    obj._FOREIGN_bookkeeping = object()
    return True


def test_attribute_access_reads_a_key() -> None:
    """The point of the class: a key is readable as an attribute."""
    section = AttrDict({'planet': 'SATURN'})
    assert section.planet == 'SATURN'


def test_setting_an_attribute_writes_a_key() -> None:
    """And the inverse, which is what makes a foreign write dangerous."""
    section = AttrDict()
    # setattr rather than ``section.planet =``, which mypy refuses on a class
    # whose attributes are its keys; the runtime path is the same one a foreign
    # library takes.
    setattr(section, 'planet', 'SATURN')  # noqa: B010
    assert section['planet'] == 'SATURN'


def test_a_missing_attribute_raises_attribute_error() -> None:
    """A missing key reads as a missing attribute, not a KeyError."""
    section = AttrDict()
    with pytest.raises(AttributeError, match="Attribute 'planet' not found"):
        _ = section.planet


def test_the_opt_out_marker_is_advertised() -> None:
    """A cooperating library can see that this mapping is not its scratch space."""
    assert hasattr(AttrDict(), '_IS_IMMUTABLE')


def test_the_marker_is_not_itself_a_key() -> None:
    """Advertising the opt-out must not put anything in the data.

    The marker lives on the class, so it is visible to ``hasattr`` without
    occupying the mapping; a marker stored as a key would be the same defect it
    exists to prevent.
    """
    section = AttrDict({'planet': 'SATURN'})
    assert '_IS_IMMUTABLE' not in section


def test_the_marker_leaves_the_mapping_equal_to_its_plain_dict() -> None:
    """Nothing about length, iteration, or equality changes."""
    assert AttrDict({'planet': 'SATURN'}) == {'planet': 'SATURN'}


def test_a_cooperating_walker_writes_nothing() -> None:
    """The behavior the marker exists to produce.

    ``oops`` builds a Backplane from an observation, and the observation holds
    the shared Config, whose sections are these mappings.  Without the opt-out
    the walk reaches them and every section gains a key that no configuration
    file declares, which the logging-config validator then rejects.
    """
    section = AttrDict({'planet': 'SATURN'})
    wrote = walk_and_annotate(section, honor_optout=True)
    assert not wrote


def test_a_cooperating_walker_leaves_the_keys_untouched() -> None:
    """Same case, stated as the invariant a caller actually depends on."""
    section = AttrDict({'planet': 'SATURN'})
    walk_and_annotate(section, honor_optout=True)
    assert list(section) == ['planet']


def test_an_ordinary_object_is_still_annotated() -> None:
    """The walker under test really does write when nothing opts out.

    Without this the cooperating-walker tests above would pass against a
    walker that never writes to anything.
    """

    class Plain:
        """An object with an ordinary attribute dict."""

    plain = Plain()
    assert walk_and_annotate(plain, honor_optout=True)


def test_a_subclass_inherits_the_marker() -> None:
    """A specialized section is protected for the same reason the base is."""

    class Section(AttrDict):
        """A hypothetical AttrDict subclass."""

    assert not walk_and_annotate(Section({'planet': 'SATURN'}), honor_optout=True)
