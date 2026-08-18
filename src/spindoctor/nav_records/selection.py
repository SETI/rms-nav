"""Which of a source's records a caller is asking for.

A program that reads navigation records almost never wants all of them.  It
wants one root of several, or the top-level directories a run was restricted to,
or the images a queue task handed it, or one mission, or one span of time.  Every
one of those is a narrowing of the same question, so they arrive together in one
value and each backend applies whichever of them it can answer cheaply: a walk
narrows by looking at fewer directories, a query narrows in its ``WHERE`` clause,
and the answer is the same either way.

A selection is a value rather than a builder.  It is frozen, so a caller can hand
the same one to two sources and know that neither changed it, and so that a
selection no record can satisfy -- an inverted range, a bound that is not a
number, a key that names no image under a root -- is refused where it is written
rather than where it is applied.

Everything a selection carries is checked here, and that is what makes the two
backends agree.  A walk and a query cannot be made to refuse alike by writing the
same refusal twice; they refuse alike because there is one place a selection can
be wrong and neither of them is reached from it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from spindoctor.nav_records.document import stub_refusal, subtree_refusal

__all__ = ['Selection']


@dataclass(frozen=True)
class Selection:
    """The records a caller is asking a source for.

    Every field narrows, and the fields combine: a selection naming a mission
    and a time span covers that mission's records inside that span.  A field
    left at its default narrows nothing, so the empty selection covers every
    record the source holds.

    There is deliberately no parameter asking for an order, and its absence is a
    decision rather than an omission.  Neither storage can promise a total order
    over a stream without giving up the streaming: a walk cannot know an image's
    epoch until it has read the document, so ordering by time means holding
    every record of the selection in memory, which is the cost the stream exists
    to avoid; and a database sorting text sorts it under the server's own
    collation, so the same rows come back in one order from one server and
    another order from the next.  Each source therefore yields in the order it
    finds records and says so, and a caller that needs a total order calls
    ``sorted()`` on the stream and pays for it knowingly.

    Parameters:
        roots: The roots to read, of the roots the source holds.  Empty means
            all of them.  A root is bound to the source rather than named here
            because a source has to know its roots before it can answer for one
            image; this only narrows to some of what it already holds.
        subtrees: The top-level directories under each root to descend into.
            Empty means the whole root.  It is named for what it is rather than
            for what a PDS3 tree calls it, since the same first path component
            of a stub is a bundle or a collection under PDS4.  Each is one
            directory immediately under the root: a walk joins it and a query
            compares it to the first component of a stub, and only a single
            component means one thing to both.
        stubs: The images to read, named outright.  This is what a queue task
            carries: a worker must read exactly the files it was given, and read
            them in batches.  It names images rather than narrowing to them, so
            a source reads them in the order they are given here.  Each is a key
            under one root rather than a path, so none of them may be absolute,
            name a parent directory, or carry a null byte.
        instrument: The mission to keep, matched against what each record names
            as its instrument.  None keeps every mission.
        start_et: Earliest exposure midtime to keep, in TDB seconds past J2000,
            or None for no lower bound.
        stop_et: Latest exposure midtime to keep, on the same scale, or None for
            no upper bound.  Both bounds are inclusive, so an exposure exactly
            on one is inside.

    Raises:
        ValueError: If a field is not what it says it is -- a sequence field
            that is not a sequence of text, a mission that is not text, a bound
            that is not a finite number -- or if a named subtree is not one
            directory under a root, or a named stub is not a key under one, or
            both time bounds are given with the start after the stop.  A
            selection is checked where it is written rather than where it is
            applied, because a backend meeting a value it cannot use refuses in
            its own storage's terms: one raises out of a query builder in a
            language its caller cannot name, another yields nothing, and a
            swapped range selects nothing at all -- which a run cannot tell from
            a clean pass over a quiet span.
    """

    roots: tuple[str, ...] = ()
    subtrees: tuple[str, ...] = ()
    stubs: tuple[str, ...] = ()
    instrument: str | None = None
    start_et: float | None = None
    stop_et: float | None = None

    def __post_init__(self) -> None:
        """Refuse anything neither backend could answer alike.

        Raises:
            ValueError: If a field is not of the type it declares, if a subtree
                is not one directory under a root, if a stub is not a key under
                one, if a time bound is not a finite number, or if both bounds
                are given with the start after the stop.
        """
        _text_sequence(self.roots, 'roots')
        _text_sequence(self.subtrees, 'subtrees')
        _text_sequence(self.stubs, 'stubs')
        if self.instrument is not None and not isinstance(self.instrument, str):
            raise ValueError(
                f'the instrument is a {type(self.instrument).__name__}, and a mission is '
                f'named by text'
            )
        for subtree in self.subtrees:
            refusal = subtree_refusal(subtree)
            if refusal is not None:
                raise ValueError(f'{subtree!r} is not a subtree of a results root: {refusal}')
        for stub in self.stubs:
            refusal = stub_refusal(stub)
            if refusal is not None:
                raise ValueError(f'{stub!r} is not a results path stub: {refusal}')
        _finite_bound(self.start_et, 'start_et')
        _finite_bound(self.stop_et, 'stop_et')
        if self.start_et is not None and self.stop_et is not None and self.start_et > self.stop_et:
            raise ValueError(
                f'the time range is inverted: its start {self.start_et!r} is after its '
                f'stop {self.stop_et!r}'
            )

    @property
    def bounded_in_time(self) -> bool:
        """Whether this selection places any bound on a record's exposure midtime.

        Returns:
            True when either bound is given, which is when a record recording no
            usable midtime cannot be shown to belong to the selection.
        """
        return self.start_et is not None or self.stop_et is not None


def _text_sequence(value: Sequence[str], field: str) -> None:
    """Refuse a field that should hold several names and holds something else.

    A single string is a sequence of one-character strings, so a caller that
    wrote one where a tuple belongs would otherwise narrow to a root spelled
    ``/`` and a root spelled ``d``.

    Parameters:
        value: What the field holds.
        field: Its name, for the refusal.

    Raises:
        ValueError: If it is not a tuple or list of strings.
    """
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise ValueError(
            f'{field} is a {type(value).__name__}, and it names zero or more values as a tuple'
        )
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError(f'{field} carries a {type(entry).__name__}, and each of them is text')


def _finite_bound(value: float | None, field: str) -> None:
    """Refuse a time bound that is not a number a record can be compared against.

    A boolean is an integer in Python and would bound a span at zero or one
    second past J2000; a NaN is a number no comparison is ever true of, so a walk
    keeps every record and a query keeps none, which is the sharpest disagreement
    the two storages can have.

    Parameters:
        value: The bound, or None for an absent one.
        field: Its name, for the refusal.

    Raises:
        ValueError: If it is not a finite real number.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f'{field} is a {type(value).__name__}, and a time bound is a number of seconds'
        )
    if not math.isfinite(value):
        raise ValueError(
            f'{field} is {value!r}, and a time bound is a finite number of seconds: a walk '
            f'keeps every record against one and a query keeps none'
        )
