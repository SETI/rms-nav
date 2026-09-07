"""How much of a pass over a results tree runs at once.

A pass over a cloud root is latency and not bandwidth: a listing is one round
trip, a document is another, and a navigation document is a few kilobytes.  How
much of that latency a pass can overlap is the difference between a routine
re-ingest and one to plan around, so the walk lists several directories at once
and retrieval fetches several documents at once.

**The right values are a property of a machine and its network, not of this
program.**  The shipped ones were tuned on one machine against one bucket, and
the ceiling they found was that service's round trip on that day.  Somewhere
else -- a different provider, a link with more or less latency, a service with
its own opinion about concurrent requests, or a local disk that pays no latency
at all -- the useful values are different, and nothing here can know them.  They
are therefore defaults rather than constants, and ``results_index`` in the
configuration is where a machine says otherwise.

The library takes them as an argument rather than reading the configuration
itself: :mod:`spindoctor.nav_records` answers about documents and roots and
depends on nothing that decides policy, so a caller that has a configuration
passes what it says and a caller that has none gets the defaults.
"""

from dataclasses import dataclass, fields
from typing import Any

__all__ = ['TreeTuning']


@dataclass(frozen=True)
class TreeTuning:
    """How much of a pass runs at once.

    Parameters:
        walk_threads: How many directories are listed at once.
        walk_directories_at_once: How many directories one round of the walk
            takes off its frontier.  This bounds what a round holds rather
            than what it fetches: the walk keeps the entries of the
            directories it is listing, so a round over a whole level of a wide
            tree would hold that level at once.
        retrieve_threads: How many documents are fetched at once within one
            batch.
        retrieve_batch_size: How many documents are retrieved in one batched
            download.  Below ``retrieve_threads`` the pool runs a fraction of
            the requests it could, so a batch is several times the thread
            count and the two are raised together.
    """

    walk_threads: int = 32
    walk_directories_at_once: int = 256
    retrieve_threads: int = 64
    retrieve_batch_size: int = 1024

    def __post_init__(self) -> None:
        """Refuse a value that would stop a pass rather than tune it.

        Raises:
            ValueError: If any setting is not a positive integer, or if the
                batch is smaller than the thread count it feeds -- which is
                not a slow configuration but a pool that cannot fill, and is
                worth saying at startup rather than leaving to be measured.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f'results_index.{field.name} must be a positive integer; got {value!r}'
                )
        if self.retrieve_batch_size < self.retrieve_threads:
            raise ValueError(
                'results_index.retrieve_batch_size must be at least '
                'results_index.retrieve_threads, or the download pool cannot fill; '
                f'got batch {self.retrieve_batch_size} against {self.retrieve_threads} threads'
            )

    @classmethod
    def from_config_section(cls, section: Any) -> 'TreeTuning':
        """Build the tuning a configuration asks for, defaulting what it omits.

        Parameters:
            section: The ``results_index`` configuration section, or anything
                that answers to the field names by attribute; None for a
                caller with no configuration to consult.

        Returns:
            The tuning, with every field the section does not name left at its
            default.

        Raises:
            ValueError: If a value the section does name is not usable, naming
                the setting.
        """
        if section is None:
            return cls()
        named = {
            field.name: getattr(section, field.name)
            for field in fields(cls)
            if getattr(section, field.name, None) is not None
        }
        return cls(**named)
