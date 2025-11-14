from typing import Any

from oops.observation.snapshot import Snapshot

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""

    def __init__(self,
                 snapshot: Snapshot,
                 **kwargs: Any) -> None:
        """Initializes a new ObsSnapshotInst instance.

        Parameters:
            snapshot: The Snapshot object to wrap.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(snapshot, **kwargs)
