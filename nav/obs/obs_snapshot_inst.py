from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""
    pass
