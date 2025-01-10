import re
from typing import Any, cast

from .dataset_pds3 import DataSetPDS3


class DataSetVoyagerISS(DataSetPDS3):

    _MIN_5xxx_VOL1 = 5101
    _MAX_5xxx_VOL1 = 5120
    _MIN_5xxx_VOL2 = 5201
    _MAX_5xxx_VOL2 = 5214
    _MIN_6xxx_VOL1 = 6101
    _MAX_6xxx_VOL1 = 6121
    _MIN_6xxx_VOL2 = 6201
    _MAX_6xxx_VOL2 = 6215
    _MIN_7xxx_VOL2 = 7201
    _MAX_7xxx_VOL2 = 7207
    _MIN_8xxx_VOL2 = 8201
    _MAX_8xxx_VOL2 = 8210

    @staticmethod
    def _get_filespec(row: dict[str, Any]) -> str:
        return cast(str, row['FILE_SPECIFICATION_NAME'])

    @staticmethod
    def _parse_filespec(filespec: str,
                        volumes: list[str],
                        volume: str,
                        index_tab_abspath: str) -> str | None:
        parts = filespec.split('/')
        if parts[0].upper() != 'DATA':
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'PRIMARY_FILE_SPECIFICATION "{filespec}"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 8 or range_dir[0] != 'C':
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'PRIMARY_FILE_SPECIFICATION "{filespec}"')
        if not img_name.endswith('_GEOMED.LBL'):
            return None
        if volumes:
            found_full_spec = False
            good_full_spec = False
            for vol in volumes:
                idx = vol.find('/')
                if idx == -1:
                    continue
                # Looking for volume/subdir
                found_full_spec = True
                if f'{volume}/{range_dir}' == vol:
                    good_full_spec = True
                    break
            if found_full_spec and not good_full_spec:
                return None
        return img_name

    @staticmethod
    def _extract_image_number(f: str) -> int | None:
        m = re.match(r'C(\d{7})_GEOMED\.\w+', f)
        if m is None:
            return None
        return int(m[1])

    @staticmethod
    def _extract_camera(f: str) -> str | None:
        return 'X'

    _DATASET_LAYOUT = {
        'all_volume_names': [f'VGISS_{x:04d}' for x in
                             list(range(_MIN_5xxx_VOL1, _MAX_5xxx_VOL1+1)) +
                             list(range(_MIN_5xxx_VOL2, _MAX_5xxx_VOL2+1)) +
                             list(range(_MIN_6xxx_VOL1, _MAX_6xxx_VOL1+1)) +
                             list(range(_MIN_6xxx_VOL2, _MAX_6xxx_VOL2+1)) +
                             list(range(_MIN_7xxx_VOL2, _MAX_7xxx_VOL2+1)) +
                             list(range(_MIN_8xxx_VOL2, _MAX_8xxx_VOL2+1))],
        'extract_image_number': _extract_image_number,
        'extract_camera': _extract_camera,
        'get_filespec': _get_filespec,
        'parse_filespec': _parse_filespec,
        'volset_and_volume': lambda v: f'VGISS_{v[6]}xxx/{v}',
        'volume_to_index': lambda v: f'VGISS_{v[6]}xxx/{v}/{v}_index.lbl',
    }

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def image_name_valid(name: str) -> bool:
        """True if an image name is valid for this instrument."""

        name = name.upper()

        # CddddddddddR
        if len(name) != 12 or name[0] != 'C' or name[11] != 'R':
            return False
        try:
            _ = int(name[1:12])
        except ValueError:
            return False

        return True
