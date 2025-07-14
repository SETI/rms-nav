from typing import Any, cast

from .dataset_pds3 import DataSetPDS3


class DataSetCassiniISS(DataSetPDS3):
    """Implements dataset access for Cassini ISS (Imaging Science Subsystem) data.

    This class provides specialized functionality for accessing and parsing Cassini
    ISS image data stored in PDS3 format.
    """

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116

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
        if len(range_dir) != 21 or range_dir[10] != '_':
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'PRIMARY_FILE_SPECIFICATION "{filespec}"')
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
        if f[0] not in 'NW':
            return None
        if f[11] != '_':
            return None
        img_num_str = f[1:11]
        try:
            img_num = int(img_num_str)
        except ValueError:
            return None
        return img_num

    @staticmethod
    def _extract_camera(f: str) -> str | None:
        if f[0] not in 'NW':
            return None
        return f[0]

    @staticmethod
    def _map_filename(f: str) -> str:
        assert f.endswith(('.LBL', '.IMG'))
        return f[:-4] + '_CALIB' + f[-4:]

    _DATASET_LAYOUT = {
        'all_volume_names': [f'COISS_{x:04d}' for x in
                             list(range(_MIN_1xxx_VOL, _MAX_1xxx_VOL+1)) +
                             list(range(_MIN_2xxx_VOL, _MAX_2xxx_VOL+1))],
        'extract_image_number': _extract_image_number,
        'extract_camera': _extract_camera,
        'get_filespec': _get_filespec,
        'parse_filespec': _parse_filespec,
        'volset_and_volume': lambda v: f'COISS_{v[6]}xxx/{v}',
        'volume_to_index': lambda v: f'COISS_{v[6]}xxx/{v}/{v}_index.lbl',
        'index_columns': ('FILE_SPECIFICATION_NAME',),
        'volumes_dir_name': 'calibrated',
        'map_filename_func': _map_filename,
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='DataSetCassiniISS', **kwargs)

    @staticmethod
    def image_name_valid(name: str) -> bool:
        """True if an image name is valid for this instrument."""

        name = name.upper()

        # [NW]dddddddddd[_d[d]]
        if name[0] not in 'NW':
            return False
        if 13 <= len(name) <= 14:
            if name[11] != '_':
                return False
            try:
                _ = int(name[12:])
            except ValueError:
                return False
            try:
                _ = int(name[1:11])
            except ValueError:
                return False
            return True

        if len(name) != 11:
            return False
        try:
            _ = int(name[1:])
        except ValueError:
            return False

        return True
