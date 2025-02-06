import re
from typing import Any, cast

from .dataset_pds3 import DataSetPDS3


class DataSetNewHorizonsLORRI(DataSetPDS3):

    @staticmethod
    def _get_filespec(row: dict[str, Any]) -> str:
        return cast(str, row['PATH_NAME']) + cast(str, row['FILE_NAME'])

    @staticmethod
    def _parse_filespec(filespec: str,
                        volumes: list[str],
                        volume: str,
                        index_tab_abspath: str) -> str | None:
        parts = filespec.split('/')
        if parts[0].upper() != 'DATA':
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'path "{filespec}"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 15 or range_dir[8] != '_':
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'path "{filespec}"')
        if not img_name.endswith('_sci.lbl'):
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
        m = re.match(r'lor_(\d{10})_0x\d{3}_sci\.\w+', f)
        if m is None:
            return None
        return int(m[1])

    @staticmethod
    def _extract_camera(f: str) -> str | None:
        return 'X'

    _DATASET_LAYOUT = {
        'all_volume_names': ['NHLALO_2001', 'NHJULO_2001',
                             'NHPCLO_2001', 'NHPELO_2001',
                             'NHKCLO_2001', 'NHKELO_2001'],
        'extract_image_number': _extract_image_number,
        'extract_camera': _extract_camera,
        'get_filespec': _get_filespec,
        'parse_filespec': _parse_filespec,
        'volset_and_volume': lambda v: f'NHxxLO_xxxx/{v}',
        'volume_to_index': lambda v: f'NHxxLO_xxxx/{v}/{v}_index.lbl',
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='DataSetNewHorizonsLORRI', **kwargs)

    @staticmethod
    def image_name_valid(name: str) -> bool:
        """True if an image name is valid for this instrument."""

        name = name.upper()

        # lor_0297859350_0x633_sci.fit
        if len(name) != 28 or not name.startswith('lor_'):
            return False
        try:
            _ = int(name[5:15])
        except ValueError:
            return False

        return True
