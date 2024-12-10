import re

from .dataset_pds3 import DataSetPDS3


class DataSetGalileoSSI(DataSetPDS3):

    _MIN_0xxx_VOL = 2
    _MAX_0xxx_VOL = 23

    @staticmethod
    def _get_filespec(row):
        return row['FILE_SPECIFICATION_NAME']

    @staticmethod
    def _parse_filespec(filespec, volumes, volume, index_tab_abspath):
        parts = filespec.split('/')
        if parts[0].upper() in ('REDO', 'RAW_CAL', 'VENUS', 'EARTH', 'MOON',
                                'GASPRA', 'IDA'):
            img_name = parts[1]
        elif parts[0].upper() in ('C9', 'C10', 'C20', 'C21', 'C22', 'C30',
                                  'E4', 'E6', 'E11', 'E12', 'E14', 'E15',
                                  'E17', 'E18', 'E19', 'E26',
                                  'G1', 'G2', 'G7', 'G8', 'G28', 'G29',
                                  'I24', 'I25', 'I27', 'I31', 'I32', 'I33',
                                  'J0'):
            img_name = parts[2]
        else:
            raise ValueError(
                f'Index file "{index_tab_abspath}" contains bad '
                f'PRIMARY_FILE_SPECIFICATION "{filespec}"')
        if not img_name.endswith('.IMG'):
            return None
        return img_name

    @staticmethod
    def _extract_image_number(f):
        m = re.match(r'C(\d{10})[RS]\.\w+', f)
        if m is None:
            return None
        return int(m[1])

    @staticmethod
    def _extract_camera(f):
        return 'X'

    _DATASET_LAYOUT = {
        'all_volume_names': [f'GO_{x:04d}' for x in
                             list(range(_MIN_0xxx_VOL, _MAX_0xxx_VOL+1))],
        'extract_image_number': _extract_image_number,
        'extract_camera': _extract_camera,
        'get_filespec': _get_filespec,
        'parse_filespec': _parse_filespec,
        'volset_and_volume': lambda v: f'GO_0xxx/{v}',
        'volume_to_index': lambda v: f'GO_0xxx/{v}/{v}_index.lbl',
    }

    def __init__(self, *args, **kwargs):

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
