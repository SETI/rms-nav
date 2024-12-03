import argparse
import csv
import os
from pathlib import Path
import random
import re

from typing import Optional

from filecache import FCPath, FileCache
from pdstable import PdsTable

from .dataset_pds3 import DataSetPDS3


class DataSetCassiniISS(DataSetPDS3):

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116

    @staticmethod
    def _parse_filespec(filespec, volumes, volume, index_tab_abspath):
        parts = filespec.split('/')
        if parts[0] != 'data':
            raise ValueError(
                f'Index file {index_tab_abspath} contains bad '
                f'PRIMARY_FILE_SPECIFICATION {filespec}')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 21 or range_dir[10] != '_':
            raise ValueError(
                f'Index file {index_tab_abspath} contains bad '
                f'PRIMARY_FILE_SPECIFICATION {filespec}')
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
    def _extract_image_number(f):
        m = re.match(r'[NW](\d{8})(_\d{1,2})', f)
        if m is None:
            return None
        return m[1]

    _DATASET_LAYOUT = {
        'all_volume_nums': [f'COISS_{x:04d}' for x in
                            list(range(_MIN_1xxx_VOL, _MAX_1xxx_VOL+1)) +
                            list(range(_MIN_2xxx_VOL, _MAX_2xxx_VOL+1))],
        'is_valid_volume_name':
            lambda v: re.match(r'COISS_[12]\d{3}(/\d{8}_\d{8})', v) is not None,
        'extract_image_number': _extract_image_number,
        'parse_filespec': _parse_filespec,
        'volset_and_volume': lambda v: f'COISS_{v[6]}xxx/{v}',
        'volume_to_index': lambda v: f'COISS_{v[6]}xxx/{v}/{v}_index.lbl',
    }


    def __init__(self):
        ...

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