#!/usr/bin/env python3
"""Write the cloud-tasks files for Voyager ISS, one per planetary encounter.

The four encounters are navigated as four separate batches because their
volumes are disjoint and their images differ enough in character that an
operator usually wants to run, watch and re-run one encounter at a time.  The
encounter is spelled by the volume set: VGISS_5xxx is Jupiter, 6xxx Saturn,
7xxx Uranus and 8xxx Neptune, each holding both spacecraft's volumes.

Usage:
    make_vgiss_tasks.py --holdings-root gs://BUCKET/holdings [--output-dir DIR]
                        [--planets jupiter,saturn,uranus,neptune]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task_gen_common as common

DATASET_NAME = 'vgiss'

PLANET_VOLUME_DIGIT = {
    'jupiter': '5',
    'saturn': '6',
    'uranus': '7',
    'neptune': '8',
}
"""The digit of a VGISS volume name that names the encounter."""


def planet_volumes(planet: str) -> list[str]:
    """The Voyager volumes of one encounter, in archive order.

    Parameters:
        planet: The encounter name, a key of PLANET_VOLUME_DIGIT.

    Returns:
        The volume names of that encounter.

    Raises:
        ValueError: If the encounter is not one of the four, or the dataset
            declares no volume for it.
    """
    try:
        digit = PLANET_VOLUME_DIGIT[planet]
    except KeyError:
        valid = ', '.join(PLANET_VOLUME_DIGIT)
        raise ValueError(f'Unknown planet "{planet}"; valid names: {valid}') from None
    volumes = [name for name in common.volume_names(DATASET_NAME) if name[6] == digit]
    if not volumes:
        raise ValueError(f'No VGISS volumes for {planet}')
    return volumes


def main() -> None:
    """Write one Voyager ISS task file per selected encounter."""
    parser = argparse.ArgumentParser(
        description='Write the cloud-tasks files for Voyager ISS, one per planet'
    )
    parser.add_argument(
        '--planets',
        default=','.join(PLANET_VOLUME_DIGIT),
        metavar='NAMES',
        help="""Comma-separated encounters to write files for (default: all four:
        jupiter, saturn, uranus, neptune)""",
    )
    common.add_common_arguments(parser)
    arguments = parser.parse_args()

    planets = [name.strip().lower() for name in arguments.planets.split(',') if name.strip()]
    if not planets:
        parser.error('--planets selected no encounters')

    written = []
    for planet in planets:
        volumes = planet_volumes(planet)
        output_path = Path(arguments.output_dir) / f'{DATASET_NAME}_tasks_{planet}.json'
        print(
            f'Enumerating {planet} ({len(volumes)} volumes, '
            f'{volumes[0]} to {volumes[-1]}) into {output_path}'
        )
        count = common.write_task_file(
            output_path,
            arguments=arguments,
            dataset_name=DATASET_NAME,
            volumes=volumes,
        )
        written.append((output_path, count))

    common.report_files(written)


if __name__ == '__main__':
    main()
