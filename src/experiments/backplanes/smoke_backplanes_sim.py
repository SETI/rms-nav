#!/usr/bin/env python3
import json
from pathlib import Path

from filecache import FileCache, FCPath

from nav.dataset.dataset import ImageFiles, ImageFile
from nav.obs import inst_name_to_obs_class
from backplanes.backplanes import generate_backplanes_image_files


def main():
    # Locate the local simulated model JSON in this directory
    here = Path(__file__).resolve().parent
    sim_json = here / 'simulated_body_params.json'
    assert sim_json.exists(), f'Missing simulated JSON: {sim_json}'

    # FileCache roots
    fc = FileCache()
    results_root = fc.new_path('/tmp/nav_backplanes_results')
    metadata_root = fc.new_path('/tmp/nav_backplanes_metadata')

    # Prepare image file entry using the local JSON
    image_files = ImageFiles(image_files=[
        ImageFile(
            image_file_url=FCPath(sim_json),
            label_file_url=FCPath(sim_json),
            results_path_stub=sim_json.with_suffix('').name,
        )
    ])

    # Create synthetic prior metadata with a successful status and offset from the sim JSON
    sim = json.loads(sim_json.read_text())
    dv = float(sim.get('offset_v', 0.0))
    du = float(sim.get('offset_u', 0.0))
    meta_fc = metadata_root / (sim_json.with_suffix('').name + '_metadata.json')
    with meta_fc.open('w') as f:
        json.dump({
            'status': 'success',
            'offset': [dv, du],
            'observation': {
                'image_path': sim_json.as_posix(),
                'image_name': sim_json.name,
            }
        }, f)

    # Observation class
    obs_class = inst_name_to_obs_class('SIM')

    ok, meta = generate_backplanes_image_files(
        obs_class,
        image_files,
        nav_results_root=metadata_root,
        backplane_results_root=results_root,
        write_output_files=True,
    )
    print('OK:', ok)
    print('Metadata keys:', list(meta.keys()))


if __name__ == '__main__':
    main()
