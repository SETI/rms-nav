"""Stamp campaign sidecars with their shortlist row and report grid coverage.

Run after manual-navigation sessions.  For every sidecar in the campaign
directory whose ``image_id`` matches a shortlist row, appends the row's
stratification (region role, resolution band, face, opening regime, catalog
edge-rms range) to the sidecar's ``notes`` if not already present.  Then
prints the region x resolution grid with done / remaining counts so the
operator can see which cells still lack truth.
"""

import csv
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1] / 'tests/integration/image_library/campaigns/ring_2026'
STAMP = 'ring-curation shortlist:'


def main() -> int:
    with (HERE / 'shortlist.csv').open() as fp:
        rows = {r['image_id']: r for r in csv.DictReader(fp)}
    done: set[str] = set()
    for path in sorted(CAMPAIGN.glob('*.yaml')):
        d = yaml.safe_load(path.read_text())
        image_id = str(d.get('image_id', ''))
        base = image_id.replace('_CALIB', '')
        row = rows.get(base)
        if row is None:
            print(f'  (not on the shortlist: {image_id})')
            continue
        done.add(base)
        gt = d.get('ground_truth')
        stamp = (
            f'{STAMP} {row["role"]}, res {row["res_kmpx"]} km/px '
            f'({row["res_band"]}), {row["face"]} face, opening {row["opening_bin"]} '
            f'({row["obs_opening_deg"]} deg), {row["background"]} background, '
            f'regions {row["regions"]}, '
            f'catalog edge rms {row["edge_rms_min_km"]}-{row["edge_rms_max_km"]} km.'
        )
        if isinstance(gt, dict):
            notes = str(gt.get('notes') or '')
            if STAMP not in notes:
                gt['notes'] = (notes.rstrip() + '\n' if notes.strip() else '') + stamp + '\n'
                path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
                print(f'  stamped {path.name}')

    primaries = [r for r in rows.values() if r['role'].startswith('primary:')]
    print(
        f'\n{len(done)} shortlist images have campaign sidecars; '
        f'{sum(1 for r in primaries if r["image_id"] in done)} of {len(primaries)} '
        f'primary cells satisfied\n'
    )
    regions = sorted({r['role'].split(':', 1)[1] for r in primaries})
    bands = ['0-5', '5-25', '25-100', '100-300', '300-1000', '1000-3000']
    print(f'  {"":16s}' + ''.join(f'{b:>11s}' for b in bands))
    for region in regions:
        line = f'  {region:16s}'
        for band in bands:
            cell = [
                r
                for r in rows.values()
                if region in r['regions'].split('+') and r['res_band'] == band
            ]
            hit = any(r['image_id'] in done for r in cell)
            line += f'{"done" if hit else ("-" if not cell else "open"):>11s}'
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
