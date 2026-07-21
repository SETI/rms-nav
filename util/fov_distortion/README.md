# FOV distortion and camera-twist analysis

Measures, per instrument and directly from star fields, the two components of the
camera-pointing error that survive the geometry oops already applies:

- **Rotational error (FOV twist)** -- a single rigid rotation about the optical
  center. A twist that is the same on every frame of an instrument is a static
  camera-frame alignment error correctable in a pointing kernel; a twist that
  scatters frame to frame is genuine per-frame attitude error that navigation
  must fit per frame. The analysis produces a per-instrument verdict and a
  `fit_camera_rotation` recommendation.
- **Lateral residual distortion** -- the field-position-dependent displacement
  left after the known distortion model is removed, fitted to the same low-order
  radial model the simulator plants so the coefficients feed the simulator
  distortion stage.

## Layout

| Module | Role |
| --- | --- |
| `decompose.py` | Pure-numpy decomposition of a per-star residual field into a rigid twist plus a radial distortion model. No spindoctor imports. |
| `aggregate.py` | Per-instrument twist-consistency verdict and rotation-fitting recommendation. No spindoctor imports. |
| `measure.py` | Per-frame measurement: load, navigate for the translation prior, centroid every predictable star, decompose. |
| `results.py` | Per-instrument aggregation over frames, including the pooled radial model. |
| `plots.py` | Per-frame residual-field figure and per-instrument twist, radial, and 2-D non-rotational distortion-map figures. |
| `config.py` / `configs/` | Per-instrument-and-camera cohort definitions (frame lists + detection parameters). |
| `run.py` | Campaign driver. |
| `tests/` | Unit tests on synthetic point clouds and images (no holdings needed). |

## Running

After `source /seti/newnav/setup.sh`, from the repository root:

```bash
python util/fov_distortion/run.py util/fov_distortion/configs/coiss_nac.yaml --workers 6
python util/fov_distortion/run.py util/fov_distortion/configs/*.yaml --workers 8
```

Artifacts default to `_work/fov_distortion/<cohort>/`:

- `<cohort>_frames.csv` -- one row per frame (twist, radial coefficients, RMS residuals).
- `<cohort>_summary.json` -- the instrument summary and recommendation.
- `figures/` -- the twist, radial, non-rotational distortion-map, and
  representative sample figures.

Pass `--report-figures` to also write the twist, radial, and sample figures into
`docs/fov_distortion_report/_figures/` for the documentation report. Pass
`--limit N` for a quick run over the first N frames of each cohort.

## Method

For each frame the tool runs star-only navigation to recover the translation
prior, then PSF-centroids every predictable catalog star and hands the
predicted / detected pairs to the decomposition. The rotation fit is done on the
collected pairs independent of the navigation `fit_camera_rotation` setting, so
an instrument that ships with rotation fitting off is still measured. The
twist and the radial term are decoupled by alternating their fits.

A frame contributes to the instrument summary only if navigation locks and
enough stars survive the centroid and residual gates. The per-frame CSV records
the status of every attempted frame.

## Tests

```bash
python -m pytest util/fov_distortion/tests/
```

The tests run on synthetic point clouds and images and need no holdings,
kernels, or star catalogs.
