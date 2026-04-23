# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RMS-NAV is a spacecraft image navigation system that determines precise pointing
offsets for images from Cassini ISS, Voyager ISS, Galileo SSI, and New Horizons
LORRI by comparing observed images against theoretical models generated from SPICE
kernels. It also generates PDS4 bundles, per-pixel backplanes, and body/ring
mosaics. Python 3.10+, distributed as `rms-nav` on PyPI.

## Commands

All development commands assume the project virtualenv is active.

### Install

```bash
pip install -e ".[dev]"          # runtime + dev (ruff, mypy, pytest, docs)
```

### Test

```bash
pytest                            # full suite
pytest -n auto --dist=loadfile    # parallel (matches CI; loadfile avoids PyQt6 worker crashes)
pytest tests/nav/reproj/test_bodies.py            # one file
pytest tests/nav/reproj/test_bodies.py::test_foo  # one test
pytest --cov                      # with coverage
```

Tests that reach external holdings require env vars (set by CI in
`.github/workflows/run-tests.yml`):

```bash
export PDS3_HOLDINGS_DIR=https://pds-rings.seti.org/holdings
export PDS4_HOLDINGS_DIR=https://pds-rings.seti.org/pds4
export OOPS_RESOURCES=https://storage.googleapis.com/rms-node-oops-resources
export UCAC4_PATH=https://storage.googleapis.com/rms-node-star-catalogs/UCAC4
export YBSC_PATH=https://storage.googleapis.com/rms-node-star-catalogs/YBSC
```

SPICE kernels must be available at `$SPICE_PATH` for any real navigation run.

### Lint, types, docs

```bash
ruff check src tests
ruff format --check src tests
mypy src tests                    # strict mode is on; MYPYPATH=src from pyproject
sphinx-build -W -b html docs docs/_build
pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md
./scripts/run-all-checks.sh       # wraps all of the above (parallel by default)
```

### CLI entry points (from `pyproject.toml [project.scripts]`)

`nav_offset`, `nav_backplanes`, `nav_create_bundle`, `nav_mosaic_rings`,
`nav_mosaic_body`, `nav_mosaic_display_rings`, `nav_mosaic_display_body`,
`nav_backplane_viewer`, `nav_create_simulated_image`, plus `_cloud_tasks`
variants of the batch drivers. All dispatch modules live in `src/main/`.

## Architecture

### Pipeline (single image)

1. `DataSet` (`src/nav/dataset/`) enumerates images and builds `ImageFile`s
   from CLI args, index files, or holdings dirs. One subclass per
   mission/holdings layout (`DataSetPDS3CassiniISSSaturn`, etc.); registry
   + factory in `src/nav/dataset/__init__.py`.
2. `ObsSnapshotInst` (`src/nav/obs/`) subclasses read an image file via
   `.from_file(path)` and wrap an oops `Observation`. Per-instrument subclass
   (`ObsCassiniISS`, `ObsVoyagerISS`, ...); registry in `src/nav/obs/__init__.py`.
3. `NavMaster` (`src/nav/nav_master/nav_master.py`) is the orchestrator. It
   builds the requested `NavModel`s, combines them, and runs the requested
   `NavTechnique`s to produce a final `(dv, du)` offset. Offset convention:
   predicted position `(v, u)` means actual position is `(v + dv, u + du)`.
4. `NavModel` (`src/nav/nav_model/`) generates synthetic images of what
   *should* be at each pixel: `NavModelStars`, `NavModelBody` (one per body
   in FOV), `NavModelRings`, `NavModelTitan`, `NavModelCombined`.
5. `NavTechnique` (`src/nav/nav_technique/`) matches model to data.
   `NavTechniqueCorrelateAll` (cross-correlation over all models) is the
   default; `NavTechniqueManual`, `NavTechniqueTitan` are specialized.
6. `navigate_image_files` (`src/nav/navigate_image_files.py`) is the top-level
   function called by `nav_offset`; it writes JSON metadata and a PNG preview
   under `nav_results_root`.

Every class in steps 2-5 inherits from `NavBase` (`src/nav/support/nav_base.py`)
for shared `config` + `logger` (`pdslogger.PdsLogger`).

### Reprojection / mosaicing (`src/nav/reproj/`)

Parallel two-class design: `BodyMosaic` (lat/lon grid) and `RingMosaic`
(radius/longitude sparse grid). Each exposes `.reproject(obs, ...)` returning
a `*ReprojResult` dataclass, and `.add(result)` to accumulate into a mosaic.
`create_cartographic_model()` projects a body mosaic back onto image
coordinates for correlation-based navigation.

Thread safety: `RingMosaic.reproject()` mutates oops global precision via
`_reduced_oops_precision`; `BodyMosaic.reproject()` and
`create_cartographic_model()` build `Backplane` objects from `obs`. None is
safe for concurrent use on the same `obs` — give each thread its own.

`src/reproj_cli/` holds CLI-only helpers (arg parsing, factories, offset
loading, path conventions) shared by `nav_mosaic` and `nav_mosaic_display`;
this package is not part of the importable `nav` public API.

### Config

`Config` (`src/nav/config/config.py`) loads YAML from
`src/nav/config_files/` in filename order (the `config_NN_*.yaml` prefix is
the merge order). `DEFAULT_CONFIG` is the module-level singleton. User-level
overrides come from `nav_default_config.yaml` at the project root via
`load_default_and_user_config`. Sections are exposed as `AttrDict` properties
(`config.general`, `config.offset`, `config.bodies`, `config.rings`,
`config.stars`, `config.titan`, `config.bootstrap`, `config.backplanes`,
`config.pds4`, `config.environment`).

`config.environment` holds `pds3_holdings_root`, `nav_results_root`,
`backplane_results_root`, `bundle_results_root` — usually also settable via
env vars (`PDS3_HOLDINGS_DIR`, etc.) or CLI flags.

### Downstream stages

- `src/backplanes/` generates per-pixel geometry products (lon, lat, angles,
  etc.) from a navigated image. Driver: `nav_backplanes`.
- `src/pds4/` assembles PDS4 bundles (labels via `pdstemplate`, browse
  products, collections). Driver: `nav_create_bundle`. Per-dataset PDS4
  hooks (template dir, LID/LIDVID builders, template variables) live on the
  `DataSet` subclass — see `DataSetPDS3CassiniISS` for the reference impl.
- `src/nav/sim/` synthesizes test images for a given geometry; driven by
  `nav_create_simulated_image`.
- `src/nav/ui/` holds PyQt6 widgets (manual nav dialog, mosaic viewer).
  PyQt6 imports are kept out of core nav paths.

### Adding a new instrument / dataset

1. Subclass `ObsSnapshotInst` in `src/nav/obs/`, register in
   `src/nav/obs/__init__.py`.
2. Subclass `DataSet` (or `DataSetPDS3`) in `src/nav/dataset/`, register in
   `src/nav/dataset/__init__.py`. Implement `_img_name_valid`, the
   `yield_image_files_*` methods, `add_selection_arguments`, and — for PDS4
   support — the `pds4_*` template/LID methods.
3. Add instrument-specific config in `src/nav/config_files/config_3N_inst_*.yaml`.

## Project conventions

These come from `.cursor/rules/` and apply to all new code:

- **Line length**: 100. Ruff format with single-quote strings.
- **Ruff select**: `E, F, W, I, UP, B, SIM, C4, A, N, PT, RUF` — keep this
  set; do not disable categories wholesale. `N802/N803/N806` are ignored
  project-wide (oops/SPICE APIs use camelCase).
- **Mypy**: `strict = true`. No module-level `# type: ignore` without a
  specific error code; no broad `exclude` additions. The PyQt6 mosaic_viewer
  modules are the exception (scoped overrides already in `pyproject.toml`).
- **Docstrings**: Google style with `Parameters:` (not `Args:`), wrapped at
  90 chars. Every module/class/function gets one. Do NOT use unicode smart
  quotes, em-dashes, or arrows inside `.py` files (OK in `.rst`/`.md`).
- **Function signatures**: at most 3 positional params; the rest keyword-only
  after `*`. Prefer RORO (dataclass in, dataclass out) over long tuples.
- **Imports**: three alphabetical groups (stdlib / third-party / local) at
  the top of the file. Inline imports only for heavy optional deps (GUI).
- **Modules**: keep under 1000 lines; split into a package if larger.
- **No backwards-compat shims** unless explicitly requested.
- **Logging**: use `pdslogger` via `NavBase.logger`; never bare `print()` in
  library code.
- **Tests**: pytest, `-n auto`, every test independent, type annotations on
  test functions, one assert per condition (no `and`), use `pytest.raises`
  as a context manager and assert on message content. Target 90% line
  coverage across the full suite.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`,
  `build:`, `test:`, `perf:`, `ci:`, `chore:`, `style:`). Subject imperative,
  <=50 chars, no trailing period. One logical change per commit. PRs
  squash-merge to `main`.

## Gotchas

- Editable installs + mypy: export `SETUPTOOLS_ENABLE_FEATURES=legacy-editable`
  if mypy can't find the `nav` package.
- `pytest-xdist` must run with `--dist=loadfile`; default scheduling crashes
  PyQt6 workers when tests from one file split across processes.
- `src/nav/support/correlate_old.py` and `src/experiments/` are excluded
  from ruff/mypy — don't treat them as authoritative.
- `main/*.py` scripts adjust `sys.path` before importing `nav`, so imports
  there are intentionally non-top (ruff `E402` is silenced for that path).
