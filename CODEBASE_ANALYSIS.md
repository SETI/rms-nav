# Codebase analysis: rms-nav

Date: 2026-04-22 · Branch: `rf_260420_reproj` · Analysis scope: whole repo
(source tree, tests, docs, configs, templates, CI, packaging), including
staged / uncommitted files per `git status`.

Each finding cites the exact file (and, where useful, line). Findings are
tagged **Critical / High / Medium / Low**; the project rulebook referenced
throughout is `.cursor/rules/python_best_practices.mdc` (and peer files in
`.cursor/rules/`), plus `CLAUDE.md`.

## Summary

Overall the codebase is typed, documented, and well-organized along clear
architectural seams (`dataset` → `obs` → `nav_model` → `nav_technique` →
`nav_master`, and the parallel `nav.reproj` path). The recently refactored
`nav.reproj` and `nav.nav_model.rings` subpackages are exemplary: frozen
dataclasses with constructor-time validation, clean protocols, and
first-class docstrings. The rest of the tree has not caught up — there is a
steady accretion of commented-out code, unresolved `TODO`s, module-level
mutable state, broad `try/except`, and a handful of concrete rule
violations that `ruff`/`mypy` don't catch today.

Top-three priorities:

1. **Fix the packaging/typing gap**: `pyproject.toml` declares a
   `py.typed` marker under `nav`, but no such file exists, and top-level
   packages `main`, `reproj_cli`, `backplanes`, `pds4`, `util` ship
   alongside the library with generic names — a PyPI install of `rms-nav`
   will clobber anyone else's `main` package.
2. **Close the "escape hatches" from mypy strict mode**:
   `src/nav/support/correlate.py` and `correlate_old.py` each start with
   `# mypy: ignore-errors`, the `AttrDict` class is `@no_type_check`d, and
   there are multiple bare `# type: ignore` lines without error codes.
3. **Prune dead code**: `src/nav/support/flux.py` is 1174 lines of which
   only an 8-line function is live; every dataset and navigation module
   carries multi-screen commented-out blocks. The rules prohibit these
   ("NEVER include comments that merely…describe modification history"),
   and `ruff` doesn't see them because they're comments.

A secondary theme is **duplication between CLI entry points** (6 of the
`main/*.py` modules replicate the same `parse_args`/dataset-bootstrap/
global-state pattern) and **between the PDS4 per-dataset classes** (stub
`NotImplementedError` overrides replicated in three dataset modules).

---

## 1. Structure and layout

- **High — Top-level package names collide with the Python ecosystem.**
  `src/main`, `src/backplanes`, `src/pds4`, `src/reproj_cli`, and (as a
  namespace package) `src/util` are installed as top-level importable
  names alongside `nav`. `pyproject.toml` lines 186-187
  (`[tool.setuptools.packages.find] where = ["src"]`) discovers them all;
  `[project.scripts]` lines 197-210 use them as `main.nav_offset:main`,
  `reproj_cli.paths`, etc. Anyone else on PyPI (or in a larger app) using
  a top-level `main` / `util` / `backplanes` will silently have their
  imports shadowed. **Suggestion:** move them under `nav` (e.g.
  `nav._main`, `nav._cli`, `nav.backplanes`, `nav.pds4`). The
  `reproj_cli` package already has a CLAUDE.md comment ("this package is
  not part of the importable `nav` public API") that hints at this
  intent; finish the job.

- **Critical — Declared `py.typed` marker does not exist.**
  `pyproject.toml` line 190: `"nav" = ["py.typed"]`. `find
  /seti/newnav/rms-nav/src -name py.typed` returns nothing. The wheel
  will therefore install without a `py.typed` marker (setuptools silently
  skips missing data files), so downstream users of the package get
  `Skipped analyzing "nav": found module but no type hints or library
  stubs` from mypy. **Suggestion:** `touch src/nav/py.typed`.

- **High — Oversized modules.**
  The rulebook (`python_best_practices.mdc` §2) says "ALWAYS keep modules
  under 1000 lines." Over the limit today:
  - `src/main/nav_create_simulated_image.py` — 2509 lines (>2.5× limit)
  - `src/nav/reproj/bodies.py` — 1662 lines
  - `src/nav/reproj/rings.py` — 1646 lines
  - `src/main/nav_backplane_viewer.py` — 1542 lines
  - `src/nav/ui/mosaic_viewer/ring_window.py` — 1484 lines
  - `src/nav/ui/mosaic_viewer/tiled_image_widget.py` — 1293 lines
  - `src/nav/support/flux.py` — 1174 lines (but see §9 — 99% dead)
  - `src/nav/ui/manual_nav_dialog.py` — 1041 lines
  - `src/nav/nav_model/nav_model_stars.py` — 1004 lines
  `reproj/bodies.py` and `reproj/rings.py` each contain two dataclasses
  + one mosaic class and could be split into `_result.py`, `_data.py`,
  `_mosaic.py` per side.

- **Medium — `src/nav/support/__init__.py` is empty (1 line).**
  No package docstring; reader can't tell what's in `nav.support`
  without listing the directory.

- **Medium — Parallel "instrument" registries live in two places.**
  `src/nav/obs/__init__.py` lines 10-16 maintain
  `_INST_NAME_TO_OBS_CLASS_MAPPING`; `src/nav/dataset/__init__.py` lines
  12-43 maintain `_DATASET_NAME_TO_CLASS_MAPPING` and
  `_DATASET_NAME_TO_INST_NAME_MAPPING`. The module-load `assert sorted(a)
  == sorted(b)` (line 46) is a workaround for that duplication; a single
  registry with `obs_class`+`dataset_class` entries would remove the
  keep-in-sync burden (and the assert, which disappears under `python
  -O`).

- **Medium — `src/util/report_profile.py` is an orphan.**
  The `util/` directory has one 11-line script and no `__init__.py`. It
  is discovered by setuptools as a namespace package but has no entry
  point in `pyproject.toml [project.scripts]`, no tests, no docs, and is
  not imported anywhere. Either wire it up or delete it.

- **Low — Commented-out directory of imports.**
  `src/nav/support/flux.py` lines 1-17 consists of 11 lines of
  commented-out imports.

---

## 2. Best practices alignment

Reference: `.cursor/rules/python_best_practices.mdc` (cited as "PBP"
below). Most of the repo follows these rules; the issues below are the
exceptions.

- **Critical — Module-level mypy disable (PBP §5 "NEVER add global type
  exclusions").**
  `src/nav/support/correlate.py` line 1 and `src/nav/support/correlate_old.py`
  line 1 start with `# mypy: ignore-errors`. `correlate_old.py` is also
  ruff-excluded via `pyproject.toml` line 157 but not mypy-excluded — the
  module-level directive is what silences it. `correlate.py` is live code
  used by `nav_technique_correlate_all.py`. The same file line 2 carries
  `# TODO Clean up typing` — acknowledging this is debt.
  **Suggestion:** remove the directives and address errors; if
  `correlate_old.py` is truly dead, delete it (see §9).

- **High — `AttrDict` is class-level `@no_type_check`.**
  `src/nav/support/attrdict.py` line 4, 24. That class is the
  configuration data contract: every `Config.general`, `Config.bodies`
  etc. returns an `AttrDict`. Because `AttrDict` is opaque to mypy,
  attribute typos on `self._config.bodies.oversample_edgelim_limit` are
  silently accepted; the rule "NEVER use `getattr` just as a defensive
  measure" loses its teeth when every config lookup is effectively
  `getattr` through `__getattr__`. Returning `Any` from `Config.general`
  et al. (see `src/nav/config/config.py` lines 104-224) compounds this.
  **Suggestion:** model each section as a frozen dataclass validated at
  load time; mypy would then flag typos, and `Config.bodies` could be
  typed as `BodiesConfig`.

- **High — Bare `# type: ignore` without error code (PBP §5).**
  - `src/nav/obs/obs_snapshot.py` line 95
  - `src/main/nav_create_simulated_image.py` — 30+ lines (lines 1201,
    1202, 1291, 1292, 1352, 1353, 1402, 1403, 1494-1498, 1558, 1559,
    1568-1572, 1745, 1759, 1773, 1787, 1875, 1908-1912, 1921, 1942,
    1975-1978, …)
  The rule is explicit: "use a minimal line-level ignore:
  `# type: ignore[error-code]  # <brief justification>`".

- **High — `try/except AttributeError` used for control flow on
  argparse namespaces (PBP §1, CLAUDE.md "NEVER use getattr on argparse
  namespace").**
  `src/nav/config/config_helper.py` uses this pattern six times (lines
  25-28, 30-33, 62-65, 67-70, 100-103, 105-108). The three near-identical
  functions `get_backplane_results_root`, `get_nav_results_root`,
  `get_pds4_bundle_results_root` are copy/paste (~30 lines each);
  duplication plus exception-as-control-flow. **Suggestion:** one helper
  `_get_root(arg_name, env_var, cli_flag)` that returns the resolved
  string or raises.

- **High — `print()` and `sys.exit` in `src/main/*.py`.**
  `sys.exit` appearing throughout `main/` is appropriate (these are CLI
  entry points). But the `print()`-before-sys.exit error pattern
  (`nav_offset.py` lines 59-76, `nav_backplanes.py` lines 44-61,
  `nav_create_bundle.py` lines 94-103, `nav_mosaic.py` lines 138-158,
  `nav_mosaic_display.py` lines 99-101) bypasses the configured
  `pdslogger` and prints to stdout rather than stderr. **Suggestion:** a
  small helper in `nav.config` or `main/__init__.py` that prints usage
  to `sys.stderr` and exits — or use `argparse.ArgumentParser.error`.

- **High — Module-level mutable globals in CLI modules (PBP §2).**
  `src/main/nav_offset.py` lines 39-44 (`DATASET`, `DATASET_NAME`,
  `NUM_FILES_PROCESSED`, …); similar in `nav_backplanes.py` 34-36,
  `nav_create_bundle.py` 36-38, `nav_mosaic.py` 99-100. Each is mutated
  inside `parse_args()` via `global`. This makes them non-reentrant (the
  cloud-tasks variants in the same directory already hit this — they
  don't use these globals but pass state explicitly). **Suggestion:**
  wrap each driver in a class, or pass state explicitly.

- **High — Lazy module-level caches are not thread-safe.**
  Module-level mutable state exists in several modules and would race
  under multithreading:
  - `src/nav/nav_model/nav_model_stars.py` lines 90-116
    (`_STAR_CATALOG_UCAC4`, `_STAR_CATALOG_TYCHO2`, `_STAR_CATALOG_YBSC`
    with `global` inside getters).
  - `src/nav/support/misc.py` lines 122-175 (`_GIT_VERSION_CACHE`,
    `_LOCAL_HOST_NAME_CACHE`).
  - `src/nav/support/image.py` line 409 (`_FOOTPRINT_CACHE`).
  - `src/nav/annotation/annotation_text_info.py` `@functools.cache` on
    `_load_font` (line 30) is thread-safe at the CPython level, fine.
  `reproj/rings.py` and `reproj/bodies.py` already document thread-safety
  caveats in their docstrings; the star-catalog getters should do the
  same or be made reentrant.

- **High — Broad `except Exception` for control flow.**
  - `src/nav/obs/obs_snapshot.py` line 93-96 (checking whether
    `_closest_planet` exists).
  - `src/nav/nav_master/nav_master.py` line 100-104 (checking whether
    `obs.spice_kernels` exists).
  - `src/nav/support/misc.py` lines 142-143 and 173-174 (git / hostname
    lookups).
  - `src/backplanes/merge.py` lines 49-54 (naif ID lookup — OK, fallback
    documented).
  - `src/main/nav_mosaic.py` lines 233, 265, 350, 378 — broad `except
    Exception:` around each reprojection/mosaic-add call. The rulebook
    permits this only at "applications" top level; these are per-image
    handlers that should at least catch the specific oops/SPICE classes.
  - `src/nav/ui/manual_nav_dialog.py` lines 388-393 `try: btn.setAutoDefault(False);
    btn.setDefault(False); except Exception: pass` — silencing all errors
    from two setter calls.

- **Medium — Files without `encoding='utf-8'` on `open()`.**
  The config loader correctly uses `encoding='utf-8'` (`config.py` line
  52). These do not:
  - `src/nav/dataset/dataset_pds3.py` line 356 (`with open(filename) as
    csvfile:`) and line 376 (`with open(filename) as fp:`). These read
    user-provided CSV/list files.
  - `src/main/nav_create_simulated_image.py` lines 2381 (`open(filename,
    'w')`), 2395 (`open(filename)`).
  - `src/experiments/fov_twist/find_fov_twist.py` line 601 (excluded, but
    still a quality signal).
  Matters on Windows where the default is cp1252.

- **Medium — `DEFAULT_LOGGER` kept as alias for backward compatibility.**
  `src/nav/config/logger.py` line 31 — docstring says "alias for
  IMAGE_LOGGER retained for backward compatibility". CLAUDE.md
  conventions: "No backwards-compat shims unless explicitly requested."

- **Medium — Stale comment in `DataSet.ImageFile`.**
  `src/nav/dataset/dataset.py` line 30: `# Convert this to use a default
  factory` — the next line already uses `default_factory=dict`. The
  comment is a leftover.

- **Medium — Inline debug blocks behind `if False:` / `_DEBUG_*`.**
  Examples: `src/nav/support/correlate_old.py` line 144 `if False:`;
  `src/nav/nav_model/nav_model_stars.py` line 42
  `_DEBUG_STARS_MODEL_IMGDISP = False` gated import at line 786-794.
  Prefer proper debug logging.

- **Low — Docstring style drift in a few tests.**
  Several tests under `tests/nav/nav_model/` adopt a Google-style header
  with "Parameters", "Returns", "Raises" — even though they're `def
  test_x() -> None:` and take no parameters. The rulebook requires Google
  style but not boilerplate "N/A" blocks. See
  `tests/nav/nav_model/test_nav_model_rings.py` lines 1-22.

- **Low — Ruff `extend-ignore` includes `N802/N803/N806`.**
  `pyproject.toml` line 173. Justified by the oops/SPICE/scientific
  naming conventions (camelCase for API names, uppercase letters for
  math variables). This is documented in the file but worth a one-line
  comment in the rulebook if anyone asks "why is `R_min` allowed?"

---

## 3. Types and static checks

- **High — Wide use of `Any` and `dict[str, Any]` for interop data.**
  - `NavMaster._offsets: dict[str, Any] = {}  # TODO Type`
    (`src/nav/nav_master/nav_master.py` line 78).
  - All `metadata` dicts are `dict[str, Any]`.
  - `Config.general/bodies/rings/...` all return `Any` because the
    backing store is `AttrDict` which is `@no_type_check`d.
  - `RingsRenderContext.obs: Any` and `logger: Any` (`ring_render_context.py`
    lines 70, 76) — explicitly to avoid an oops import at module top.
  **Suggestion:** Protocols or TypedDicts for the common metadata shapes
  (observation, navigation, backplane) to recover static checking at
  boundaries.

- **Medium — `cast` is used where a narrower runtime check would be
  more faithful.**
  Examples: `src/nav/config/config.py` line 150, 165, 172 — `cast(list[str],
  self._config_dict.get(...).get(..., []))`. If the YAML contains a
  non-list, mypy is happy and Python raises a confusing error later.
  `src/nav/nav_technique/nav_technique_correlate_all.py` line 142 casts a
  `NavModel` to `NavModelStars` without the narrower `isinstance` check.

- **Medium — Mypy overrides silently swallow missing stubs for
  non-scientific packages as well.**
  `pyproject.toml` lines 96-146 list explicit `ignore_missing_imports`
  entries for astropy, cspyce, imgdisp, julian, oops, pdslogger,
  pdstable, pdstemplate, polymath, psfmodel, scipy, starcat — all
  scientific deps, reasonable. But no entry for `filecache` or
  `cloud_tasks` which do ship typed (they have `py.typed`); consider
  removing the ones that are no longer needed.

- **Medium — Tests import private attributes.**
  `tests/conftest.py` line 15: `if not DEFAULT_CONFIG._config_dict:`. A
  public `Config.is_loaded` property or `Config.ensure_loaded()` would
  avoid reaching into `_config_dict`.

- **Low — `Literal` usage could be broader.**
  `src/nav/reproj/bodies.py` and `rings.py` use `Literal['centric',
  'graphic', 'squashed']` for `latlon_type`. Good — but the same strings
  appear as bare `str` in `cartographic_model.py` line 51 (also
  `Literal`-typed, good) and `ui/mosaic_viewer/common.py` line 334
  (plain `str`). Consider a shared type alias.

---

## 4. Testing

**Config.** `pyproject.toml [tool.pytest.ini_options]` only sets
`pythonpath = ["src"]`. No `addopts`, no `-n auto`, no `--dist=loadfile`
— so plain `pytest` does a serial run. CLAUDE.md tells developers to
invoke `pytest -n auto --dist=loadfile` manually, and `scripts/run-all-
checks.sh` line 249 uses `-n auto` (missing the `--dist=loadfile`!) and
the CI workflow at `.github/workflows/run-tests.yml` line 91 uses both.
**Suggestion:** put both flags in `pyproject.toml` so all invocations
are consistent.

- **Critical — `.coveragerc` scopes coverage to `nav` only.**
  `.coveragerc`: `source = nav`. This excludes `backplanes/`, `main/`,
  `pds4/`, `reproj_cli/`, `util/` — i.e. all the CLI entry points, the
  backplanes generator, the PDS4 bundle generator, and the reproj CLI
  helpers. The PBP target of 90% coverage is therefore misleadingly
  easy. **Suggestion:** `source = nav, backplanes, pds4, reproj_cli`
  (leave `main/` out if you also leave entry-point scripts untested).
  The `[run]` section should also set `parallel = True` if tests ever
  run under xdist with coverage.

- **Critical — `codecov.yml` disables coverage enforcement.**
  Both `coverage.status.project.default.target` and `patch.default.target`
  are set to `0` — i.e., codecov will never block a PR on coverage drop.
  If the intent is "informational only", the file is correct; if
  enforcement is wanted, set realistic thresholds.

- **High — `pytest.raises` without message content (PBP §7
  "ALWAYS...assert on the exception message content").**
  Several tests use `pytest.raises(Exception)` without `match=`. Examples:
  - `tests/nav/support/test_misc.py` line 16: `with pytest.raises(ValueError):`
  - `tests/nav/dataset/test_dataset_pds3_cassini_iss.py` line 130.
  With `pytest.raises(ValueError, match=...)` you guard against the
  wrong ValueError being raised and accidentally "passing".

- **High — Most `tests/nav/inst/*` tests are 8-line smoke tests.**
  `tests/nav/inst/test_inst_cassini_iss.py`, `_galileo_ssi.py`,
  `_newhorizons_lorri.py`, `_voyager_iss.py` are each 8 lines and assert
  a single magic number (e.g. `obs.midtime == 196177280.54761`). They
  depend on live PDS holdings. **Suggestion:** add a coverage test for
  `get_public_metadata()`, `star_min_usable_vmag`, `star_max_usable_vmag`
  for each instrument.

- **High — Tests in `tests/nav/dataset/` depend on remote PDS index
  counts.**
  `tests/nav/dataset/test_dataset_pds3_cassini_iss.py` line 54 asserts
  `len(ret) == 8868` on a volume index. This is fragile against remote
  index updates. Documenting the frozen expectation is fine; the
  assertion should at least check a lower bound (`>= 8868`) or be
  parametrised with the expected volume cut.

- **Medium — Ring-model tests use `MagicMock` throughout.**
  `tests/nav/nav_model/test_nav_model_rings.py` is a heavy `MagicMock`
  suite (>1000 lines, 25+ tests). Good coverage of the orchestrator, but
  many tests set up mocks three pages deep — a small helper or fixture
  factory in `conftest.py` would cut duplication.

- **Medium — `tests/__init__.py` and several other `__init__.py` files
  are empty.**
  `tests/__init__.py`, `tests/nav/ui/__init__.py`, `tests/nav/reproj/__init__.py`.
  Empty namespaces are fine; but when test imports use
  `from tests.config import URL_…`, the loader depends on the empty
  `__init__.py` existing. Better: either make them proper packages with
  a docstring, or remove them and rely on pytest's implicit rootdir.

- **Medium — Test suite largely skips UI rendering.**
  `tests/nav/ui/test_common.py` starts with three layered `pytest.skip`
  guards (lines 14-37) — if PyQt6 / EGL / `nav.ui.common` import fails,
  the whole module is skipped. In CI this is fine (EGL is installed per
  `.github/workflows/run-tests.yml` line 73). For maintenance, document
  this skip path.

---

## 5. Performance and resource use

- **Medium — `reproj/bodies.py::_extract_region` has a double Python loop.**
  Lines 1592-1610: `for out_col, lon_bin in …: for out_row, lat_bin in
  …:` touching eight arrays per iteration, walking what can be a 62×1800
  grid for a full-circle Saturn mosaic. Reasonable for prototype scale,
  but the inner loop is pure numpy indexing that can be vectorized with
  fancy indexing (`out_img[...] = self._img[row_idx, col_idx]`) once the
  bin→row/col translations are precomputed.

- **Medium — `NavModelBody._create_backplane_model` creates a
  full-frame `make_extfov_zeros` per body.**
  `src/nav/nav_model/nav_model_body.py` lines 276-278, 534-539. For
  images with many bodies this is O(n·pixels). Acceptable but
  measurable.

- **Low — `np.mgrid` usage in `array_zoom` is memory-hungry.**
  `src/nav/support/image.py` lines 267-270 — produces an index array the
  size of the zoomed output. For 1024×1024×N zoom, that's a 4× memory
  increase. `numpy`'s `repeat(axis=0).repeat(axis=1)` would be lighter.

- **Low — `_find_correlated_offset` recomputes `global_min = np.min(slyce)`
  before masking (already dead code; see §9).**

No clear "hot path" bug, but several opportunities for vectorization
in the retrieval code.

---

## 6. Maintainability and extensibility

- **High — `nav.reproj` and `nav.nav_model.rings` set a quality standard
  the rest of the tree has not adopted.**
  These modules have:
  - Immutable frozen dataclasses with `__post_init__` validation.
  - Module docstrings that list the public API and its rules.
  - One-line `# noqa: A002` markers with justifications for unavoidable
    builtin shadowing (`format`).
  - Narrow type ignores with error codes.
  By contrast, `nav.nav_master`, `nav.nav_model.nav_model_stars`,
  `nav.nav_technique.nav_technique_correlate_all` are still in an older
  style (`dict[str, Any]` metadata, broad excepts, TODO-marked magic
  constants). **Suggestion:** adopt the rings/reproj pattern as the
  house style; open issues to bring remaining modules up to it.

- **High — PDS4 "not supported" stubs duplicated three times.**
  `dataset_pds3_voyager_iss.py`, `dataset_pds3_galileo_ssi.py`, and
  `dataset_pds3_newhorizons_lorri.py` each contain six methods
  (`pds4_bundle_path_for_image`, `pds4_path_stub`,
  `pds4_image_name_to_browse_lid`, `_lidvid`, `_to_data_lid`, `_lidvid`,
  `pds4_template_variables`) that all raise `NotImplementedError` —
  12+40+20+40+20+40 lines of boilerplate per file. **Suggestion:** move
  them to a `DataSetPDS4NotImplementedMixin` and mix it in where PDS4 is
  not yet supported; delete one copy from each subclass.

- **High — `pds4_template_variables` for Cassini is a 165-line dict
  assignment.**
  `src/nav/dataset/dataset_pds3_cassini_iss.py` lines 472-636. Each of
  ~60 `vars_dict['cassini:...'] = index_row.get('...', '')` lines is
  basically data. A YAML or Python mapping table plus a short loop would
  be half the code and much easier to audit for missing columns.

- **High — Several magic numbers flagged for config but never moved.**
  Grep flagged 50+ `# TODO Move to config` / `# TODO make config
  parameter` occurrences. Notable:
  - `src/nav/nav_model/nav_model_body.py` lines 22
    (`BODIES_POSITION_SLOP_FRAC = 0.05`), 503 (`0.01` — dark-side glow),
    520 (`0.05` — terminator glow).
  - `src/nav/nav_master/nav_master.py` line 541
    (`overlay[overlay < 128] = 0  # TODO Hard-coded constant`).
  - `src/nav/sim/sim_body.py` lines 293-294, 383, 412-415, 471-472.
  - `src/nav/sim/render.py` line 72 (`max_move_steps = 1  # TODO
    configurable`).

- **Medium — `DataSet` base class has duplicated PDS4 method doc
  comments.**
  `src/nav/dataset/dataset.py` lines 154-282: each `pds4_*` method
  carries an identical "We don't make PDS4 methods as @abstractmethod"
  comment (8 repetitions). Pick one, write it once in the section
  heading.

- **Medium — `ObsInst` subclasses duplicate `star_min_usable_vmag`
  return values.**
  `obs_inst_cassini_iss.py` line 82-92 has an `if self.detector == 'WAC':
  return 0.0` that falls through to `return 0.0` — identical branches.
  Voyager/Galileo/NH-LORRI all hardcode `return 0.0` and `return 10 #
  TODO`. This data should live in the instrument config (see
  `_inst_config['star_psf_sizes']` already there).

- **Medium — `CONTRIBUTING.md` and `developer_guide_best_practices.rst`
  are stale vs the real rulebook.**
  - `CONTRIBUTING.md` line 27 says `pip install -r requirements.txt` and
    line 29-33 mentions `pre-commit install` — but the repo has no
    `.pre-commit-config.yaml`.
  - Same file line 56: commit message format `Add feature:` conflicts
    with Conventional Commits (CLAUDE.md and `.cursor/rules/git_workflow.mdc`).
  - `docs/developer_guide_best_practices.rst` lines 8-28 summarises "use
    PEP 8 / type hints / pytest / ensure backward compatibility" — the
    "ensure backward compatibility" advice **directly contradicts**
    CLAUDE.md ("No backwards-compat shims unless explicitly requested").
  - `README.md` tells users `pip install -r requirements.txt`, but
    `requirements.txt` is just `-e .\n-e .[dev]` — that works but
    surprises anyone trying to install for production. The dependency_management.mdc
    rule actually says `requirements.txt` should contain only `-e .` for
    backwards compat.

- **Low — `__init__.py` signatures inconsistent.**
  `nav/ui/__init__.py` does not exist (no such file; `src/nav/ui/` has
  no `__init__.py` — therefore the package is implicit-namespace). The
  sibling `nav/ui/mosaic_viewer/__init__.py` does exist with a proper
  docstring. Either `nav.ui` is a regular package or it isn't — make it
  consistent (add an `__init__.py` with a docstring).

---

## 7. Security and robustness

- **High — SPICE error matching by string.**
  `src/nav/navigate_image_files.py` lines 79-95:
  `if 'SPICE(CKINSUFFDATA)' in str(e) or 'SPICE(SPKINSUFFDATA)' in str(e)
   or 'SPICE(NOFRAMECONNECT)' in str(e)`. Fragile against SPICE message
  changes and locale. Catch the dedicated `SpiceyPyError` subclasses or
  use an error-code attribute.

- **Medium — `_cisscal_solar_flux()` (commented) and other dead code
  uses `allow_pickle=True`-flavored patterns.**
  Not currently live, but if re-enabled take care: `numpy.load` with
  `allow_pickle=True` is a deserialization pitfall (arbitrary code
  execution). The live `src/nav/reproj/_serialization.py` already uses
  `np.load(local_path, allow_pickle=False)` (line 304) — good; document
  that as the house rule somewhere.

- **Medium — `subprocess.check_output` in `current_git_version()`.**
  `src/nav/support/misc.py` line 138. List form, fine; wraps a broad
  `except Exception`. Acceptable but the lookup runs in every logged
  batch — the cache amortises it, so impact is low.

- **Medium — User-supplied paths opened with Python default
  permissions.**
  `dataset_pds3.py` CSV/list file opens don't set `encoding=` (see §2)
  and don't validate that the path is within a trusted root — but these
  are CLI-provided, and the PBP rule ("File operations guard against
  path traversal") applies at library boundaries, not user commands.
  Low risk.

- **Low — `hash()` used for RNG seeds.**
  `src/nav/nav_model/nav_model_stars.py` / `sim_body.py` both seed
  `np.random.RandomState` from `hash(...)` which is PYTHONHASHSEED-
  dependent and therefore non-reproducible across runs unless
  `PYTHONHASHSEED` is fixed. `sim_body.py` line 148-151 correctly uses
  an explicit `seed` when provided and notes "hash-based seed" as
  fallback; make that documented in the public docstring so callers
  know to pass `seed=` for reproducibility.

---

## 8. Dependencies and tooling

- **High — CI publishing workflows are inconsistent with each other
  and with the project rulebook.**
  - `.github/workflows/publish_to_pypi.yml` uses `actions/checkout@v4`,
    `actions/setup-python@v5`; while `run-tests.yml` uses `@v6`
    (`environment_best_practices.mdc` line 17 pins to a major tag — so
    different major tags across workflows is a minor inconsistency).
  - The PyPI publish workflow uses a `PYPI_API_TOKEN` secret rather than
    PEP 740 Trusted Publishers (also called for in the rulebook under
    "Publishing workflow … Trusted Publishers or token auth"). Token
    auth works but Trusted Publishers is recommended.
  - No `pip audit` job — `dependency_management.mdc` §5 says "Run `pip
    audit` in CI".
  - No Dependabot or Renovate configuration (nothing under `.github/`
    matches); also explicitly called out by the rulebook.

- **Medium — Separate `.coveragerc` when `pyproject.toml` supports
  `[tool.coverage.run]`.**
  `dependency_management.mdc` §6 explicitly says "Do NOT create separate
  config files (`.coveragerc`…) when the tool supports `pyproject.toml`".

- **Medium — `ruff.extend-exclude` hides `src/nav/support/correlate_old.py`.**
  `pyproject.toml` line 156-157. Perfectly reasonable IF the file is
  going to be deleted soon — but the file has been there long enough to
  accumulate substantial content, and the `# mypy: ignore-errors` at
  line 1 suggests it is NOT being actively maintained to standard. See
  §9.

- **Medium — `requirements.txt` shape is non-standard.**
  `-e .\n-e .[dev]`. `dependency_management.mdc` §1 says "If a
  `requirements.txt` is kept, it should contain only `-e .` for backward
  compatibility." Having `-e .[dev]` in `requirements.txt` means
  anyone running `pip install -r requirements.txt` installs dev deps
  — surprising. Suggest just `-e .[dev]` and a comment pointing users
  at `pip install -e ".[dev]"` as the preferred command.

- **Medium — Python matrix in CI is broader than `pyproject.toml`
  "Development Status".**
  `pyproject.toml` line 35 says `Development Status :: 2 - Pre-Alpha`
  while CI tests across Python 3.10-3.13 and the README carries PyPI
  badges (Release, Downloads). If you're publishing to PyPI, `2 -
  Pre-Alpha` is unusual.

- **Medium — `build` and `.venv` directories committed-adjacent.**
  Not in git, but the presence of `build/`, `.venv/`, `.coverage`,
  `.code_planner_cache.db`, `_work/` in the repo root means someone's
  local tooling leaks through. `.gitignore` covers them, but
  `.code_planner_cache.db` is 200 KB, tracked? let's not speculate; a
  pre-commit that refuses to commit `.db` / `.coverage` would help.

- **Low — `scripts/run-all-checks.sh` pytest command omits
  `--dist=loadfile`.**
  Line 249: `python -m pytest tests -q --cov -n auto`. CLAUDE.md calls
  this out: "pytest-xdist must run with `--dist=loadfile`; default
  scheduling crashes PyQt6 workers". Add the flag or point the script
  at the CI command.

- **Low — `pyinstrument` and `pytest-profiling` in dev deps but no
  docs on how to use them.**
  Not a correctness issue, but "why is this installed" is a question
  that comes up.

---

## 9. Technical debt and risk

- **Critical — `src/nav/support/flux.py` is ~99% commented-out code
  (1174 lines total; 8 lines live).**
  Only the 8-line `clean_sclass` function is live and imported; all
  CISSCAL photometric / filter / stellar-spectrum / DN-from-spectrum
  machinery is dead commented-out code. Per PBP §4 "ALWAYS preserve
  existing comments that are still accurate and relevant. Remove or
  update stale comments." **Suggestion:** move `clean_sclass` to
  `nav.support.misc` and delete `flux.py`. If the CISSCAL logic is
  intended for revival, it belongs in a branch or a private repo, not
  in `src/`.

- **Critical — `src/nav/support/correlate_old.py` is effectively dead
  code.**
  763 lines; excluded from ruff, silenced by `# mypy: ignore-errors`,
  documented in CLAUDE.md as "don't treat as authoritative". Nothing in
  `src/` imports it (`grep -rn 'correlate_old' src/` only matches the
  file itself). Delete, or move out of the PyPI-shipped tree.

- **High — `_work/FINDINGS.md` is an investigation diary committed
  at the repo root.**
  Not tracked (per `git status` it's untracked), but sitting in the repo
  root where it will surface for any `find`/globber. If the findings
  drove real fixes, they belong in a PR description or ADR, not in a
  permanent root-level file.

- **High — 80+ `TODO`/`FIXME`.**
  Grep density (non-experiments, non-correlate_old): the heaviest are
  `nav_model_stars.py` (8), `sim_body.py` (6), `nav_master.py` (6),
  `dataset_pds3.py` (6), `support/correlate.py` (4). CLAUDE.md says
  TODOs are fine in principle, but many of these are ~1 year old.
  Examples of still-relevant items worth triaging:
  - `nav_master.py` line 78, 103, 370, 434, 486.
  - `nav_model_stars.py` line 310 (unexplained `copy.deepcopy`), line
    709 (instrument-specific code path placeholder), line 741 (post-
    offset value in metadata).
  - `nav_technique_correlate_all.py` line 133, 335.
  - `sim_body.py` lines 383, 412-415 (crater math magic numbers).

- **High — Large blocks of commented-out code throughout.**
  Representative list (excluding `flux.py`, `correlate_old.py`):
  - `nav_model_body.py` lines 120-130, 162-190, 407-490, 513-519 — 130+
    commented lines in one file.
  - `dataset_pds3.py` lines 221-225, 278-304, 312-316, 396-450,
    528-537, 625-650, 699-705, 779-818 — ~200 commented lines.
  - `nav_master.py` lines 194-196, 405-410, 473-476, 545-554.
  - `obs_inst_voyager_iss.py`, `_galileo_ssi.py`, `_newhorizons_lorri.py`
    lines 99-122 etc. — commented SCLK fields.
  - `support/image.py` lines 210-253 and 854-899.
  - `support/misc.py` lines 193-198 (AWS EC2 block).

- **Medium — Mild deprecation risk: NumPy 2.x cast semantics.**
  `pyproject.toml` requires `numpy>=2.2.0`. The code uses `np.asarray(…,
  float)`, `np.array`, and masked-array operations heavily. A targeted
  pass to ensure that masked-array fill values and dtype promotion
  behaviours aren't shifting under NumPy 2.3+ would be worth a ticket.

- **Medium — Python version / platform assumptions.**
  - `src/nav/config_files/config_01_general.yaml` line 13:
    `truetype_font_dir: /usr/share/fonts/truetype  # TODO`. Fails on
    macOS (`/System/Library/Fonts/`) and Windows. The `# TODO` is
    honest but long-standing.
  - `subprocess.check_output(['git', 'describe', ...])` assumes `git`
    on PATH. Already wrapped in try/except, low risk but platform-
    dependent.

- **Low — Empty planet-ring config files.**
  `src/nav/config_files/config_20_jupiter_rings.yaml`,
  `config_22_uranus_rings.yaml`, `config_23_neptune_rings.yaml` are
  0 bytes — documented in the YAML filename convention but a reader
  can't tell "empty on purpose" from "accidentally truncated".

- **Low — Empty PDS4 template `.lblx` files.**
  `src/pds4/templates/cassini_iss_saturn_1.0/global_index_bodies.lblx`
  and `global_index_rings.lblx` are 0 bytes, but
  `src/pds4/collections.py` lines 292-319 look them up with `if
  template.exists()`. If these template files should exist but not be
  empty, it's a bug; if not, omit them.

- **Low — One commented-out test.**
  `tests/nav/dataset/test_dataset_pds3_voyager_iss.py` lines 93-104:
  `# def test_voyager_iss_camera(): # TODO: Figure this out`.

---

## 10. Packaging and distribution

- **Critical — `py.typed` marker missing from the installed package.**
  (Repeated from §1 because it also applies here.) The wheel installs
  without `nav/py.typed`; downstream users lose typed support.

- **High — Top-level packages `main`, `util`, `backplanes`, `pds4`,
  `reproj_cli` ship generic names on PyPI.**
  (Repeated from §1.) Rename into `nav.*`.

- **High — `[project.classifiers]` says Pre-Alpha while the package is
  on PyPI with versioned releases.**
  `pyproject.toml` line 35. If 0.x releases are in flight, `Development
  Status :: 3 - Alpha` or `:: 4 - Beta` is a better signal to users.

- **High — Version string: single source of truth is setuptools-scm,
  but `src/nav/_version.py` is `write_to=...` (tracked at commit, not
  generated).**
  `pyproject.toml` line 194. When `setuptools_scm` is in `write_to`
  mode the file should be in `.gitignore` (it isn't — `_version.py` is
  missing from the current `.gitignore` listing). If a stale
  `_version.py` is committed, editable installs will prefer it over
  `setuptools_scm`'s dynamically computed version.

- **Medium — Missing classifiers.**
  No `Intended Audience`, no `Topic :: Scientific/Engineering :: Image
  Processing`, no `Programming Language :: Python :: 3 :: Only`. Low
  priority.

- **Medium — `[project].description` is fine but `maintainers` has one
  entry (Robert French) and no `authors` list.**
  Common convention is to populate both. Not critical.

- **Medium — Test distribution.**
  The `tests/` tree is not in `src/` and not declared in
  `[tool.setuptools.packages.find]` (which has `where = ["src"]`), so
  tests are never installed into the wheel — good. `conftest.py` still
  does `from nav.config import DEFAULT_CONFIG` which works via `pythonpath
  = ["src"]` in `pyproject.toml` line 149-151.

- **Medium — README install instructions drift from the actual
  preferred command.**
  `README.md` line 78: `pip install -r requirements.txt`. CLAUDE.md §
  "Install": `pip install -e ".[dev]"`. They install the same thing by
  different paths; pick one and document it.

- **Medium — No `Changelog`/`CHANGES.md`.**
  The PR template (`.github/pull_request_template.md` line 43) has a
  checkbox "CHANGES.md updated (if user-facing change)" — but the file
  does not exist. Either add one and start tracking, or remove the
  checkbox.

- **Low — `codecov.yml` target: 0 means codecov never fails PRs.**
  See §4.

---

## Audit coverage and caveats

- **Source tree (non-experiments):** Read in full for
  `src/nav/{config,support,obs,dataset,reproj,nav_master,nav_technique,
  annotation,sim,nav_model/rings,ui/common.py,ui/manual_nav_dialog.py,
  ui/mosaic_viewer/common.py,ui/mosaic_viewer/photometric_display.py,
  ui/mosaic_viewer/matplotlib_qt.py,ui/mosaic_viewer/__init__.py}`,
  `src/reproj_cli`, `src/backplanes`, `src/pds4`, `src/main` (most
  drivers), `src/util`. Three very large UI files (`mosaic_viewer/
  ring_window.py` 1484L, `body_window.py` 976L,
  `tiled_image_widget.py` 1293L) and two very large `main/` files
  (`nav_create_simulated_image.py` 2509L,
  `nav_backplane_viewer.py` 1542L) were sampled for specific issues
  (bare `# type: ignore`, imports, entry points, broad excepts) but
  not walked line-by-line. Given the density of bare `# type: ignore`
  markers already found in `nav_create_simulated_image.py`, expect
  similar issues in the other large UI/main files.
- **Experiments (`src/experiments/`) and `src/nav/support/correlate_old.py`:**
  not audited for quality (both excluded from ruff/mypy per
  `pyproject.toml`).
- **Tests:** representative sample (config, conftest, dataset tests,
  instrument smoke tests, snapshot tests, support helpers, UI tests,
  rings tests, reproj tests) — sufficient to assess style and coverage
  scope.
- **Docs:** index, api references, developer guide, best practices,
  user guide (rings/bodies) reviewed. Not every `.rst` file was walked.
- **Configs:** all 19 YAMLs enumerated; contents of config_01, config_30
  (cassini) reviewed. Ring data files (config_21_saturn_rings.yaml 33 KB)
  not walked.
- **PDS4 templates:** file inventory and sizes verified; contents of
  `.lblx` templates not reviewed (XML/PDS4 schema).

---

## Recommended priorities

1. **Packaging & typing (Critical):** create `src/nav/py.typed`; rename
   top-level `main`, `util`, `backplanes`, `pds4`, `reproj_cli` packages
   under `nav` (or mark them as non-public via `{nav}._*`); fix
   `setuptools_scm` `write_to` / `.gitignore` interaction.
2. **Dead code sweep (Critical):** delete
   `src/nav/support/correlate_old.py`; strip `src/nav/support/flux.py`
   to the one live function and delete the file; remove commented-out
   blocks in `nav_model_body.py`, `dataset_pds3.py`, `nav_master.py`,
   and the obs-instrument metadata stubs; triage the 80+ TODOs.
3. **Coverage & CI (High):** extend `.coveragerc` to include
   `backplanes`, `pds4`, `reproj_cli`; set non-zero codecov thresholds;
   add `pip audit` (and Dependabot) to CI; align `scripts/run-all-
   checks.sh` pytest command with `--dist=loadfile`.
4. **Type hygiene (High):** remove module-level `# mypy: ignore-errors`
   in `correlate.py`; replace `@no_type_check` on `AttrDict` with typed
   config dataclasses; eliminate bare `# type: ignore` in
   `nav_create_simulated_image.py` and `obs_snapshot.py`.
5. **Deduplicate CLI bootstrap (High):** extract the
   `parse_dataset_arg/parse_args/set globals/setup_logging` boilerplate
   shared by `nav_offset.py`, `nav_backplanes.py`, `nav_create_bundle.py`,
   `nav_mosaic.py`, and their `_cloud_tasks.py` siblings into one
   helper; replace the three copy/paste path-getters in
   `config_helper.py` with one parameterised function; collapse the
   repeated PDS4 `NotImplementedError` stubs into a mixin.
6. **Rulebook consistency (Medium):** update `CONTRIBUTING.md` to match
   `.cursor/rules/` (Conventional Commits, `pip install -e ".[dev]"`,
   drop pre-commit mention or add a hook config); rewrite
   `docs/developer_guide_best_practices.rst` to point at the `.cursor/
   rules/` files rather than restating outdated guidance.
7. **Module size (Medium):** split `reproj/bodies.py` and
   `reproj/rings.py` into `_result`, `_data`, and `_mosaic` modules;
   split `nav_model_stars.py` and `nav_create_simulated_image.py`.
8. **Thread-safety documentation (Medium):** document the star-catalog
   getters as not thread-safe; consider protecting the caches with
   `threading.Lock`; add a similar note to `image.py::_FOOTPRINT_CACHE`.
9. **Config portability (Low):** turn the hard-coded
   `truetype_font_dir: /usr/share/fonts/truetype` into a platform-
   dependent default or an env-var with a documented fallback.
10. **Docs polish (Low):** remove `:undoc-members:` from
    `api_reference/api_*.rst` so `autodoc` stops silently shipping
    undocumented members; add the `developer_guide_nav_master.rst`
    cross-ref; empty planet-ring YAMLs should carry a one-line comment
    explaining the convention.
