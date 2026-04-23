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

## 10. Packaging and distribution

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
