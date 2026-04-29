# Phase 3 Code Review — Test Suite

Review using `.cursor/skills/critique-test-suite/SKILL.md` heuristics.

## New tests added

- `tests/nav/config_files/test_body_shape_citations.py` — five tests enforcing the `_sources` schema for `config_220_body_shape.yaml` (Part 0 §74 binding).
- `tests/nav/config_files/test_config_load.py` — four tests asserting the renumbered config loads cleanly, bootstrap angles are degrees, per-instrument required Phase 3 fields are present, and the body-shape section is exposed via `Config.body_shape`.
- `tests/nav/nav_orchestrator/test_instrument_config.py` — ten tests covering the `instrument_settings_from_obs` translator: defaults, raw_dn happy path, calibrated_if happy path, missing-data_units rejection, unknown-data_units rejection, missing-noise rejection, missing-saturation_dn rejection, max_rotation_deg validation, calibrated_if NaN-marker default, end-to-end validation against every shipped 4N0 block.
- `tests/nav/nav_orchestrator/test_provenance.py` — three new tests covering `collect_provenance_metadata`: dataclass shape, hash coverage of static-data YAMLs, byte-identical output across consecutive calls.
- `tests/nav/nav_technique/test_nav_technique.py` — two new tests covering `validate_registered_confidence_specs`: shipped specs pass, bogus specs fail with a useful message.
- `tests/nav/inst/test_inst_cassini_iss.py` — one new test asserting that `_CALIB.IMG` filenames load the calibrated-IF block (regression for the original "blank" misclassification bug).
- `tests/nav/nav_model/stars/test_predicted_snr.py` — one new test asserting `psfmodel.GaussianPSF(sigma=...)` is read via per-axis `sigma_x`/`sigma_y` (regression for the AttributeError that broke every Cassini star-feature extraction).

## Strengths

- **Each test asserts one behaviour.** No `and`-joined conditions; one assertion per condition.
- **Type annotations on every test function.**
- **`pytest.raises` used as a context manager** with `match=` patterns asserting the error-message substring.
- **Synthetic-obs test fixtures continue to work without `inst_config`** because `instrument_settings_from_obs` returns legacy-equivalent defaults when `obs.inst_config is None`. Existing `_FakeObs` test stand-ins are unchanged.

## Notes

- The new tests deliberately **do not** mock `subprocess.run` for the git-SHA lookup; the helper handles non-git environments cleanly via its broad `except`, so the tests exercise the real path.
- The test for the "bogus confidence spec" creates a `NavTechnique` subclass inside the test function so the registry pollution is scoped; the `try/finally` removes the class from `NavTechnique._registry` on cleanup.
- Star-listing log assertion is delegated to live use of `capsys`; explicit per-line assertion is left for Phase 4 once the exact string format is calibrated.

## Open items

_None._
