# Phase 3 Code Review — Python Best Practices

Review against `.cursor/rules/python_best_practices.mdc`.

## Strengths

- **Frozen dataclasses with validation in `__post_init__`** — `InstrumentSettings`, `ProvenanceMetadata`, `Provenance`, `ConfidenceBreakdown`, `ConfidenceTermContribution` all follow the project's binding pattern.
- **Single-responsibility helpers** — `instrument_settings_from_obs`, `_resolve_git_sha`, `_resolve_spice_kernels`, `_resolve_static_data_hashes`, `log_confidence_breakdown` each do one thing and are independently testable.
- **No top-level dynamic imports.** `subprocess` is used unconditionally; `cspyce` is the one optional dependency, guarded by `try/except ImportError`. The `nav.nav_technique.nav_technique` import inside `Config._validate_registered_techniques` stays inline because the package init order otherwise deadlocks (`nav.nav_technique.nav_technique` -> `nav.support.nav_base` -> `nav.config`); the docstring documents this as a load-bearing constraint, not lazy import.
- **Module-level constants in `ALL_CAPS`** — `_STATIC_DATA_PREFIXES` carries a docstring explaining its intent.
- **Three-positional-arg ceiling honoured** — every new helper uses keyword-only arguments after the first positional or none at all.
- **No wrappers around shared helpers.** `psf_sigma_px` is called directly from every site; the previous `_psf_sigma` and `_star_psf_sigma` wrappers were deleted as soon as they were noticed.

## Notes

- **Broad `except Exception` in `provenance.py`** is restricted to cspyce-call sandboxes (kernel queries should never block a navigation) and carries `# pragma: no cover` plus inline rationale comments.
- **Saturation-mask `empty` fallback** (`np.zeros(image.shape, dtype=bool)`) is documented in the docstring; the WARNING log line is only emitted for the `calibrated_if` branch where the absence of a saturation DN is the load-bearing signal.
- **`evaluate_sigmoid_combination` return type is `Any` to express the dual-mode contract.** When `return_breakdown=False` (default) the function returns `float`; when True it returns `tuple[float, ConfidenceBreakdown]`. The docstring documents both shapes and tests cover both call sites.

## Open items

_None._
