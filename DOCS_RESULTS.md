# Documentation migration results

This file records issues from the developer-guide relocation
(`DOCS_PROMPT_1_MIGRATE.md`) that could not be resolved within the scope of
the migration step, plus one discrepancy between the migration table and the
live file set.

## Migration table lists 21 files but only 20 exist

- **Where**: `DOCS_PROMPT_1_MIGRATE.md` migration table; repo tree
  `docs/developer_guide*.rst`.
- **What**: The migration table enumerates 21 files, including
  `docs/developer_guide_image_library.rst`. That file does not exist in the
  repository. `git ls-files 'docs/developer_guide*.rst'` returns exactly 20
  paths, none of which is the image-library page.
- **Action taken**: Moved the 20 files that exist into `docs/dev_guide/`,
  renaming each `developer_guide<suffix>.rst` to `dev_guide<suffix>.rst` via
  `git mv`. Did not attempt to move the absent image-library file. The
  image-library page is authored separately at
  `docs/dev_guide/dev_guide_image_library.rst`.
- **Follow-up needed**: None for the migration itself. The image-library page
  is created in the content-authoring step and added to the developer-guide
  toctree.

## Pre-existing autodoc warning blocks the zero-warning build

- **Where**: `src/nav/nav_technique/nav_technique_body_terminator.py`,
  class docstring for `BodyTerminatorNav` (reported as docstring offset 7).
- **What**: `sphinx-build -W -b html docs docs/_build` reports exactly one
  warning: `ERROR: Unexpected indentation. [docutils]` raised by autodoc when
  rendering the `BodyTerminatorNav` class docstring. This warning is present
  on the tree before the migration and after it; the enumerated list in that
  docstring parses cleanly in standalone docutils, so the warning is a
  Sphinx autodoc rendering quirk rather than a documentation cross-reference.
  It is unrelated to the relocation of the developer-guide files.
- **Action taken**: Left the Python source unchanged, since the migration step
  is scoped to relocating files and retargeting cross-references and forbids
  content edits to the moved files. Confirmed the renamed layout builds at
  least as cleanly as the pre-rename layout (identical single warning before
  and after). Without `-W`, the HTML build succeeds (exit 0) with this one
  warning, and `sphinx-build -b linkcheck` reports zero broken internal links.
- **Resolution**: Fixed. The `BodyTerminatorNav` class docstring now keeps only
  short single-line entries under `Class attributes:` and moves the multi-line
  rationale into a following prose paragraph (matching `BodyLimbNav`), so
  `sphinx-build -W` exits clean.
- **Follow-up needed**: None.

## Linkcheck reports one broken external link

- **Where**: `docs/index.rst` line 146 (external project link).
- **What**: `sphinx-build -b linkcheck` flags
  `https://github.com/SETI/rms-cloud_tasks` as a 404. This is an external URL,
  not an internal cross-reference, and is unrelated to the developer-guide
  relocation.
- **Action taken**: Resolved. The URL had an underscore in the slug; corrected
  to `https://github.com/SETI/rms-cloud-tasks` in `README.md` (the source that
  MyST includes into `index.rst`).
- **Follow-up needed**: None.

# Foundation pass (DOCS_PROMPT_2_WRITE.md) — stub skeleton

The entries below were added during the FOUNDATION (stub skeleton) pass for
`DOCS_PROMPT_2_WRITE.md`. They record the page inventory re-derivation, the
`__all__`-versus-prompt-table divergences, and the landing-vs-class naming
resolution. Deep five-section content is written in a later wave.

## `__all__` re-derivation versus prompt inventory tables

- **Page**: page inventory for models, techniques, and orchestrator components.
- **Sources**: `src/nav/nav_model/__init__.py`,
  `src/nav/nav_technique/__init__.py`,
  `src/nav/nav_orchestrator/__init__.py` (`__all__` in each) versus the
  inventory tables in `DOCS_PROMPT_2_WRITE.md`.
- **Resolution**: Trusted `__all__` per the prompt's instruction. Findings:
  - Models: the six concrete models in the prompt table
    (`NavModelStars`, `NavModelBody`, `NavModelBodySimulated`,
    `NavModelRings`, `NavModelRingsSimulated`, `NavModelTitan`) all appear in
    `__all__`. `__all__` additionally exports the abstract base `NavModel`,
    the body/ring annotation bases `NavModelBodyBase` / `NavModelRingsBase`,
    and the helper `build_models_for_obs`. These are base classes / helpers,
    not concrete models, so they get no per-class page (consistent with the
    prompt). No concrete-set divergence.
  - Techniques: all ten concrete techniques in the prompt table appear in
    `__all__`. `__all__` additionally exports the ABC `NavTechnique`, the
    result dataclass `NavTechniqueResult`, the feasibility report
    `NavFeasibilityReport`, the confidence helpers (`ConfidenceSpec`,
    `ConfidenceTerm`, `evaluate_sigmoid_combination`), every per-technique
    `*Diagnostics` dataclass, and the helpers `filter_technique_names` /
    `run_manual_nav`. These map to the five shared-infrastructure pages
    (dt_fitting, image_derivatives, confidence, feasibility, diagnostics)
    rather than to per-technique pages. No concrete-set divergence.
  - Orchestrator: the nine component pages enumerated in the prompt
    (`NavOrchestrator`, `NavContext`, `NavResult`, `ensemble`,
    `NavImageClassifier`, `InstrumentSettings`, `Provenance`, curator,
    `NavFeatureSummary`) all resolve to symbols in `__all__`. Divergences:
    `__all__` also exports `OrchestratorPrep` (the return type of
    `NavOrchestrator.prepare`; documented on the orchestrator page, not a
    standalone page), `derive_confidence_rank` (a helper alongside `ensemble`;
    documented on the ensemble page), `ImageDerivativesConfig` /
    `build_image_edge_dt` / `compute_all_image_derivatives` /
    `compute_image_gradient_vu` (documented on the shared-infrastructure
    image-derivatives page, not under the orchestrator per the prompt),
    `ImageQualityThresholds` and `NavImageClassifierResult` (folded into the
    image-classifier page), `assert_diagnostic_fields_present` (folded into
    the curator page), and `STATUS_REASON_INFO_TEMPLATE` (folded into the
    orchestrator page per the prompt). `InstrumentSettings` and
    `instrument_settings_from_obs` are NOT re-exported from this package
    `__all__`; they live in `src/nav/nav_orchestrator/instrument_config.py`.
    The instrument-config page is named file-aligned
    (`dev_guide_orchestrator_instrument_config.rst`) per the prompt's
    file-aligned naming rule, so this does not affect the page filename.
- **Follow-up needed**: None. Page inventory is consistent with `__all__`;
  the extra exports are bases, helpers, or sub-components folded into an
  existing page per the prompt.

## Landing-vs-class naming collision (models family)

- **Page**: `dev_guide_navigation_models.rst` and its model family pages.
- **Sources**: `DOCS_PROMPT_2_WRITE.md` lists
  `dev_guide_navigation_models_bodies/_rings/_stars/_titan.rst` as family
  landing pages, but the file-aligned per-class names for the single-model
  rings/stars/titan families collide with those landing names.
- **Resolution** (per FOUNDATION-pass instruction):
  - `dev_guide_navigation_models_bodies.rst` is the body-family LANDING page;
    its toctree lists the two per-class pages
    `dev_guide_navigation_models_body.rst` (`NavModelBody`) and
    `dev_guide_navigation_models_body_simulated.rst`
    (`NavModelBodySimulated`).
  - `dev_guide_navigation_models_rings.rst` is the `NavModelRings` PER-CLASS
    page (not a separate ring landing); its sibling per-class page is
    `dev_guide_navigation_models_rings_simulated.rst`
    (`NavModelRingsSimulated`).
  - `dev_guide_navigation_models_stars.rst` (`NavModelStars`) and
    `dev_guide_navigation_models_titan.rst` (`NavModelTitan`) are single-model
    families, so the existing page IS the per-class page; no separate landing.
  - The ring, stars, titan per-class pages and the bodies-family landing all
    hang directly off the top models landing `dev_guide_navigation_models.rst`
    toctree.
- **Follow-up needed**: None. The content wave replaces the five-section body
  of each per-class page; the existing `_stars`, `_rings`, `_titan`, and
  `_bodies` pages keep their current content until that wave.
