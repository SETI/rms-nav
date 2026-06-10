# Prompt 2 of 2: Write per-model and per-technique developer documentation

This is the second of two prompts. Run `DOCS_PROMPT_1_MIGRATE.md` first; it relocates the developer guide to `docs/dev_guide/` and renames the legacy `developer_guide_*.rst` files to `dev_guide_*.rst`. This prompt assumes that layout already exists.

## Project context

`rms-nav` is a Python 3.11+ spacecraft image navigation system distributed on PyPI as `rms-nav`. It determines precise pointing offsets for images from Cassini ISS, Voyager ISS, Galileo SSI, and New Horizons LORRI by comparing observed images against synthetic models generated from SPICE kernels. The repo also produces PDS4 bundles, per-pixel backplanes, and body / ring mosaics. The pipeline has five stages, each with a class hierarchy:

1. `DataSet` (`src/nav/dataset/`) enumerates images.
2. `ObsSnapshotInst` (`src/nav/obs/`) reads one image into an oops `Observation`.
3. **The orchestrator subsystem** (`src/nav/nav_orchestrator/`) drives the pipeline.  Its top-level driver class is `NavOrchestrator`; its public surface includes the `NavContext` per-image state dataclass, the `NavResult` final-output dataclass, the `ensemble` reconciler, the `NavImageClassifier` quick-fail classifier, the per-instrument `InstrumentSettings`, the `Provenance` reproducibility envelope, and the JSON curator.  Each gets its own page (see "Orchestrator components" below).
4. **`NavModel`** (`src/nav/nav_model/`) renders synthetic predictions of what *should* be at each pixel.
5. **`NavTechnique`** (`src/nav/nav_technique/`) matches model predictions against the real image to produce a `(dv, du)` offset and (when geometry allows) a rotation.

Offset convention: predicted position `(v, u)` means actual position is `(v + dv, u + du)`. Every class inherits from `NavBase` (`src/nav/support/nav_base.py`) for shared `config` and a `pdslogger.PdsLogger`.

Configuration is loaded from YAML files in `src/nav/config_files/`, merged in numeric filename order. Per-technique tuning lives under `techniques.<TechniqueName>.tuning` in `config_510_techniques.yaml`. Per-model config sections vary; see `Config` in `src/nav/config/config.py`.

## Audience

These pages are for **internal contributors to the `rms-nav` codebase** — engineers and scientists who will read, modify, or extend the navigation pipeline. Assume the reader is comfortable with Python, numpy, basic image-processing concepts (gradients, distance transforms, NCC), and least-squares optimisation, but not necessarily with this codebase or with SPICE / oops. Explain pipeline-internal terms (`NavContext`, `extfov`, `feature`, `NavFeasibilityReport`) on first use; do not explain numpy or scipy.

## Task

Produce one RST file per concrete navigation model, one per concrete navigation technique, and one per shared-infrastructure topic. Output is reStructuredText built by Sphinx; the repo CI runs `sphinx-build -W -b html docs docs/_build` (warnings as errors), so produced files must build cleanly. Also run `sphinx-build -b linkcheck docs docs/_linkcheck` and confirm no broken internal links.

## Inventory

Re-derive the current set of concrete classes from the source before writing — do not trust this prompt's tables blindly. The canonical lists are `__all__` in `src/nav/nav_model/__init__.py` and `src/nav/nav_technique/__init__.py`. If the live `__all__` disagrees with the tables below, **trust `__all__`** and append the divergence to `DOCS_RESULTS.md`.

### Concrete navigation models

| Class | Source file |
|---|---|
| `NavModelStars` | `src/nav/nav_model/stars/nav_model_stars.py` |
| `NavModelBody` | `src/nav/nav_model/nav_model_body.py` |
| `NavModelBodySimulated` | `src/nav/nav_model/nav_model_body_simulated.py` |
| `NavModelRings` | `src/nav/nav_model/nav_model_rings.py` |
| `NavModelRingsSimulated` | `src/nav/nav_model/nav_model_rings_simulated.py` |
| `NavModelTitan` | `src/nav/nav_model/nav_model_titan.py` |

### Concrete navigation techniques

| Class | Source file |
|---|---|
| `BodyLimbNav` | `src/nav/nav_technique/nav_technique_body_limb.py` |
| `BodyTerminatorNav` | `src/nav/nav_technique/nav_technique_body_terminator.py` |
| `BodyDiscCorrelateNav` | `src/nav/nav_technique/nav_technique_body_disc.py` |
| `BodyBlobNav` | `src/nav/nav_technique/nav_technique_body_blob.py` |
| `RingEdgeNav` | `src/nav/nav_technique/nav_technique_ring_edge.py` |
| `RingAnnulusNav` | `src/nav/nav_technique/nav_technique_ring_annulus.py` |
| `StarFieldFromCatalogNav` | `src/nav/nav_technique/nav_technique_star_field.py` |
| `StarUniqueMatchNav` | `src/nav/nav_technique/nav_technique_star_unique_match.py` |
| `StarRefineNav` | `src/nav/nav_technique/nav_technique_star_refine.py` |
| `NavTechniqueManual` | `src/nav/nav_technique/nav_technique_manual.py` |

### Shared infrastructure topics

Each gets its own page; concrete-class pages cross-reference these instead of duplicating their content.

- DT fitting machinery — `src/nav/nav_technique/dt_fitting.py` (used by limb, terminator, ring-edge, ring-annulus).
- Image-side derivatives — `src/nav/nav_orchestrator/image_derivatives.py` (gradient magnitude, gradient vector, edge distance transform).
- Confidence calibration — `src/nav/nav_technique/confidence.py` and `confidence_config.py`.
- Feasibility reporting — `src/nav/nav_technique/feasibility.py`.
- Per-technique diagnostics dataclasses — `src/nav/nav_technique/diagnostics.py`.

### Orchestrator components

The orchestrator (`src/nav/nav_orchestrator/`) is the top-level driver that turns one observation into one ``NavResult``.  It is more than a single class: it is a small subsystem of cooperating dataclasses, helpers, and a pipeline.  Each public component below gets its own page following the same five-section template as a technique, with section adaptations spelled out under "File locations and naming" below.  Re-derive the inventory from ``__all__`` in ``src/nav/nav_orchestrator/__init__.py`` and trust ``__all__`` over this table; append divergences to ``DOCS_RESULTS.md``.

| Component | Source file | Kind |
|---|---|---|
| `NavOrchestrator` | `src/nav/nav_orchestrator/orchestrator.py` | Driver class (two-pass pipeline + ensemble) |
| `NavContext` | `src/nav/nav_orchestrator/nav_context.py` | Per-image frozen dataclass |
| `NavResult` | `src/nav/nav_orchestrator/nav_result.py` | Final-output frozen dataclass |
| `ensemble` (function) + `EnsembleConfig` | `src/nav/nav_orchestrator/ensemble.py` | Per-technique reconciler |
| `NavImageClassifier` + `ImageQualityThresholds` + `NavImageClassifierResult` | `src/nav/nav_orchestrator/image_classifier.py`, `image_classifier_result.py` | Quick-fail classifier |
| `InstrumentSettings` + `instrument_settings_from_obs` | `src/nav/nav_orchestrator/instrument_config.py` | Per-instrument resolved settings |
| `Provenance` + `collect_provenance_metadata` | `src/nav/nav_orchestrator/provenance.py` | Reproducibility envelope |
| `build_metadata_dict` + `assert_diagnostic_fields_present` | `src/nav/nav_orchestrator/curator.py` | JSON curation |
| `NavFeatureSummary` | `src/nav/nav_orchestrator/feature_summary.py` | Per-feature post-mortem entry |
| `STATUS_REASON_INFO_TEMPLATE` | `src/nav/nav_orchestrator/status_reason_info.py` | Per-status-reason operator log lines |

The image-derivatives module (`image_derivatives.py`) is documented under shared-infrastructure rather than under the orchestrator (techniques sample its products directly) — a single page covers both concerns.

## File locations and naming

All new pages live in `docs/dev_guide/`. Filenames are file-aligned with the source module (not the class name), so a class rename does not force a doc rename:

- Models: `dev_guide_navigation_models_<source-module-suffix>.rst`. Strip the `nav_model_` prefix from the source filename. Examples:
  - `NavModelBody` (`nav_model_body.py`) → `dev_guide_navigation_models_body.rst`
  - `NavModelBodySimulated` (`nav_model_body_simulated.py`) → `dev_guide_navigation_models_body_simulated.rst`
  - `NavModelStars` (`stars/nav_model_stars.py`) → `dev_guide_navigation_models_stars.rst` (already exists from migration; replace its content per this prompt)
- Techniques: `dev_guide_techniques_<source-module-suffix>.rst`. Strip the `nav_technique_` prefix. Examples:
  - `BodyLimbNav` (`nav_technique_body_limb.py`) → `dev_guide_techniques_body_limb.rst`
  - `BodyDiscCorrelateNav` (`nav_technique_body_disc.py`) → `dev_guide_techniques_body_disc.rst` (file-aligned, not class-aligned).
- Shared infrastructure: `dev_guide_techniques_<topic>.rst`, where `<topic>` is the source module suffix or a short topic name. Examples: `dev_guide_techniques_dt_fitting.rst`, `dev_guide_techniques_image_derivatives.rst`, `dev_guide_techniques_confidence.rst`, `dev_guide_techniques_feasibility.rst`, `dev_guide_techniques_diagnostics.rst`.
- Orchestrator components: `dev_guide_orchestrator_<source-module-suffix>.rst`, file-aligned with the source module:
  - `NavOrchestrator` (`orchestrator.py`) → `dev_guide_orchestrator_orchestrator.rst`
  - `NavContext` (`nav_context.py`) → `dev_guide_orchestrator_nav_context.rst`
  - `NavResult` (`nav_result.py`) → `dev_guide_orchestrator_nav_result.rst`
  - `ensemble` (`ensemble.py`) → `dev_guide_orchestrator_ensemble.rst`
  - `NavImageClassifier` (`image_classifier.py`) → `dev_guide_orchestrator_image_classifier.rst`
  - `InstrumentSettings` (`instrument_config.py`) → `dev_guide_orchestrator_instrument_config.rst`
  - `Provenance` (`provenance.py`) → `dev_guide_orchestrator_provenance.rst`
  - Curator (`curator.py`) → `dev_guide_orchestrator_curator.rst`
  - `NavFeatureSummary` (`feature_summary.py`) → `dev_guide_orchestrator_feature_summary.rst`
  - `STATUS_REASON_INFO_TEMPLATE` (`status_reason_info.py`) → folded into the `NavOrchestrator` page; not a standalone page.

The current `dev_guide_techniques.rst` (a single 500+ line file covering all techniques) and the per-family pages (`dev_guide_navigation_models_bodies.rst`, `_rings.rst`, `_stars.rst`, `_titan.rst`) become **landing pages**.  A landing page is *only*:

1. A short prose overview (a few paragraphs) introducing the family and listing the registered concrete subclasses with one-line descriptions plus a `:class:` cross-reference each.
2. A `.. toctree::` block enumerating every per-class and shared-infrastructure page in the family.

A landing page must not contain configuration tables, per-class call paths, confidence-formula rows, emission-rule tables, or any other detail that belongs on a per-class page.  When stripping the legacy detailed content out of a landing page, the goal is overview + toctree only — duplication between the landing page and a per-class page is a defect.  The legacy content may be discarded; it is not authoritative (see Source-of-truth rules below).

Add every new file to the appropriate landing page's `toctree`.  Bump `:maxdepth:` on `docs/dev_guide/dev_guide.rst` so per-class pages reachable through three levels of recursion (e.g. `dev_guide.rst` → `dev_guide_navigation_models.rst` → `dev_guide_navigation_models_bodies.rst` → `dev_guide_navigation_models_body.rst`) appear in the inline TOC; verify every page is reachable from the top-level `docs/index.rst` chain after the toctree edits.

If a referenced public class lives in a module that is not yet included in the corresponding `docs/api_reference/api_*.rst` file, add an `automodule::` directive there so the cross-reference resolves.  Likewise add an entry to `intersphinx_mapping` in `docs/conf.py` when the page references a third-party Python library (e.g. SciPy) that the existing mapping does not already cover.

## Required structure for every page

Every page uses these five top-level sections in this order. Each section's underline must use `=` to make it a level-1 heading; subsections use `-` for level 2 and `~` for level 3.

```rst
Overview
========

Theory
======

Configuration
=============

Implementation
==============

Examples
========
```

What goes in each section is described in the next subsection. The "Code walkthrough" of earlier drafts is folded into Implementation; the "Examples" section is new.

### What each section must cover

**Overview**

- One paragraph naming the image feature this model predicts (or the feature this technique exploits) and the high-level approach.
- For techniques: one sentence on when feasibility passes and one on when feasibility fails.
- For models: one sentence on which observations get an instance and how many instances are produced (e.g. one per body whose bounding box overlaps the extended FOV).

**Theory**

- A self-contained mathematical and algorithmic description.
- **No Python identifiers in this section.** No file names, class names, function names, or attribute names. Describe the algorithm in mathematical and prose terms a reader could implement from scratch in any language. (The Code-walkthrough material lives in Implementation; Theory is the conceptual layer.)
- Required content:
  - The geometric or photometric reasoning behind the approach.
  - The mathematical formulation in `.. math::` blocks. Equations may be numbered with `:label:` if cross-referenced; otherwise unlabelled is fine.
  - The optimisation algorithm: cost function, search strategy, convergence criterion, robustness mechanism (e.g. M-estimator weighting, RANSAC, etc.).
  - Restrictions and assumptions: what scenes the technique fails on, what SPICE / image quality is required, what geometric configurations are unobservable (e.g. rank-deficient cases).
  - Sources of uncertainty: what the reported covariance does and does not capture, expressed as a factual description of the algorithm's current behaviour. Do not speculate about future improvements.

**Configuration**

- Name the YAML file and the section path (e.g. ``techniques.BodyLimbNav.tuning`` in ``src/nav/config_files/config_510_techniques.yaml``).
- A wrap-friendly bullet list — *not* an `.. list-table::` — with one bullet per key.  The Read-the-Docs theme renders multi-column tables in a narrow column that forces horizontal scrolling, so every Configuration and Confidence-formula entry must use a bullet list whose lines wrap naturally at the 100-column prose limit.
- Each bullet's lead text is `` ``key_name`` — type, default \`\`value\`\` units.  Effect description.``  Example:
  - `` ``min_arc_px`` — float, default ``30.0`` px.  Minimum surviving vertex count per LIMB_ARC for feasibility. ``
  - Every key under the technique's `tuning` block (or the model's config section) must appear as a bullet.
  - "Default" reads the literal value from the YAML, including `null` if that is the literal value.  For a value that is currently `null`, just write `null` — do not annotate it as a placeholder, do not mention pending calibration, do not flag it for tuning.
  - "Units" is `dimensionless` if the value carries no physical unit.  Use `(dimensionless)` or `(count)` parenthetically when there is no natural unit suffix.
  - "Effect" is one short prose sentence — what the value controls and which direction tightens / loosens behaviour.
- For per-instrument overrides (`config_4N0_inst_*.yaml`), follow the bullet list with a brief subsection listing which keys can be overridden per instrument and pointing at the relevant per-instrument YAML.
- For confidence-formula coefficients (`alpha0`, `terms`, `hard_zero_if`) — these live alongside `tuning` in the same YAML stanza but are a separate concern.  Document them in their own subsection ("Confidence formula") with a parallel bullet list: each term as `` ``feature_name`` — alpha = X, offset = Y, divisor = Z, cap at W (or "no cap").  What the diagnostic measures.``  Cross-reference the shared confidence page with `:doc:` for the sigmoid mathematics rather than re-deriving it.

**Implementation**

- Source files involved (paths only; no line numbers).
- Public class name and base class. Defer signature detail to autodoc — link with `:py:class:`, do not duplicate.
- For techniques: list the `accepts_feature_types`, the value of `requires_prior`, and the `confidence_attributes` set, by attribute name.
- A narrative trace of the call path — what `navigate()` (or the model's `create_model()` / `to_features()`) does, in the order it does it, naming the helper functions invoked at each stage. Refer to functions and classes by name only; never by line number.
- When the class delegates to shared infrastructure (DT fitting, confidence evaluation, feasibility, image derivatives), cross-reference the shared-infrastructure page with `:doc:` and summarise the call site in one or two sentences. Do not re-explain the shared algorithm.

**Examples**

Every page includes at least one worked example. Use the named scenes in `tests/integration/image_library/images/` as a stable corpus to draw from. Available scene names include `body_full_fov`, `body_mostly_offscreen`, `body_partial_overflow`, `below_resolution_body`, `multi_body`, `high_phase_terminator`, `one_bright_star_no_body`, `ring_only_curved`, and `star_dominated`; check `git ls-files tests/integration/image_library/images/` for the live list.

For each example, include:

- The scene name and a one-sentence description of what is in the image.
- For a model page: which features are emitted (counts, kinds), and any reliability gates that fire.
- For a technique page: whether feasibility passes, the rough `(dv, du)` outcome, and which diagnostics drive the confidence value.
- For a shared-infrastructure page: a small numerical illustration (e.g. given a 200-vertex polyline on a 1024² image and a 30 px search window, the integer-NCC stage scans X candidate offsets in Y operations).

If a scene's expected behaviour is not documented in the YAML sidecar (`tests/integration/image_library/images/<scene>/<image>.yaml`) or recoverable from the integration test, append a note to `DOCS_RESULTS.md` and pick a different scene rather than guessing. Do not invent numerical results.

## Falsifiable completeness rules

A page is complete when all of the following hold.  Use these as a checklist before declaring done.

For every page:

- All five required sections are present with the canonical headings.
- The Configuration bullet list contains one bullet for every key under the relevant YAML section.  Run `grep -c '^      [a-z_]\+:' <path>` against the YAML stanza and confirm the count matches the bullet count, or document why it does not.
- Every public method (every method without a leading underscore) on the documented class is named at least once in Implementation.
- Every shared-infrastructure call site delegates explanation to a `:doc:`-linked page; the same algorithm is not described twice.
- At least one Example uses a named scene from `tests/integration/image_library/images/`.

For technique pages additionally:

- The branches inside `navigate()` that change the result shape (1-star vs. 2-star, with-rotation vs. without, prior vs. no-prior, etc.) are each described in Implementation.
- Every field on the technique's `Diagnostics` dataclass is named in either Implementation or the Confidence-formula subsection.
- Every entry in `confidence_attributes` is named.

For model pages additionally:

- Every `NavFeatureType` the model emits is named.
- The `instances_for_obs` policy is described (one per obs, one per body, etc.).

## Orchestrator pages

The orchestrator subsystem has heterogeneous component shapes — a driver class, frozen dataclasses, free functions, and per-instrument resolvers — that the technique-shaped five-section template does not fit cleanly.  Every orchestrator page still uses the canonical five top-level headings (`Overview`, `Theory`, `Configuration`, `Implementation`, `Examples`) and obeys the cross-reference rules above; the per-section adaptations below replace the technique-shaped guidance for the sections that do not apply.

`dev_guide_orchestrator.rst` becomes a landing page (overview + toctree only) following the same rule as the per-family landing pages.

### Per-component section adaptations

**`NavOrchestrator` page** (`dev_guide_orchestrator_orchestrator.rst`)
  - **Overview:** what the driver does (one observation → one `NavResult`).  Name the two passes (prior-free, prior-required) and the four short-circuit gates (`_HARD_FAILURE_TO_REASON`, no-features, all-gated, no-feasible-techniques).
  - **Theory:** the algorithmic flow as a numbered pipeline; the hard-failure short-circuit policy; the "models are plugin-sandboxed" exception-handling discipline (every `NavModel` / `NavTechnique` exception is logged and treated as zero output rather than raised through to the caller).  No Python identifiers per the existing constraint.
  - **Configuration:** YAML keys consumed by the orchestrator are entirely upstream — the per-instrument `config_4N0_inst_*.yaml` blocks (`data_units`, `noise`, `image_quality_thresholds`, `camera_rotation`, `signal_dn_to_image_unit_scale`).  Document the keys that flow in via `instrument_settings_from_obs` as a wrap-friendly bullet list, plus the constructor-level overrides (`only_models`, `only_techniques`, `EnsembleConfig`, `ImageDerivativesConfig`, `ImageQualityThresholds`, `rms_nav_version`).  Cross-reference the per-component pages with `:doc:` for the dataclass details.
  - **Implementation:** source files involved; the public methods (`prepare`, `navigate`); the private call path of `navigate` step-by-step including pass-1 → ensemble → pass-2 → final ensemble; the four gate-failure paths through `_fail`; the glob-pattern filter for models and techniques.  Reference `STATUS_REASON_INFO_TEMPLATE` here.
  - **Examples:** worked example using a named scene from `tests/integration/image_library/images/` showing what a `NavResult` carries after a successful (`ok`) navigation, a `failed` navigation, and a `conflicted` navigation.  At least one example should walk the pass-1 → pass-2 hand-off (e.g. `StarUniqueMatchNav` produces the prior, `StarRefineNav` consumes it).

**Dataclass-only pages** (`NavContext`, `NavResult`, `NavFeatureSummary`, `Provenance`, `EnsembleConfig`, `ImageQualityThresholds`, `NavImageClassifierResult`, `InstrumentSettings`)
  - **Overview:** what role the dataclass plays in the pipeline.  Name the constructor (`NavOrchestrator._make_context`, `NavResult.ok`/`.failed`/`.conflicted`, etc.) and the consumers (which orchestrator phase or technique reads each field).
  - **Theory:** when the dataclass encodes a non-trivial invariant or convention (e.g. `NavContext` fields are extfov-shaped, `NavResult` enforces `status='failed'` ⇒ `offset_px=None`), describe those invariants here.  When the dataclass is a pure container with no algorithmic content, the Theory section is a single paragraph stating that — do *not* invent algorithmic content for an inert container.
  - **Configuration:** if the dataclass has YAML-tunable defaults (e.g. `EnsembleConfig` mirrors `config_540_orchestrator.yaml` defaults; `ImageQualityThresholds` mirrors per-instrument YAML), bullet-list every field as on a technique page.  When defaults live solely as Python module-level constants and there is no YAML override path today, say so in one sentence and bullet-list the constants instead.
  - **Implementation:** every public field on the dataclass must be named in this section as a `:py:attr:` cross-reference.  Mention every classmethod / property / `__post_init__` invariant.  Use plain code formatting for any private fields the dataclass keeps.
  - **Examples:** a short worked example showing field values for a representative scene (e.g. `NavContext.image_classifier.image_class == 'clean'`, `NavResult.status == 'ok'` with sigma values from `body_partial_overflow`).  When no scene-derived numerical example is meaningful (e.g. `Provenance` carries timestamps and SHAs), provide a short YAML-like illustration of the populated fields instead.

**Free-function pages** (`ensemble`, `curator`'s `build_metadata_dict`, `instrument_settings_from_obs`)
  - **Overview / Theory:** as for a technique page — the algorithm or transformation, identifier-free in Theory.  For `ensemble` specifically, walk the seven-step reconciliation: drop spurious, drop at-edge unless empty, single-link Mahalanobis grouping, summed-confidence selection, precision-weighted merge, disagreement penalty, conflict detection.
  - **Configuration:** for `ensemble`, document `EnsembleConfig` fields as a wrap-friendly bullet list.  For `build_metadata_dict`, document the rounding policy constants (`PIXEL_DECIMALS`, `CONFIDENCE_DECIMALS`, `ET_DECIMALS`) and the JSON-inf sentinel.  For `instrument_settings_from_obs`, document the per-instrument YAML keys it consumes.
  - **Implementation:** source file; the public entry point's signature deferred to autodoc; private helpers named in the call path.
  - **Examples:** worked numerical illustrations (e.g. given two technique results with disagreeing offsets, show how the Mahalanobis grouping decides whether they fuse or conflict).

**Pages whose configuration block is short** (`Provenance`, `NavFeatureSummary`, `STATUS_REASON_INFO_TEMPLATE`)
  - When a component has no configuration in the technique sense, make the Configuration section a single short paragraph that says so and points at the upstream source (e.g. "`Provenance` is populated at navigate time by `collect_provenance_metadata`; no YAML knobs apply").  The section heading must still be present.

### Falsifiable completeness rules — orchestrator additions

In addition to the universal completeness rules below:

- For the `NavOrchestrator` page: every public method named on the class is described in Implementation; every short-circuit `NavStatusReason` returned by `_fail` is named at least once.
- For each dataclass-only page: every public field is a `:py:attr:` cross-reference in Implementation.  Every classmethod constructor is named.  Every `__post_init__` invariant is mentioned.
- For the `ensemble` page: each of the seven reconciliation steps is described in Theory; every `EnsembleConfig` field appears in Configuration; the rank-deficient-covariance handling via `pinvh` is described in either Theory or Implementation.



A page is complete when all of the following hold. Use these as a checklist before declaring done.

For every page:

- All five required sections are present with the canonical headings.
- The Configuration bullet list contains one bullet for every key under the relevant YAML section.  Run `grep -c '^      [a-z_]\+:' <path>` against the YAML stanza and confirm the count matches the bullet count, or document why it does not.
- Every public method (every method without a leading underscore) on the documented class is named at least once in Implementation.
- Every shared-infrastructure call site delegates explanation to a `:doc:`-linked page; the same algorithm is not described twice.
- At least one Example uses a named scene from `tests/integration/image_library/images/`.

For technique pages additionally:

- The branches inside `navigate()` that change the result shape (1-star vs. 2-star, with-rotation vs. without, prior vs. no-prior, etc.) are each described in Implementation.
- Every field on the technique's `Diagnostics` dataclass is named in either Implementation or the Confidence-formula subsection.
- Every entry in `confidence_attributes` is named.

For model pages additionally:

- Every `NavFeatureType` the model emits is named.
- The `instances_for_obs` policy is described (one per obs, one per body, etc.).

## Cross-reference and autodoc policy

- **Trust autodoc.** The repo already has `docs/api_reference/` with autogenerated API pages. Do not duplicate signatures, parameter lists, or attribute lists in the dev-guide pages. Cross-reference with `:py:class:`, `:py:func:`, `:py:meth:`, `:py:attr:`, `:py:mod:`, `:py:exc:` — these survive file renames; they break on symbol renames, the same as any other code reference.
- **Every** class, function, method, and dataclass attribute name that exists in the project's autodoc surface must be a real cross-reference, not a backtick-quoted plain-code mention.  This applies to every section of the page: Overview, Configuration, Implementation, and Examples.  The Theory section remains identifier-free per the constraint above.
- Built-in Python exceptions (`RuntimeError`, `ValueError`, etc.) and standard-library symbols mentioned in prose use `:py:exc:` / `:py:class:` / `:py:func:` and resolve via the Python intersphinx mapping; do not leave them as plain backticks.  Add a third-party project (e.g. SciPy) to `intersphinx_mapping` in `docs/conf.py` if a referenced symbol does not yet resolve.
- The exception is package-private modules and underscore-prefixed names (private helpers, `_registry`, `__all__`, dunders).  These are not part of the public autodoc surface; quote them as plain `` ``identifier`` `` with no role.
- Use `~` to elide module prefixes for readability: `:py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav`` renders as `BodyLimbNav`.
- Cross-link sibling dev-guide pages with `:doc:` using the document name without the `.rst` extension.
- Verify resolution by running `sphinx-build -W -n -b html docs docs/_build` (the `-n` nitpicky flag surfaces every unresolved cross-reference as a warning) on the pages you produced.  Pre-existing nitpicky warnings on unrelated files are out of scope; warnings inside any new or edited dev-guide page must be zero.
- Do not write inline-rst-link URLs to GitHub or external services.

## Style constraints

- **One space after sentence-ending periods.** Never two. Applies to every prose paragraph in every RST file.
- **Wrap prose at 100 columns.** Tables, code blocks, and `.. math::` directives may exceed.
- **Page titles use title case.** "Body Navigation Model", "DT Fitting", "Navigation Techniques", not "Body navigation model" or "Navigation models".  Subsection headings can follow whichever case the surrounding pages already use.
- **No emojis** anywhere.
- **No backwards-compatibility shims, no speculative future-feature documentation, no `.. note::` flags about unresolved questions** (those go in `DOCS_RESULTS.md`).
- **No line numbers anywhere.** Refer to code by file, class, and function names only.
- **Document only what the code currently does.** Do not mention pending calibration, placeholder coefficients, planned tuning sweeps, future phases, `PLACEHOLDER` markers, or anything described as "TODO" / "Phase N" / "pending" in `AUTONAV_PLAN.md` or YAML comments. Record current numeric defaults verbatim and move on; the reader does not need to know they will be retuned.
- **RST idiom only — not content — should be inferred from existing files.** Match the existing files' directive choice, table syntax, heading underline characters, and indentation. Do not treat their prose, structure, or coverage as a template.

## Source-of-truth rules

When sources disagree, trust them in this order:

1. The Python source code.
2. Module / class / function docstrings.
3. YAML comments in `src/nav/config_files/*.yaml`.
4. `CLAUDE.md` (project instructions).
5. `AUTONAV_PLAN.md` (project history; verify against code, never carry pending-work statements into the published docs).

The pre-existing `dev_guide_*.rst` files (originally `developer_guide_*.rst` before prompt 1's migration) are **background reading** — they may help an author orient themselves, but they are not part of the source-of-truth ranking. Their wording, structure, claims, omissions, and depth must not restrict, modify, or influence the rewritten pages. A new page that ends up looking nothing like the file it replaced is a normal outcome.

## Unresolved-issue routing

Both **gaps in source material** ("the docstrings do not justify the claim I would need to make to complete this section") and **conflicts between sources** ("the docstring and the YAML comment disagree") go to `DOCS_RESULTS.md` at the repo root, never into the published RST. Format each entry as:

```markdown
## <short title>

- **Page**: <which RST page being written>
- **Sources**: <file paths and excerpts that disagree, or "no source for X">
- **Resolution**: <which side I chose and why, or "skipped this claim">
- **Follow-up needed**: <what the operator should review>
```

The published RST never contains `.. note::` or `.. TODO::` flags about uncertainty.

## Pilot batch

Before producing the full set, deliver the following four pages and stop for review. The pilot is chosen to exercise every substantially different page shape exactly once.

1. `dev_guide_navigation_models_body.rst` — per-body iteration model with a simulated sibling. Exercises: per-body instance count, cross-link to `_body_simulated`, `LIMB_ARC` and terminator feature emission.
2. `dev_guide_techniques_body_limb.rst` — DT-based, no prior, single-pass technique. Exercises: shared-infrastructure cross-references (DT fitting, image derivatives, confidence), Diagnostics-dataclass coverage, named-scene example.
3. `dev_guide_techniques_star_refine.rst` — prior-required (pass-2) technique. Exercises: the `requires_prior=True` shape, Procrustes / 1-inlier branches, rank-deficient covariance.
4. `dev_guide_techniques_dt_fitting.rst` — shared-infrastructure page. Exercises: page shape with no Configuration section in the technique sense (this is shared algorithmic infrastructure, not a directly-configurable component) — adapt the Configuration section to document only the constants exposed in the module's public API, and explain in one sentence why the section is short.

After the pilot is approved, write the remaining model, technique, shared-infrastructure, and orchestrator pages.  Order: shared-infrastructure first (they unblock cross-references from techniques), then techniques grouped by family (body / ring / star, then `NavTechniqueManual`), then the remaining model pages, then the orchestrator pages (driver class first, then the dataclasses and free functions whose cross-references the driver page consumes).

## Workflow

1. Confirm the migration from prompt 1 is complete: `docs/dev_guide/dev_guide_*.rst` exists, `git ls-files docs/developer_guide*.rst` is empty, `sphinx-build -W` is clean.
2. Re-derive the inventory from `__all__` in the two `__init__.py` files. If it disagrees with the tables in this prompt, trust `__all__` and append a note to `DOCS_RESULTS.md`.
3. Produce the pilot batch (four pages).  Run `sphinx-build -W -n -b html docs docs/_build` (warnings as errors, nitpicky cross-references) and `sphinx-build -b linkcheck`.  Filter the nitpicky warnings to your new pages and confirm zero pilot-page warnings; pre-existing warnings on unrelated files are out of scope.  Stop and report.
4. After review, produce the remaining pages in the order described under **Pilot batch**.  Run the two Sphinx builds after each batch and confirm zero warnings on new and edited pages plus zero broken internal links.
5. Leave changes uncommitted. The user folds them into the phase commit themselves.

## Done criteria

- One RST file per concrete model (six files), one per concrete technique (ten files), one per shared-infrastructure topic (five files), one per orchestrator component (nine files: `NavOrchestrator`, `NavContext`, `NavResult`, `ensemble`, `NavImageClassifier`, `InstrumentSettings`, `Provenance`, curator, `NavFeatureSummary`).
- Every file uses the five-section template with the canonical headings.
- Every Configuration bullet-list count matches the YAML key count for that section, or the divergence is logged in `DOCS_RESULTS.md`.  No `.. list-table::` blocks anywhere on a per-class page; Configuration and Confidence-formula entries are always wrap-friendly bullet lists.
- Every landing page is overview + toctree only, with no per-class detail.  Every per-class and shared-infrastructure page is reachable from the top-level `docs/index.rst` chain via the recursive toctree (verify by reading the rendered sidebar after the build).
- Every class, function, method, and dataclass-attribute name in the prose resolves to a real autodoc anchor in the rendered HTML.  Run `sphinx-build -W -n -b html` and confirm zero warnings on new or edited pages; spot-check the rendered HTML for `href` attributes on the cross-references.
- `sphinx-build -W -b html docs docs/_build` exits clean.
- `sphinx-build -b linkcheck docs docs/_linkcheck` reports no broken internal links.
- The published RST contains no `.. TODO::` directives, no `.. note::` flags about unresolved questions, no mentions of pending calibration or placeholder values.
- All unresolved questions encountered during writing live in `DOCS_RESULTS.md`.
