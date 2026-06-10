# Prompt 1 of 2: Relocate and rename the developer guide

This is the first of two prompts. Run this one to completion (verify a clean Sphinx build, fix every internal cross-reference) before running prompt 2 (`DOCS_PROMPT_2_WRITE.md`), which writes the actual new content against the relocated layout.

## Project context

`rms-nav` is a Python 3.11+ spacecraft image navigation system distributed on PyPI as `rms-nav`. The repo's Sphinx documentation lives in `docs/`. The "developer guide" portion is currently a flat collection of files at `docs/developer_guide*.rst`. This prompt moves all of those files into a new `docs/dev_guide/` subdirectory and renames each file from the legacy `developer_guide_<xxx>.rst` prefix to a shorter `dev_guide_<xxx>.rst`.

No content is rewritten in this step. Only the directory layout, filenames, and the cross-references that point at them change.

## Task

1. Move every existing `docs/developer_guide*.rst` file into `docs/dev_guide/` and rename it by stripping the `developer_` prefix.
2. Update every internal reference (`:doc:` roles, `:ref:` roles, `toctree` entries, `docs/index.rst`, `docs/conf.py`, and any path-shaped tokens in `README.md`, `CONTRIBUTING.md`, `docs/contributing.rst`, `.cursor/rules/`) so the docs still build and link correctly.
3. Verify that `sphinx-build -W -b html docs docs/_build` exits with zero warnings and `sphinx-build -b linkcheck docs docs/_linkcheck` reports no broken internal links.
4. Append any unresolved issue (a path that could not be migrated cleanly, a cross-reference whose intent is unclear, a conflict between two sources of truth) to `DOCS_RESULTS.md` at the repo root, creating that file if it does not exist.
5. Leave all changes uncommitted on the working tree. The user folds them into the phase commit themselves.

## Migration table

The current set of files is the following. Re-confirm against `git ls-files docs/developer_guide*.rst` immediately before starting; if entries here no longer exist or new ones have appeared, update accordingly rather than running the table blind.

| Old path | New path |
|---|---|
| `docs/developer_guide.rst` | `docs/dev_guide/dev_guide.rst` |
| `docs/developer_guide_backplanes.rst` | `docs/dev_guide/dev_guide_backplanes.rst` |
| `docs/developer_guide_best_practices.rst` | `docs/dev_guide/dev_guide_best_practices.rst` |
| `docs/developer_guide_building_docs.rst` | `docs/dev_guide/dev_guide_building_docs.rst` |
| `docs/developer_guide_class_hierarchy.rst` | `docs/dev_guide/dev_guide_class_hierarchy.rst` |
| `docs/developer_guide_configuration.rst` | `docs/dev_guide/dev_guide_configuration.rst` |
| `docs/developer_guide_extending.rst` | `docs/dev_guide/dev_guide_extending.rst` |
| `docs/developer_guide_image_library.rst` | `docs/dev_guide/dev_guide_image_library.rst` |
| `docs/developer_guide_introduction.rst` | `docs/dev_guide/dev_guide_introduction.rst` |
| `docs/developer_guide_logging.rst` | `docs/dev_guide/dev_guide_logging.rst` |
| `docs/developer_guide_navigation_models.rst` | `docs/dev_guide/dev_guide_navigation_models.rst` |
| `docs/developer_guide_navigation_models_bodies.rst` | `docs/dev_guide/dev_guide_navigation_models_bodies.rst` |
| `docs/developer_guide_navigation_models_rings.rst` | `docs/dev_guide/dev_guide_navigation_models_rings.rst` |
| `docs/developer_guide_navigation_models_stars.rst` | `docs/dev_guide/dev_guide_navigation_models_stars.rst` |
| `docs/developer_guide_navigation_models_titan.rst` | `docs/dev_guide/dev_guide_navigation_models_titan.rst` |
| `docs/developer_guide_orchestrator.rst` | `docs/dev_guide/dev_guide_orchestrator.rst` |
| `docs/developer_guide_reprojection.rst` | `docs/dev_guide/dev_guide_reprojection.rst` |
| `docs/developer_guide_rotation.rst` | `docs/dev_guide/dev_guide_rotation.rst` |
| `docs/developer_guide_static_data.rst` | `docs/dev_guide/dev_guide_static_data.rst` |
| `docs/developer_guide_techniques.rst` | `docs/dev_guide/dev_guide_techniques.rst` |
| `docs/developer_guide_uncertainty.rst` | `docs/dev_guide/dev_guide_uncertainty.rst` |

Use `git mv` rather than delete-and-create so rename detection picks the move up cleanly.

## Internal references to fix

After the moves, retarget every reference to the old paths.

- `:doc:` roles that name `developer_guide_*` targets (with or without a leading `/`).
- `:ref:` roles whose label values contain the old token (rare but possible).
- `toctree` entries — including those wrapped in `.. toctree::` blocks with `:caption:`, `:hidden:`, `:numbered:`, etc.
- `docs/index.rst` — its top-level toctree references `developer_guide`; replace with `dev_guide/dev_guide`.
- `docs/conf.py` — any `master_doc`, `html_extra_path`, `exclude_patterns`, or templated path that names the old prefix.
- Plain-text path references in `README.md`, `CONTRIBUTING.md`, `docs/contributing.rst`, `scripts/*.sh`, and `.cursor/rules/*.md`.

### Verification of internal references

After fixing, run:

```bash
grep -rEn '(^|[^a-zA-Z_])developer_guide([_./]|$)' docs/ src/ scripts/ README.md CONTRIBUTING.md .cursor/
```

The grep should return zero lines that are path-shaped tokens (file paths, `:doc:` arguments, `toctree` entries, shell-script arguments). Pure prose mentions of "the developer guide" as a concept are fine; references that look like filesystem paths or Sphinx targets must all be updated.

If anything in that grep output is genuinely ambiguous (e.g. a sentence in a CHANGELOG-like file describing the rename historically), append it to `DOCS_RESULTS.md` and leave it as-is rather than guessing.

## Verification

Run both commands and confirm zero warnings / zero broken links:

```bash
sphinx-build -W -b html docs docs/_build
sphinx-build -b linkcheck docs docs/_linkcheck
```

If either fails, fix the underlying cross-reference rather than suppressing the warning. The renamed layout must build at least as cleanly as the pre-rename layout.

## Constraints

- **No content changes.** This step renames files and retargets references only. Do not edit prose, headings, or directives inside any moved file. Content rewrites happen in prompt 2.
- **One space after sentence-ending periods** in any prose you do touch (e.g. CHANGELOG-style notes, `DOCS_RESULTS.md` entries). Never two.
- **No commits, no pushes.** Leave changes uncommitted on the working tree.
- **No emojis** anywhere.

## Source-of-truth rules

When a rename target is ambiguous (e.g. a `:doc:` role uses a relative path that resolves differently after the move), trust sources in this order, and append the conflict to `DOCS_RESULTS.md`:

1. The behaviour Sphinx requires for the build to remain warning-free.
2. The Python source code (the ground truth for runtime behaviour, including any hardcoded doc URLs).
3. Module / class / function docstrings.
4. `CLAUDE.md` (project instructions).
5. `AUTONAV_PLAN.md` (project history; treat as background).

Background reading only, never authoritative on its own: the pre-rename `developer_guide_*.rst` files. They may be consulted to understand a cross-reference's intent but cannot resolve a tie.

## DOCS_RESULTS.md format

Append one section per unresolved issue:

```markdown
## <short title>

- **Where**: <file path> (or path + line context)
- **What**: <one-paragraph description of the issue>
- **Action taken**: <what you did, if anything>
- **Follow-up needed**: <what the operator should review>
```

The published RST never carries `.. note::` or `.. TODO::` flags about migration uncertainty. All such uncertainty lives in `DOCS_RESULTS.md`.

## Done criteria

- All 21 files (or whatever the live count is at run time) are at `docs/dev_guide/dev_guide_<xxx>.rst`.
- `git ls-files docs/developer_guide*.rst` returns nothing.
- `sphinx-build -W -b html docs docs/_build` exits clean.
- `sphinx-build -b linkcheck docs docs/_linkcheck` reports no broken internal links.
- The grep above returns no path-shaped tokens for `developer_guide`.
- `DOCS_RESULTS.md` either does not exist or contains only the issues you explicitly chose not to resolve in this step.
