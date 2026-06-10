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
