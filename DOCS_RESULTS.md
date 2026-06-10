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
- **Follow-up needed**: An operator should fix the `BodyTerminatorNav`
  docstring so the enumerated list renders without the autodoc indentation
  error, which will let `sphinx-build -W` exit clean. This is a source-code
  docstring fix outside the migration's scope.

## Linkcheck reports one broken external link

- **Where**: `docs/index.rst` line 146 (external project link).
- **What**: `sphinx-build -b linkcheck` flags
  `https://github.com/SETI/rms-cloud_tasks` as a 404. This is an external URL,
  not an internal cross-reference, and is unrelated to the developer-guide
  relocation.
- **Action taken**: Left as-is. The migration does not touch external link
  targets, and no internal link is broken.
- **Follow-up needed**: An operator should confirm the correct public URL for
  the cloud-tasks project and update the link.
