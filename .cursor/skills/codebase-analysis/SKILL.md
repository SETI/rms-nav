---
name: codebase-analysis
description: Analyzes a codebase and produces high-level recommendations for restructuring, refactoring, and alignment with modern best practices. Use when the user asks to analyze the codebase, audit code quality, suggest improvements, refactoring ideas, or assess maintainability, performance, testability, or technical debt.
---

# Codebase Analysis

Produce a structured analysis and recommendations report. Do not implement changes unless the user asks; focus on **high-level findings and actionable suggestions**.

## Workflow

1. **Scope**: Confirm or infer scope (whole repo, a package, or a path). Default to the project root.
2. **Explore**: Scan layout (directories, key config files), entry points, tests, and docs. Use list_dir, grep, and semantic search; avoid reading every file.
3. **Assess**: Evaluate each dimension below. Note evidence (file paths, patterns) and severity (critical / high / medium / low).
4. **Synthesize**: Write the report using the output template. Prioritize by impact and effort; group related items.

## Dimensions to Assess

### 1. Structure and layout

- Package/module boundaries: clear separation, no circular imports, src-layout vs flat.
- File and module size: modules > ~500–1000 lines; single-file "god" modules.
- Naming: consistent with language norms (e.g. Python: lowercase_with_underscores, TitleCase for classes).
- Dead or orphaned code: unused modules, commented-out blocks, unreachable branches.
- Duplication: copy-paste, similar logic that could be shared (DRY).

**Evidence**: Paths, line counts, import graphs if available.

### 2. Best practices alignment

Compare against project rules when present (e.g. `.cursor/rules/python_best_practices.mdc`). Check:

- Naming (builtin shadowing, private `_` prefix, ALL_CAPS for module-level constants).
- Explicit checks vs exception-based control flow; falsy checks (`is None`, `len(x) == 0`).
- Imports: top of file, grouped and sorted; no wildcard imports.
- Function shape: ≤3 positional args, keyword-only for the rest. Return an object rather than a tuple of many results.
- Constants: no magic numbers/strings; config or env for tunables.
- Error handling: narrow try/except; no bare except; logging over print in libraries.
- Public API: clear `__all__`, `py.typed` for typed packages, separation of public vs `_private`.

**Evidence**: Rule name or quote, example file:line or pattern.

### 3. Types and static checks

- Type coverage: annotations on public API and new code; use of `Any`, untyped defs.
- Mypy (or equivalent): strictness, per-file overrides, global ignores.
- Linting: Ruff/Flake8/Pylint enabled; which rules; consistent formatting (e.g. Ruff format / Black).
- Docstrings: presence, format (e.g. Google), consistency with signatures and behavior.

**Evidence**: Config files, sample of annotated vs unannotated code.

### 4. Testing

- Structure: tests colocated or in `tests/`; mirror of source layout; naming (`test_*`).
- Coverage: approximate line/branch coverage; untested modules or critical paths.
- Quality: one assertion per test; no tests that ignore results or swallow exceptions; use of parametrize/fixtures; independence and parallelizability.
- Gaps: missing edge cases, error paths, or integration tests for key flows.

**Evidence**: `pytest.ini`/`pyproject.toml`, coverage report or commands, example test file.

### 5. Performance and resource use

- Hot paths: unnecessary work in loops, repeated allocations, O(n²) or worse algorithms where it matters.
- I/O: blocking calls in async code; missing timeouts; large files read into memory.
- Caching: repeated computation or lookups that could be cached or memoized.
- Dependencies: heavy or unused libraries; optional features that could be lazy-loaded.

**Evidence**: File:line or function name; no profiling required unless user provides data.

### 6. Maintainability and extensibility

- Coupling: tight dependencies between modules; hard-coded dependencies instead of injection.
- Cohesion: modules/classes with a single responsibility; clear boundaries.
- Extensibility: adding features without editing many files; use of hooks, plugins, or strategy-style patterns where appropriate.
- Documentation: README, architecture overview, contribution guide; docstrings and comments where logic is non-obvious.

**Evidence**: Import structure, example functions or classes.

### 7. Security and robustness

- Input validation: external input (CLI, files, env) validated at boundaries; no trust of caller data in libraries.
- Secrets: no credentials in code or logs; use of env or secret managers.
- Dependency hygiene: known vulnerable deps (`pip audit` / Dependabot); pinned or minimum versions.
- Paths and execution: path traversal risks; subprocess/shell usage and injection.

**Evidence**: Grep for patterns (e.g. `password`, `secret`, `eval`, `subprocess` with `shell=True`).

### 8. Dependencies and tooling

- Declared deps: single source of truth (e.g. `pyproject.toml`); optional groups (dev, docs).
- Version policy: minimum versions, avoidance of global pins for libraries.
- Tooling: consistent formatter and linter; CI runs checks and tests; no obsolete or conflicting config (e.g. both `setup.py` and `pyproject.toml` without clear roles).

**Evidence**: `pyproject.toml`, `requirements*.txt`, CI config.

### 9. Technical debt and risk

- Deprecations: use of deprecated APIs (stdlib, third-party); planned removals.
- Complexity: deeply nested conditionals; long functions; high cyclomatic complexity in critical code.
- TODOs/FIXMEs: concentration in one area; unlinked or vague items.
- Compatibility: Python version support; platform assumptions (e.g. paths, encoding).

**Evidence**: Grep for deprecation warnings, TODO/FIXME; example complex function.

## Output template

Use this structure for the report. Omit sections with no findings; keep each item concise with location and suggested direction.

```markdown
# Codebase analysis: [project or path]

## Summary
[2–4 sentences: overall health, top 2–3 priorities.]

## 1. Structure and layout
- **Finding**: [what]. **Evidence**: [where]. **Suggestion**: [action].
[Repeat as needed.]

## 2. Best practices alignment
[Same pattern; reference project rules if present.]

## 3. Types and static checks
...

## 4. Testing
...

## 5. Performance and resource use
...

## 6. Maintainability and extensibility
...

## 7. Security and robustness
...

## 8. Dependencies and tooling
...

## 9. Technical debt and risk
...

## Recommended priorities
1. [Highest impact, feasible first step]
2. [Next]
3. [Next]
```

## Severity and wording

- **Critical**: Security or data integrity risk; blocks testing or deployment; pervasive violation of a core rule.
- **High**: Significant maintainability or bug risk; large refactor needed if left as-is.
- **Medium**: Clear improvement; can be scheduled with normal work.
- **Low**: Nice to have; style or minor consistency.

Use "Consider…", "Prefer…", "Avoid…" for suggestions. For critical/high, state the impact (e.g. "increases risk of…", "makes testing difficult because…").

## Project-specific rules

If the repo contains `.cursor/rules/` (e.g. `python_best_practices.mdc`), treat those as the primary standard for "best practices alignment". Mention when a finding contradicts or reinforces a project rule. For Python repos, prefer referencing the rule file rather than repeating long rule text.

## Reference

For example findings and severity phrasing, see [reference.md](reference.md).

## Scope and depth

- Prefer breadth first: touch all dimensions, then go deeper only where impact is high or the user asks.
- For large codebases, sample by package or layer (e.g. core vs CLI vs tests) and call out areas not reviewed.
- If the user asks for "quick" or "high-level" analysis, limit to summary + 1–2 findings per dimension and a short priority list.
