# Codebase analysis – reference

Use this when you need concrete examples for a dimension or wording guidance.

## Example findings (by dimension)

**Structure**
- **Finding**: Single module `utils.py` is 1,200 lines and mixes I/O, parsing, and formatting. **Evidence**: `src/utils.py`. **Suggestion**: Split into `io.py`, `parsing.py`, `formatting.py` under `utils/` and re-export from `utils/__init__.py`.

**Best practices**
- **Finding**: Several functions use `except Exception` and pass, hiding failures. **Evidence**: `src/loader.py` lines 45, 89. **Suggestion**: Catch specific exceptions, log with `logging.exception`, and re-raise or return a sentinel where appropriate.

**Types**
- **Finding**: Public API in `api.py` has no return type annotations; mypy is not run in CI. **Evidence**: `pyproject.toml` has no `[tool.mypy]`; `api.py` functions lack `->`. **Suggestion**: Add mypy to CI, enable strict mode, and annotate public functions first.

**Testing**
- **Finding**: Coverage is ~45%; module `core/solver.py` has no direct tests. **Evidence**: `coverage report`; no `tests/test_solver.py`. **Suggestion**: Add unit tests for solver entry points and key branches; aim for ≥80% on core.

**Performance**
- **Finding**: Config is re-read from disk inside a loop in `process_batch`. **Evidence**: `src/batch.py` `process_batch` calls `load_config()` per item. **Suggestion**: Load config once outside the loop and pass it in or use a module-level cache.

**Maintainability**
- **Finding**: Feature flags and environment checks are scattered across 12 files. **Evidence**: Grep for `os.getenv("FEATURE_`)`. **Suggestion**: Centralize in a `config` or `features` module and inject into call sites.

**Security**
- **Finding**: Subprocess is invoked with `shell=True` and user-controlled input. **Evidence**: `src/runner.py` line 67. **Suggestion**: Use list form of arguments and avoid `shell=True`; validate/sanitize input.

**Dependencies**
- **Finding**: Runtime deps are in `requirements.txt` and `pyproject.toml` with different versions. **Evidence**: `numpy` in requirements.txt pinned, in pyproject.toml minimum. **Suggestion**: Use `pyproject.toml` as single source of truth; remove duplicate requirements.txt or generate from it.

**Technical debt**
- **Finding**: 40+ TODO comments with no issue links or owners. **Evidence**: `grep -r TODO src`. **Suggestion**: Link TODOs to issues, or triage and remove obsolete ones; add a policy in CONTRIBUTING.

## Severity phrasing

- Critical: "must be addressed before…", "exposes…", "prevents…"
- High: "significantly increases…", "will make it difficult to…"
- Medium: "recommended to…", "would improve…"
- Low: "consider…", "optional:…"

## When project rules exist

- "Per project rule in `.cursor/rules/python_best_practices.mdc`, …"
- "This conflicts with the project's convention that …"
- "Align with project rule: … (see python_best_practices.mdc)."
