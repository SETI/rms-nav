# SpinDoctor Logging Redesign Plan

*Detail plan for reorganizing SpinDoctor's logging into two clearly-scoped
loggers with a single per-module level system, applied uniformly across
every pipeline program in the repository. Written to be handed to a
developer or an implementing model with no other briefing beyond
`/seti/newnav/CLAUDE.md`. Conventions from `CLAUDE.md` apply throughout:
line length 100, mypy strict, pdslogger-only logging, Google-style
docstrings with `Parameters:`, Conventional Commits, one logical change
per commit.*

Integration branch: `rf_logging_redesign`, cut from `main` at ce2b410. Each
phase below lands as its own pull request targeting that branch, so each gets
an independent review pass before the branch merges to `main`.

---

## 1. Purpose and scope

SpinDoctor's logging grew per driver rather than by design. The result is
that most programs have no configured logging at all, one program's log
output cannot be silenced without silencing another's, and the per-module
level keys that exist in the configuration are mostly dead. The logging
configuration itself lives inside `general`, which has become the section
that absorbs whatever does not fit elsewhere.

This plan replaces that with a two-logger contract, one level per module, a
top-level `logging` configuration section that carries its own per-program
overrides, and a single command-line surface **shared by every pipeline
program in the repository**. Consistency across those programs is a
requirement of this work, not a later cleanup: a user who learns the logging
flags for `sd_offset` knows them for `sd_mosaic` and `sd_backplanes` too.

The statistics and GUI programs are deliberately excluded and use `print()`
instead — see section 2.1.

**In scope:** the three per-image backends (`nav`, `backplane`, `reproj`),
every program listed in section 2.1 as carrying a logger, the new top-level
`logging` configuration section and its per-program overrides, the shared
command-line arguments, cloud-task console isolation, tests, and a new
user-guide section.

**Out of scope:** `util/` and `experiments/`, which are outside the
distributed package. The broader configuration restructuring this work
exposed — the flat top-level namespace, `general` as a catch-all, the absence
of any axis for program-scoped settings, and `environment` having no
documented schema — is tracked as its own issue (section 7); this plan
touches only what logging needs. `sd_create_bundle_cloud_tasks` is excluded
deliberately: running bundle assembly as cloud tasks is a structural mistake
in the current system rather than a logging problem, also tracked separately.

---

## 2. Target design

### 2.1 Two loggers, and which programs get them

**Main logger.** One per program execution, created at startup and kept for
the life of the run. It reports what the program is doing at the top level:
which image it is about to process, counts, totals, elapsed time, the path
of each image log it wrote. **Every pipeline program gets one**; the
statistics and GUI programs do not, and use `print()` instead.

**Image logger.** Scoped to one image inside one backend. It carries the
detail of that image's processing. Only programs that process images
individually, through one of the three per-image backends, get one.

| Program | Main | Image backend |
|---|---|---|
| `sd_offset` | yes | `nav` |
| `sd_offset_cloud_tasks` | no | `nav` |
| `sd_backplanes` | yes | `backplane` |
| `sd_backplanes_cloud_tasks` | no | `backplane` |
| `sd_mosaic`, `sd_mosaic_rings`, `sd_mosaic_body` | yes | `reproj` |
| `sd_mosaic_cloud_tasks` | no | `reproj` |
| `sd_create_bundle` | yes | none |
| `sd_consolidate_metadata` | yes | none |
| `sd_stats_ingest`, `sd_stats_report` | no — `print()` | none |
| `sd_create_simulated_image` | no — `print()` | none |
| `sd_backplane_viewer`, `sd_mosaic_display` | no — `print()` | none |

`sd_create_bundle` does iterate images, and `cli/pds4/bundle_data.py:48`
opens a per-image section today, but bundle assembly is a packaging step
rather than an image-processing backend. That section stays, logging into the
main log — the main logger supports sections like any other.
`bundle_data.py` already receives its logger as a parameter, so this is a
call-site change only.

The two loggers never duplicate each other. A message belongs to exactly
one.

**The statistics programs and the GUI programs carry no logger at all** and
write to the terminal with `print()`. Neither is a batch pipeline: the
statistics programs produce a report to be read as it is generated, and a GUI
surfaces problems to the operator through `QMessageBox` dialogs, which these
modules already use. Neither benefits from log files, levels, or sinks, and
giving them the full flag surface would be noise.

Three of the five already have no logging to remove — `sd_create_simulated_image`
and its `cli/sim_editor/` package, and `sd_mosaic_display`, contain zero logger
references. The conversion is therefore small: two calls in
`cli/stats/ingest.py:236,276`, one in `cli/stats/report.py:782`, one
`self._logger = MAIN_LOGGER` binding in `sd_backplane_viewer.py:209`, and four
calls across the `ui/mosaic_viewer` package (`common.py:202`,
`body_window.py:699`, `ring_window.py:337,1181`).

Two details for the conversion. The two `logger.exception('Error loading %s',
path)` calls (`body_window.py:699`, `ring_window.py:1181`) must become
`traceback.print_exc()` or equivalent, or the traceback is silently lost.
And `CLAUDE.md` states "Never bare `print()` in library code", which
`spindoctor.ui` is by location — so that rule needs an explicit, documented
exception for the GUI package, recorded in `.cursor/rules/` alongside the
existing scoped exemptions. A GUI that is half `print()` and half logger
would be worse than either.

### 2.2 Sinks and defaults

Each logger has two possible sinks, console (stdout) and file. Defaults:

| Logger | Console | File |
|---|---|---|
| main  | on  | on  |
| image | off | on  |

**When a sink is enabled, its level is the level for that module. Console
and file always share a level.** There is no per-sink level anywhere in the
system, which is what makes `logger.open(title, level=...)` sufficient to
express a module's verbosity.

The mechanism needs care. A pdslogger section level is a floor applied before
the handlers, and each handler then applies its own level, so what reaches a
sink is the *more severe* of the two. Handlers are therefore built at the most
verbose level any module could ask for, and the per-section floor does all of
the discrimination. Building them at the plain `image` level instead silently
drops every module configured more verbose than it, while the section summary
still counts the dropped records — output reported but never written.

Enabling and disabling a sink changes only which handlers are attached when
the logger is built. Nothing else in the system is conditional on it.

**Invariant: a logger must never be left with zero handlers.** A PdsLogger
with no handlers does not go silent; it falls back to a bare `print()` to
stdout that ignores every level (`pdslogger/__init__.py:2068`, and stated
outright in the pdslogger README: "if you really wish to not see any
messages, you must assign it the `NULL_HANDLER`"). When both sinks are
disabled, the builder attaches `pdslogger.NULL_HANDLER`.

### 2.3 Log file locations

A log root is resolved once per run, following the same argument /
configuration / environment precedence as every other root in
`spindoctor/config/config_helper.py`:

1. `--log-root`
2. `environment.log_root`
3. `NAV_LOG_ROOT`
4. Default: `{nav_results_root}/logs`

Layout underneath it:

```text
{log_root}/{program}/main_{datetime}.log                 # main logger
{log_root}/{backend}/{results_path_stub}_{datetime}.log  # image logger
```

`{datetime}` is `%Y-%m-%dT%H-%M-%S` **in UTC**, applied to main logs as well
as image logs. The format matches the existing per-image convention; the
timezone does not, because the three sites producing it today
(`navigate_image_files.py:164`, `sd_offset.py:305`, `sd_mosaic.py:89`) use
naive local time. Cloud-task workers processing one batch may sit in different
zones, and a local-time name is ambiguous across a daylight-saving fall-back,
so names would neither sort nor correlate. Those three sites are replaced in
Phases 6 and 7.

`{backend}` is one of `nav`, `backplane`, `reproj`. `{results_path_stub}` is the existing
`ImageFile.results_path_stub`, which is `{volume}/{filespec}` without the
extension (`dataset_pds3_cassini_iss.py:187`), so a Cassini image navigated
by either nav driver lands at
`{log_root}/nav/COISS_2001/data/1234567890_1234567890/N1234567890_1_2026-07-28T12-00-00.log`.

The image subtree is keyed by **backend**, not by program, so an image's log
for a given stage is in one predictable place regardless of which driver
produced it — `sd_offset` and `sd_offset_cloud_tasks` share
`{log_root}/nav/`. Main logs are keyed by program, since that is what a main
log describes.

This moves the reprojection image logs, which currently go to
`<output_dir>/logs/` alongside the npz/fits products
(`sd_mosaic.py:85-89`). They join the common tree at `{log_root}/reproj/`.

### 2.4 Program identity

Every dispatch module declares a `PROGRAM_NAME` constant. It is the key for
both the main log directory and the per-program configuration block, so the
two can never drift apart.

- The three mosaic entry points (`sd_mosaic`, `sd_mosaic_rings`,
  `sd_mosaic_body`) all declare `sd_mosaic`; they are one program with mode
  variants. The two display entry points likewise declare
  `sd_mosaic_display`.
- Each cloud-task driver declares the same identity as its interactive
  counterpart (`sd_offset_cloud_tasks` declares `sd_offset`). One
  configuration block then governs both, matching the decision that they
  share one backend log tree. Cloud-task drivers have no main logger, so
  there is no main-log-directory collision.

### 2.5 Configuration

Logging gets its own **top-level `logging` section**, shipped in a new
`config_files/config_015_logging.yaml`, and `Config` gains a matching
`logging` property alongside the existing section properties. It does not
live under `general`: `general` currently holds a TrueType font directory
and nine `log_level_*` keys while `planets` sits outside it in the same file,
and putting a foundational subsystem's configuration in the section that
absorbs leftovers is how the present arrangement went wrong.

Per-program overrides live inside the same section, under `programs`, using
the identical schema. Keeping them there rather than in a separate top-level
`programs` tree puts each override adjacent to what it overrides, and avoids
introducing a top-level section whose only member would be logging.

```yaml
logging:
  strict_scope: false    # raise on an out-of-scope image log; true in tests

  main: info             # main logger default
  image: info            # image logger default, applies to all modules
  techniques:
    default: debug       # overrides the image default for all techniques
    titan_haze: info     # overrides the category default for one technique
  models:
    rings: warning
  other:
    annotate: error
    nav_correlate_all: debug

  programs:
    sd_mosaic:
      main: warning
      image: debug
    sd_backplanes:
      image: debug
```

Note what a program block does *not* need. Each program drives at most one
per-image backend, so "the per-image verbosity of this program" is exactly
`image` — there is no module key named after a backend. `sd_mosaic`'s
reprojection detail is `image: debug`, not `reproj: debug` under a wrapper.

`main` and `image` are the two global defaults, both `info`. Each category
(`techniques`, `models`, `other`) may carry a `default` that overrides the
image default for that whole category, and any number of module keys that
override the category default. A block under `programs` may set any of the
same keys for one program.

Every existing `log_level_*` key in `config_010_general.yaml` is removed:
the four `log_level_main_*` / `log_level_image_*` keys and the five
`log_level_model_*` / `log_level_annotate` keys.

**Module keys are snake_case.** Techniques and models each carry an explicit
`log_key` class attribute rather than deriving one from the class name, so
the key is greppable in the source and does not break when a class is
renamed. A derived default (`CamelCase` minus a trailing `Nav`, lowercased
with underscores) is available for classes that do not declare one, but every
current class declares its key explicitly:

| Category | Keys |
|---|---|
| `techniques` | `body_disc_correlate`, `body_blob`, `body_limb`, `body_terminator`, `ring_edge`, `ring_annulus`, `star_field_from_catalog`, `star_unique_match`, `star_refine`, `titan_haze`, `manual` |
| `models` | `stars`, `body`, `rings`, `titan` |
| `other` | `annotate`, `correlate`, `ensemble`, `image_derivatives`, `obs`, `orchestrator`, `provenance` |

`other` covers image-scoped components that are neither a model nor a
technique. A component earns a key only once it opens a section of its own,
because a level is applied at `logger.open()`; a key naming a component that
never opens one would validate cleanly and then do nothing, which is the
failure that left five `log_level_model_*` keys dead in the configuration this
plan replaces. Every key listed here has a section, and a test asserts that
correspondence rather than restating the list.

Six of the seven are module-level functions rather than classes, so they take
their section from the `logged_section` decorator rather than
`NavBase.log_section`. Decorating avoids reshaping each function around a
`with` block, and `ParamSpec` keeps the wrapped signature intact.

`other` carries no keys for the backends either: a backend's verbosity is the
`image` level of whichever program drives it, per the note in the previous
subsection. Main-scoped components likewise have no per-module keys; they log
at the main level.

Every key is validated against the technique and model registries and against
the fixed `other` enumeration at configuration load time, in the top-level
block and in every `programs.*` block. An unrecognized key raises
`ValueError` at startup rather than being silently ignored, which is how the
current dead keys went unnoticed. Program names under `programs` are
validated against the registered `PROGRAM_NAME` set the same way.

### 2.6 Level resolution

Resolution happens in two steps, which keeps the per-program dimension from
multiplying the precedence rules.

**Step 1 — merge.** The effective logging configuration for a run is the
top-level `logging` block deep-merged with `logging.programs.{PROGRAM_NAME}`,
with the program block winning key by key.

**Step 2 — precedence.** Against that merged configuration, most specific
wins:

```text
--log-level MODULE=LEVEL          (command line, per module)
  > <category>.<module>
  > <category>.default
  > --log-level-main / --log-level-image  (command line, per logger)
  > --log-level LEVEL                     (command line, both loggers)
  > main / image
  > INFO
```

Allowed level names are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, and
`NONE`. `NONE` is added because the current allowed set stops at `CRITICAL`,
which only reads as "silent" because nothing in the library happens to log at
`CRITICAL` — an incidental property, not a guarantee.

The resolved level is applied at each component's existing `logger.open(...)`
call as its `level=` argument. The 22 component-owned `open()` sites become
level-aware; before this work only three were. The driver-level per-image
sections are converted with their drivers.

### 2.7 Command-line surface

One shared `add_logging_arguments(parser)` helper, used by every program in
section 2.1 that carries a logger — eight dispatch modules. The same flags mean
the same thing everywhere; programs without an image logger reject the
image-specific flags with a clear message rather than accepting and ignoring
them. The statistics and GUI programs take none of these flags, since they
have no loggers to configure.

| Argument | Effect |
|---|---|
| `--log-root PATH` | Log root override |
| `--log-main-to-console` / `--no-log-main-to-console` | Toggle main console sink (default on) |
| `--log-main-to-file` / `--no-log-main-to-file` | Toggle main file sink (default on) |
| `--log-image-to-console` / `--no-log-image-to-console` | Toggle image console sink (default off) |
| `--log-image-to-file` / `--no-log-image-to-file` | Toggle image file sink (default on) |
| `--log-level LEVEL` | Global default for both loggers |
| `--log-level MODULE=LEVEL` | Per-module override; repeatable |
| `--log-level-main LEVEL` | Main logger default |
| `--log-level-image LEVEL` | Image logger default |

The four toggles use `argparse.BooleanOptionalAction`, giving the `--no-`
forms for free.

`--log-level` accepts both forms. A bare `LEVEL` sets the global default; a
`MODULE=LEVEL` pair sets one module. The flag is repeatable, so
`--log-level debug --log-level titan_haze=info` is valid and means "debug
everywhere except the haze technique".

These replace `--log-level-main-console`, `--log-level-main-file`,
`--log-level-image-console`, and `--log-level-image-file`, which are removed
along with the per-sink concept, and the ad-hoc `--log-level` in `sd_mosaic`
and `sd_consolidate_metadata`, which is subsumed.

### 2.8 Cloud-task behavior

The three cloud-task drivers in scope **create no main logger**. Their
console belongs to the cloud_tasks worker, whose verbosity is controlled by
cloud_tasks' own configuration. Image logs are written to files under the
same `{log_root}/{backend}/` tree as the corresponding interactive driver.

Three concrete guarantees, each of which closes a verified leak:

1. **No console handler is ever attached to an image logger in a cloud-task
   driver.** This closes the stdout path through the image logger's stream
   handler.
2. **Both loggers get `propagate = False`.** `cloud_tasks`'s
   `_worker_process_main` calls `logging.basicConfig(level=INFO)` inside each
   worker subprocess (`cloud_tasks/worker/worker.py:1561`), installing a root
   stderr handler. Without this, every image-logger line is re-emitted a
   second time on stderr with a `Process-N` prefix.
3. **The no-zero-handlers invariant applies.** With no main logger
   configured, any surviving main-logger call site would otherwise reach the
   `print()` fallback. In a cloud-task driver the main logger is bound to a
   null sink so those calls are inert.

### 2.9 Scope violations are programming errors

An image-role component that logs when no image scope is open is a bug, not a
supported mode. An audit of the current tree found no legitimate case:

- Mosaic accumulation looked like one — `sd_mosaic` calls `mosaic.add()`
  outside any image scope (`sd_mosaic.py:406,500`) — but `BodyMosaic.add()`
  and `RingMosaic.add()` do not log. Both modules bind their logger inside
  `reproject()` (`bodies.py:1015`, `rings.py:1041`), which always runs within
  an image scope.
- The three PyQt mosaic-viewer modules bind `logger = IMAGE_LOGGER` at module
  level (`body_window.py:52`, `ring_window.py:59`, `common.py:28`) and a
  viewer has no image scope at all. This is a mis-binding to be corrected to
  main role under this plan, not a case to be supported.

Behavior when it happens anyway:

- The message is **routed to the main logger** so it is never lost.
- A `WARNING` is emitted to the main logger naming the module and function
  that logged out of scope, **deduplicated per call site** so a loop cannot
  flood the log.
- Under `logging.strict_scope: true` (default `false`) the violation raises
  instead.

Strict scope is **opt-in for tests, not suite-wide**. The original intent was
to enable it everywhere so a new violation failed CI, but that premise does not
survive contact with the suite: enabling it globally fails 497 tests, and
essentially none of them are mis-bindings. A unit test exercising
`NavModelTitan.to_annotations()` calls it directly, outside any pipeline, which
is correct isolation testing — the component is properly image-role and the
test is properly written. "An image-role component only logs inside an image
scope" holds for production paths, not for tests that drive components
individually.

Enforcement therefore belongs where a scope genuinely should be open: tests
that drive a real pipeline. A `strict_log_scope` fixture makes that a one-line
request, and the pipeline-level tests adopt it as Phases 5 through 7 wire the
drivers. In production the warning remains the signal.

---

## 3. Current state

What follows is measured, not assumed. It is recorded here so the
implementer can recognize each thing being replaced.

**Most programs never configure logging.** Of the 14 `sd_*.py` dispatch
modules, 3 call `setup_logging()` (`sd_offset`, `sd_mosaic`,
`sd_consolidate_metadata`). Five more use the loggers without ever
configuring them — `sd_backplanes` (12 call sites), `sd_create_bundle` (11),
`sd_mosaic_cloud_tasks` (7), `sd_create_bundle_cloud_tasks` (2),
`sd_backplane_viewer` (2). Every one of those is running on pdslogger's
`print()` fallback, which no level or configuration can reach.

**Three uncontrolled console paths exist under cloud_tasks**, all reproduced
directly: the image logger's stdout stream handler (`config/logger.py:155`);
the `print()` fallback for the unconfigured main logger; and stderr
re-emission through cloud_tasks' root handler. Setting
`log_level_image_console: CRITICAL` closes only the first.

**Logging configuration lives in a catch-all section.**
`config_010_general.yaml` puts nine `log_level_*` keys next to
`truetype_font_dir` (itself marked `# TODO`), with `planets` as a separate
top-level list in the same file. There is no logging section and no
program-scoped configuration of any kind.

**Per-module levels are mostly dead.** `config_010_general.yaml:18-22`
declares five keys; only `log_level_model_rings` and `log_level_annotate` are
read (`nav_model_rings.py:269`, `annotations.py:87`).
`log_level_model_stars`, `log_level_model_bodies`, and
`log_level_model_titan` are read nowhere. No technique has a key at all.

**Level resolution skips the environment.** `_resolve_level`
(`config/logger.py:40`) goes arguments, then configuration, then a hardcoded
`'INFO'`. Every other resolver in the codebase goes arguments,
configuration, environment variable, error.

**`IMAGE_LOGGER` is a module-level singleton imported by 29 modules (94
references)** — `nav_model`, `nav_technique`, `reproj`, `obs`,
`support/correlate.py`, `ui`, and `NavBase`, which binds it for every
subclass at construction (`support/nav_base.py:27`).

**`DataSet` inherits `NavBase`** (`dataset/dataset.py:146`) and therefore
logs image enumeration — work that spans the whole run — to the image logger.
The three PyQt mosaic-viewer modules bind the image logger despite having no
image scope.

**Image log paths differ per program.** Nav writes to
`{nav_results_root}/logs/{stub}_{datetime}.log`
(`navigate_image_files.py:165`); reprojection writes to
`{output_dir}/logs/{stub}_{datetime}.log` (`sd_mosaic.py:85`).

**PdsLogger names are globally unique and never freed.** `PdsLogger.__init__`
raises `ValueError` if the name is already in `_LOOKUP`
(`pdslogger/__init__.py:294`), and nothing ever removes an entry. Measured:
constructing 5000 image loggers leaves 5004 entries in `_LOOKUP` and 5101 in
stdlib's `loggerDict`, permanently. A literal logger-per-image design would
accumulate one of each per image for the life of the process.

**There is no `tests/spindoctor/config/test_logger.py`.** The logging code
has no direct test coverage.

---

## 4. Implementation phases

Each phase is a separate commit, and each leaves the tree green.

### Phase 1 — Configuration: `logging` section and program identity

Add `PROGRAM_NAME` to each logger-carrying dispatch module per section 2.4.
Add the top-level `logging` section in a new
`config_files/config_015_logging.yaml`, and a `logging` property to `Config`.
Add registry-backed validation of module keys and program names, applied to
the top-level block and to every `programs` block.

The nine superseded `log_level_*` keys in `config_010_general.yaml` stay for
now and are removed in Phase 2, when the resolver that replaces their readers
lands. Deleting them here would leave a phase whose only effect is to drop
configurability, and the point of phasing is that each pull request is
independently coherent.

### Phase 2 — Logging core

New module `spindoctor/config/logging_config.py` (the existing
`config/logger.py` keeps the two logger objects and gains the proxy):

- `resolve_log_levels(program_name, arguments, config)` implementing the
  merge and precedence of section 2.6, returning a `LogLevels` dataclass.
- `LogSinks` dataclass: the four booleans plus the resolved log root.
- `build_main_logger(logger, program_name, sinks, levels)` and
  `build_image_log_handlers(backend, results_path_stub, sinks, levels)` —
  attach exactly the enabled handlers, or `NULL_HANDLER` if none, and report
  the path written.  The image side returns handlers rather than attaching
  them, because they are scoped to one `logger.open(...)` section; the caller
  owns closing them.  Neither builds a logger, so nothing is added to
  pdslogger's process-global registry.
- `get_log_root(arguments, config)` in `config_helper.py`, matching the shape
  of `get_nav_results_root`.
- Level-name validation extended with `NONE`.

This phase builds the core and tests it directly; it does not wire it into any
program. Wiring happens in Phases 5 through 7, which replace the eight call
sites that still reach the old `setup_logging` / `image_log_handlers` path.
The old `_resolve_level` and the nine `log_level_*` keys in
`config_010_general.yaml` are therefore retired in Phase 6, when the last of
those callers goes, along with the one test that asserts on them
(`test_config_helper.py:431`). Removing them earlier would leave a phase that
only deletes working configurability.

### Phase 3 — Image logger proxy, role binding, and scope enforcement

`IMAGE_LOGGER` becomes a proxy that resolves through a `ContextVar` to the
image logger for the currently-active backend, so none of the 94 call sites
change. The proxy is a typed delegating class, not a `__getattr__`
passthrough, because mypy strict will not accept `Any` returns from the
logging methods. It forwards the level methods, `open()`, `close()`,
`set_level()`, and `exception()`.

The proxy implements section 2.9: out-of-scope use routes to the main logger,
emits a deduplicated warning naming the offending call site, and raises under
strict scope.

`NavBase` gains an explicit logger role. Every one of the 29 importing
modules is audited once and assigned `main` or `image`. `DataSet` moves to
`main`; models, techniques, `obs`, `reproj`, annotation, ensemble, and
provenance stay on `image`. The three `ui/mosaic_viewer` modules leave the
logging system entirely, converting their four calls to `print()` per
section 2.1, together with the statistics programs and
`sd_backplane_viewer.py:209`; the `.cursor/rules/` no-print exception for
`spindoctor.ui` lands in the same commit.

### Phase 4 — Per-module levels at the `open()` sites

Add `NavBase.log_section(title, **kwargs)`, which looks up the component's
resolved level and calls `self.logger.open(title, level=level, **kwargs)`, and
`NavBase.log_key` for a component whose configured name is not the one derived
from its class. Unlike `log_role`, `log_key` is inherited normally, so a family
that shares one key declares it once on their base. Convert the 22
component-owned `open()` sites and delete the three ad-hoc
`config.general.get('log_level_*')` reads that this replaces.

`set_log_levels` installs the resolved levels for a run, so a component deep in
the pipeline can ask for its own level without the resolved set being threaded
through every constructor. Until a driver installs one — the drivers are wired
in Phases 5 through 7 — the configuration's global defaults are resolved
lazily, so the mechanism works from this phase onward rather than waiting.

The driver-level per-image sections in `navigate_image_files`, `sd_offset`,
`sd_mosaic`, `backplanes` and `bundle_data` are not component-owned and are
converted with their drivers in Phases 5 through 7.

### Phase 5 — Command-line surface

`spindoctor/cli/logging_args.py` with `add_logging_arguments(parser)`, wired
into the five interactive programs: `sd_offset`, `sd_backplanes`, `sd_mosaic`,
`sd_create_bundle` and `sd_consolidate_metadata`. Each builds its main logger
at startup through `build_run_logging`, which resolves the levels, installs
them for components to read, and stamps one timestamp for the whole run so a
run's log files share it. A program with no image logger passes
`has_image_logger=False` and so rejects the image flags by name rather than
accepting and ignoring them.

`nav`'s per-image logging moves to `build_image_log_handlers` here rather than
in Phase 6, because `navigate_image_files` is the driver-side half of the same
change and Phase 6 is what removes the *last* old caller.

The four per-sink flags go, as do the ad-hoc `--log-level` in `sd_mosaic` and
`sd_consolidate_metadata`, along with `sd_mosaic`'s `IMAGE_LOGGER.set_level`
call, which the resolver now covers.

The cloud-task drivers are wired in Phase 7 instead, with the isolation that
defines their logging: giving them the argument surface here would hand them a
main logger that section 2.8 says they must not have.

### Phase 6 — Image loggers for the backplane and reproj backends

Route `backplane` and `reproj` per-image logging through
`build_image_log_handlers`, moving reprojection logs from `{output_dir}/logs/`
to `{log_root}/reproj/`, and give both programs the image-logger flags they
withhold until they can honor them. Retire `image_log_handlers()` and the
bespoke path builders in `sd_mosaic.py` and `sd_mosaic_cloud_tasks.py`, and
`sd_mosaic`'s interim `IMAGE_LOGGER.set_level` call. Repoint
`bundle_data.py`'s per-image section at the main logger.

This removes the last caller of the old resolution path, so `setup_logging`,
`_resolve_level`, and the nine `log_level_*` keys in
`config_010_general.yaml` are deleted here, along with
`test_config_helper.py:431`, which asserts on one of them.

### Phase 7 — Cloud-task isolation

For the three cloud-task drivers in scope: no main logger,
`propagate = False` on both loggers, no console handler on the image logger,
null sink for the main logger so surviving call sites are inert. Applied
inside `process_task`, which is what runs in the worker subprocess —
configuring in the parent does not carry across the process boundary.

### Phase 8 — Tests

New `tests/spindoctor/config/test_logger.py` plus additions to the CLI tests.
Per `CLAUDE.md`, pdslogger writes through its own stream handler, so tests
capture with `capsys`, not `caplog`. Coverage:

- Level precedence at each of the six tiers in section 2.6, and the
  top-level / `programs` merge, including a program block overriding one key
  while inheriting the rest.
- Each of the four sink toggles, including all-off attaching `NULL_HANDLER`
  and producing no output.
- Log path construction for main and all three backends, including the
  datetime suffix and the default log root derivation.
- Configuration validation: unknown module key raises at load, in both the
  top-level block and a `programs` block; unknown program name raises;
  unknown level name raises; `NONE` accepted.
- `--log-level` parsing for both the bare and `MODULE=LEVEL` forms.
- Role binding: a main-role component's output lands in the main log and not
  the image log, and the reverse.
- Scope enforcement: out-of-scope image logging warns once per call site and
  raises under strict scope; the suite runs with strict scope on.
- Argument-surface consistency: a parametrized test asserting every
  logger-carrying program in section 2.1 accepts the same logging flags with
  the same defaults, and that the statistics and GUI programs accept none
  of them.
- Program identity: every logger-carrying dispatch module declares
  `PROGRAM_NAME`, and the mosaic and cloud-task aliasing of section 2.4 holds.
- No logger imports survive in `cli/stats/`, `sd_backplane_viewer.py`,
  `sd_create_simulated_image.py`, `cli/sim_editor/`, `sd_mosaic_display.py`,
  or `ui/mosaic_viewer/` — asserted by a static check
  so the conversion cannot silently regress.
- Cloud-task silence: with a worker-like `logging.basicConfig(level=INFO)`
  active, a run of each cloud-task driver emits zero bytes on stdout and
  stderr while the image log file is complete.

### Phase 9 — Documentation

New `docs/user_guide/user_guide_logging.rst`, referenced from
`docs/user_guide.rst`, covering the two loggers and what belongs in each, the
sink defaults, log file locations and naming for all three backends, the
`logging` section schema including `programs` and the full module key table,
the precedence order, the command-line arguments with worked examples, and
cloud-task behavior. Update `docs/introduction_configuration.rst` for the new
section, the per-program user-guide pages for the flags, and the developer
guide for `PROGRAM_NAME`, `log_key`, `NavBase.log_section`, and the scope
rule.

---

## 5. Acceptance criteria

1. Every cloud-task driver in scope writes **zero bytes** to stdout and
   stderr from SpinDoctor code, with a complete per-image log file, under a
   worker subprocess that has called `logging.basicConfig(level=INFO)`.
   Asserted in a test.
2. Every logger-carrying program produces a main log at
   `{log_root}/{program}/main_{datetime}.log`, and every image-processing
   program a per-image log at `{log_root}/{backend}/{stub}_{datetime}.log`.
3. Every logger-carrying program in section 2.1 accepts the identical logging
   flag set with identical defaults, and the statistics and GUI programs accept
   none of it. Asserted by a parametrized test.
4. The statistics and GUI programs import no logger and write through
   `print()`, with tracebacks preserved where `exception()` was used.
5. A `logging.programs.sd_mosaic` block changes that program's levels and no
   other program's.
6. `--log-level titan_haze=debug` raises exactly that technique's verbosity
   in both sinks, with no other module affected.
7. Each of the four sink toggles independently controls its sink, and
   disabling every sink produces no output at all.
8. No PdsLogger anywhere in the codebase can reach the `print()` fallback.
   Enforced by a test that walks every logger the builders produce and
   asserts a non-empty handler list.
9. No image-role component logs outside an image scope anywhere in the suite,
   enforced by strict scope being on in tests.
10. An unknown module key, program name, or level name in the configuration
   fails at startup with a message naming the offending key.
11. No `_LOOKUP` growth proportional to image count: a run over N images
    leaves the registry size unchanged from a run over one.
12. `ruff check`, `ruff format --check`, `mypy --strict`, `sphinx-build -W`,
    and `pymarkdown scan` all pass; suite coverage stays at or above 90%.

---

## 6. Risks and constraints

**One active image per process.** A single reused image logger with swapped
handlers assumes one image is being processed at a time in a given process.
That holds today: the batch drivers are serial and cloud_tasks isolates
workers in subprocesses. If image-level threading is ever introduced, the
image logger must become thread-local rather than context-local. Recorded
here so the constraint is not rediscovered by a corrupted log.

**Section counters count at the floor.** pdslogger tallies messages that pass
the section floor, not what each handler emitted. Because console and file
share a level in this design, the two can never disagree — but the constraint
is the reason for that choice, and per-sink levels must not be reintroduced
without revisiting it.

**Registry validation needs the registries populated.** Validating module
keys at configuration load requires the technique and model registries to be
imported by then. If load order makes that awkward, validation moves to first
logger construction, still before any image is processed.

**Reduced console output by default.** The image logger's console sink
defaults to off, so an interactive run shows top-level progress rather than
per-component detail. `--log-image-to-console` restores it. This is the
intended change, but it is a visible difference in daily use and belongs in
the user guide prominently.

**Reprojection log relocation.** Moving reprojection image logs out of
`{output_dir}/logs/` will break any operator habit or script that reads them
there. Call it out in the user guide.

---

## 7. Follow-ups

Tracking issues to file alongside the implementation issue, per the project's
convention that future work gets an issue rather than a comment:

- **Configuration structure.** The top-level namespace mixes domain objects,
  pipeline stages, instruments, output products, and infrastructure with no
  axis distinguishing them; `general` is a catch-all; `environment` is an
  exposed section with no shipped default; and the grouping encoded in the
  `config_NNN_` filename prefixes is invisible from the keys themselves.
  Lifting `logging` out of `general` under this plan is one instance of the
  fix, not the whole of it.
- **`sd_create_bundle_cloud_tasks` should not exist.** Bundle assembly is a
  packaging step over an already-processed collection, not per-image work
  suited to a task queue. Its removal, and the entry point in
  `pyproject.toml`, want their own issue.
- A `remove_logger()` / registry-eviction API for `rms-pdslogger`. Not needed
  by this design, but it is the missing piece that would make a literal
  logger-per-image implementation viable, and the constraint is worth
  recording upstream.
- Logging for `util/` tooling, which is outside the distributed package but
  shares the configuration system and would benefit from the same flags.
