# SpinDoctor Results Index Plan

*Implementation plan for an optional, rebuildable database index over the
navigation results tree, so that programs which need a few fields per image
stop reading one JSON document per image. Written to be executed by an
implementing model with no briefing beyond `/seti/newnav/CLAUDE.md` and the
repository itself. Conventions from `CLAUDE.md` and `.cursor/rules/` apply
throughout: line length 100, mypy strict, pdslogger-only logging,
Google-style docstrings with `Parameters:`, Conventional Commits, one logical
change per commit, modules under 1000 lines, no issue numbers in docstrings
or `.rst` files (only in `#` comments).*

Integration branch: `rf_results_index`, cut from `main`. Each phase below
lands as its own pull request targeting that branch, so each gets an
independent review pass before the branch merges to `main`.

---

## 1. Purpose and scope

The navigation pipeline writes one `_metadata.json` per image under
`nav_results_root`. Every program downstream of navigation then reads one of
those documents per image it processes. On a local disk that is cheap. On a
cloud root it is one paid round trip per image per program, and a
Cassini-scale run is order 400,000 images. The curation triage tool is the
extreme case: up to ten full-tree recursive globs per candidate frame.

The fields those programs actually consume are narrow: `status`,
`status_error`, the top-level `offset`, and a handful of quality numbers. One
pass that reads every document once can answer all of those questions from
then on. That pass is the index this plan builds.

**In scope:** a stub-keyed schema over the results tree; a SQLAlchemy Core
data-access layer with a connection-URL backend selection (SQLite and
PostgreSQL); incremental ingest with per-root bookkeeping; a cloud-task
ingest driver; a `--results-db` opt-in on consuming programs; and the
conversion of the statistics programs, the backplane driver, reprojection
offset lookup, and the `ResultsFilter` selection filters to read through the
index when it is given.

**Out of scope, deliberately:**

- Replacing the JSON documents. They remain the authoritative record. The
  index is derived and disposable; deleting it costs nothing but the time to
  rebuild.
- Making any pipeline program require a database. Every program except
  `sd_stats_report` (which has no file-reading mode today and reads only a
  database by construction) must run correctly with no index at all, and
  that path stays the default.
- Writing the index from the navigation pipeline. Navigation writes JSON;
  ingest reads it. Nothing in the navigation path gains a database
  dependency.
- Automatic ingest. No batch driver invokes ingest as a side effect; an
  operator runs it.
- Bundle generation (`spindoctor.cli.pds4.bundle_data`), which serializes the
  entire navigation document into the PDS4 supplemental product, and
  `sd_consolidate_metadata`, which copies raw file bytes. Neither is served
  by a column schema; both keep reading files. Section 7 records the
  extension that would serve bundle generation.
- The curation triage tool (`util/cohort_curation/triage_stage_b.py`). It
  consumes five fields the schema does not carry (`missing_frac`,
  `saturation_frac`, classifier flags, the per-technique list, and result
  file paths), reads two roots, and keys by bare image name rather than by
  stub. Serving it means widening the schema for one out-of-package tool;
  that is a follow-up (section 7), not a phase.
- Ingesting documents written by an older metadata schema. Ingest reads the
  current schema and reports anything else as an error for that file.
- MySQL. PostgreSQL is the only server backend; MySQL was considered and
  rejected.

---

## 2. Target design

### 2.1 What the index is

A database whose rows are derived from the results tree by a separate ingest
step. A consumer given `--results-db` answers its questions with SQL instead
of file reads. A consumer given no such option behaves exactly as it does
today, reading files. (`sd_stats_report` is the one exception in both
directions: it reads only a database today and after this work, and it is
the one program for which the index is not optional.)

The index is a **snapshot**. It reflects the tree as of the last ingest over
the roots it covers. Consumers trust its contents as given: there is no
staleness detection, no re-verification against the tree, and no automatic
refresh. An operator who navigates more images and wants them visible runs
ingest again. This is a deliberate simplification, and it must be stated
plainly in the user guide, because the failure mode it permits -- a consumer
silently working from stale rows -- is otherwise surprising.

### 2.2 Keying

The primary key of `images` is `(root_url, results_path_stub)`.

`results_path_stub` is the volume-and-filespec path fragment every consumer
already uses to address a result, carried on `ImageFile.results_path_stub`.
A real example: `COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB`
(the PDS3 datasets build it as `{volume}/{filespec}` with only the final
extension stripped, so a `_CALIB` suffix survives). It is the only
identifier unique across volumes; the simulated dataset produces a bare
scene basename with no path separator at all, which is a valid stub.

`results_path_stub` does not appear in the metadata document. Ingest derives
it from the file's own location: the path of a metadata file relative to the
ingest root, with the `_metadata.json` suffix removed. This is exact by
construction, because that is precisely how the navigator chose the path it
wrote to.

**The ingest root is the `nav_results_root`, not a free-form path.**
`sd_stats_ingest` resolves its roots the way every consumer does (the
command line, then `config.environment.nav_results_root`, then
`NAV_RESULTS_ROOT`); an operator who ingests a subdirectory of a results
root would produce stubs no consumer's lookup can match, so the root
identity is part of the contract, not a convenience.

`root_url` is the ingested root in **normalized** form, and every consumer
normalizes its own resolved `nav_results_root` the same way before
comparing: `FCPath(root).absolute().as_posix()` with any trailing `/`
removed. A consumer whose normalized root has no completed `ingest_runs` row
(section 2.7) fails with a message naming its root and the roots the index
does contain -- absence of a row must never be read as "nothing was
navigated".

Two derived columns avoid string surgery in queries: `volume` -- the first
path segment of the stub, or NULL when the stub contains no `/` (the
simulated dataset) -- and `image_number`, computed by the existing
`image_number_from_name` logic at ingest time. `results_path_stub` alone
carries a non-unique index. Every consumer lookup filters on
`root_url = <its normalized root>`; `sd_stats_report` gains `--root`
(repeatable, default: all ingested roots), since a report legitimately spans
roots where a pipeline lookup never does.

`images` carries two further non-unique indexes, `image_date` and
`instrument`, which the statistics report groups and filters on across a whole
root. They are carried over from the indexes `cli/stats/schema.py` already
declares, so the report does not lose an index in the move.

The child tables key on the same pair, and each additionally declares the tuple
that identifies one of its rows unique within an image:
`(root_url, results_path_stub, technique_name)` on `techniques` and
`(root_url, results_path_stub, feature_type, source_model, source_name)` on
`feature_sources`. A technique reports once for an image and the feature
inventory is aggregated by exactly that tuple, so a second row is a
contradiction rather than more detail, and a retried or duplicated ingest that
inserted one would change every count and average read from the table without
replacing anything. Each constraint's index leads with the image key, so it is
also the index a lookup by image uses and no separate one is declared.

### 2.3 Columns

The `images` table:

| Column | Source in the document | Notes |
|---|---|---|
| `root_url`, `results_path_stub` | derived (section 2.2) | primary key, NOT NULL |
| `volume` | derived (section 2.2) | NULL for a stub with no separator |
| `image_name`, `instrument` | `observation.*` | NOT NULL; ingest rejects a document lacking either |
| `camera` | `observation.camera` | present whenever the dataset index supplied it, including for an image that never loaded; NULL only when no camera column exists for the dataset |
| `image_path` | `observation.image_path` | |
| `image_et`, `image_date` | `navigation_result.provenance.image_et`, falling back to `observation.image_et` | every image is placed in time even when it never loaded |
| `status` | top-level `status` | NOT NULL; the existing `'unknown'` fallback survives |
| `status_error` | top-level `status_error` | stored **verbatim and separately** from `status_reason` |
| `status_reason` | `navigation_result.status_reason` | the navigator's vocabulary, distinct from `status_error` |
| `offset_dv`, `offset_du` | **top-level `offset`**, unrounded | the authoritative number every consumer applies |
| `sigma_dv`, `sigma_du` | `navigation_result.sigma_px` | |
| `covariance_vv`, `covariance_vu`, `covariance_uu` | `navigation_result.covariance_px2` | the 2x2 offset block only; for a twist-fitted result the rotation row/column of the 3x3 matrix is deliberately not indexed, and `sigma_rotation_deg` is the only twist uncertainty the index carries |
| `sigma_along_unobservable_px` | `navigation_result` | |
| `rotation_deg`, `sigma_rotation_deg` | `navigation_result` | present only where twist was fitted |
| `confidence`, `confidence_rank` | top-level `confidence`, `navigation_result.confidence_rank` | |
| `n_techniques`, `excluded_from_consensus` | `navigation_result` | `n_techniques` NOT NULL |
| `image_class`, `noise_sigma` | `navigation_result.image_classifier` | |
| `image_shape_v`, `image_shape_u` | `observation.image_shape` | |
| `run_start`, `run_end`, `elapsed_s` | `timing` | |
| `config_hash`, `git_sha`, `pipeline_run` | `navigation_result.provenance` | |
| `image_number` | derived from `image_name` | replaces the SQL function (section 2.5) |
| `has_summary_png` | the ingest walk (section 2.7) | Boolean |
| `start_et`, `stop_et`, `exposure_s` | `navigation_result.times.*` | NULL for a document with no navigation result (see below) |
| `sclk_start`, `sclk_midtime`, `sclk_stop` | `navigation_result.times.*` | TEXT |
| `camera_frame_id`, `ck_frame_id` | `navigation_result.pointing.*` | INTEGER |
| `cmatrix`, `cmatrix_original` | `navigation_result.pointing.*` | JSON (nine floats, row-major); `cmatrix` NULL where the navigation fitted a camera rotation |
| `source_file` | the metadata file's URL | provenance |
| `mtime_ns`, `size_bytes` | the metadata file's listing entry | incremental ingest (section 2.7) |

The `times` and `pointing` columns are the corrected-attitude fields the
navigator records (`plans/CK_KERNEL_PLAN.md` section 2.3), and they carry
that section's names and shapes exactly. They are absent from a document
that has no navigation result at all -- an image that failed to load -- and
`cmatrix` is additionally absent where the navigation fitted a camera
rotation, so both cases ingest as NULL.

Types: every pixel, ET, covariance and sigma column is declared
`sqlalchemy.Double` (not bare `Float`, which a dialect may map to single
precision); booleans are `sqlalchemy.Boolean`; `excluded_from_consensus` and
the child tables' `source_names` / `diagnostics` are `sqlalchemy.JSON`
(SQLite TEXT, PostgreSQL `jsonb`). `mtime_ns`, `size_bytes` and `image_number`
are `sqlalchemy.BigInteger`: a nanosecond epoch is far past the 32-bit range a
dialect is free to give a plain `Integer`, and an image-numbering scheme is
free to run past it too.

**Precision.** The top-level `offset` is stored as written, with no
rounding. (It is written unrounded by `navigate_image_files`; the rounded
`navigation_result.offset_px` is the curated display value and is not what
the index stores.) The acceptance test asserts a bit-exact round trip on a
value with 15 significant digits.

**`status_error` and `status_reason` are different vocabularies** and are
stored in different columns. `status_error` takes values like
`missing_spice_data` and is what the SPICE-error selection filter matches
verbatim; `status_reason` is the navigator's explanation of a non-success
outcome. The current ingest merges them into one column; the report queries
that read the merged column are rewritten in the same phase (section 4,
Phase 2) as `COALESCE(status_reason, status_error)`, which reproduces
today's report exactly.

The `techniques` and `feature_sources` child tables keep their current
column sets, re-keyed on `(root_url, results_path_stub)` and constrained to
one row per logical key (section 2.2). The feature inventory is aggregated, as
today, by `(feature_type, source_model, source_name)`; per-feature `feature_id`
and `gated` detail is not retained.

### 2.4 Schema version, creation, and no migrations

The database carries a `schema_meta` table with a single row holding
`schema_version` (an integer) and `created_utc`. The single row is enforced by
a constant primary key (`singleton`) with a `CHECK (singleton = 1)`
constraint, so a second row is a database error rather than an ambiguity the
version gate has to resolve.

Every failure `open_index` reports is a `ValueError` carrying the URL, so a
consumer that wants to report the cause rather than crash catches one type.
The exceptions a database driver raises are translated into that one --
an unparseable URL, an unknown URL scheme, an absent driver, and the ordinary
operational failures of a server that will not accept the connection -- each
keeping the driver's own exception as its `__cause__`. Without the
translation the most common operational failure of all, a PostgreSQL server
that is down or misconfigured, would reach a consumer as a SQLAlchemy
traceback naming neither the URL nor which of the three resolution levels
supplied it.

The translation is a catch-all rather than a list of expected types. A
dialect coerces its own connect arguments and reports a bad one as a bare
`ValueError` naming nothing: a non-numeric port, a non-numeric `timeout`. To
a caller, such an escape is indistinguishable from the guarantee being
broken, so everything that escapes the builder is translated, and only the
messages this layer writes itself pass through untouched.

**The URL a message names is rendered with its password masked.** These
messages are written to run logs and pasted into bug reports, and a database
password belongs in neither. Everything else about the URL survives, because
naming the URL is what tells a reader which of the three resolution levels
supplied the value.

A URL the parser rejected cannot render itself, so its password is found
structurally instead, by a stated rule rather than by a pattern:

1. If the scheme -- the text before the first `:`, with any `+driver` suffix
   and surrounding space removed -- is `sqlite`, the string is returned
   unchanged. A SQLite URL names a local path, which has no credentials and is
   free to carry the colons and at-signs that would otherwise read as some.
2. Otherwise the authority begins after the slashes that follow the scheme
   (one or two, since a hand-edited setting arrives with either). The user name
   runs to the first `:` or `/`; only a `:` introduces a password, and that
   password runs to the first `@` after it, because an `@` is what ends the
   credentials. A `/` inside that span is part of the password, which is what
   reaches a password written with an unescaped slash; an `@` inside the user
   name is kept, which is what reaches the `user@servername` login form of a
   managed server.
3. `host:5432/path@name` is genuinely ambiguous -- a host with a port and a
   path, or a user name with a slashed password. Digits decide it: a port is
   digits and a password is not, and mangling a host, a port and half a
   database name costs the identification these messages exist for.
4. Only the password is replaced. Nothing outside it is touched.

The engine factory is the only opener:

```text
open_index(url: str, *, create: bool = False) -> Engine
```

- `create=True` -- used only by `sd_stats_ingest`'s interactive path and the
  cloud-task enqueuer -- creates the tables and the `schema_meta` row when
  they are absent, and otherwise validates the version.
- `create=False` -- every consumer -- raises when the database or its
  `schema_meta` row does not exist, with a message saying to run
  `sd_stats_ingest` first. A consumer pointed at a nonexistent SQLite path
  fails; it does not create an empty database, which is a deliberate change
  from the current opener's silent-create behavior.
- Either way, a `schema_version` that does not match the version the code
  was built for raises, naming both versions and instructing the reader to
  delete the database and re-ingest. The stamped version is read and checked
  before anything is written, so a refused `create=True` open creates no table
  and writes no row; creating this version's tables inside a database
  stamped with another version would leave a mixture no single version number
  describes. (The database file is not literally untouched: the connect-time
  journal-mode selection rewrites a rollback-journal database's header before
  the gate is reached. What the gate guarantees is the schema and the rows,
  which is what a rebuild would otherwise have to undo.)

There are no migrations. Ingest is cheap relative to navigation and entirely
reproducible from the tree, so rebuilding is always available and always
correct. Any change to the column set, or to the constraints over it,
increments `schema_version`. This replaces the current column-set comparison,
which detects a changed column set but not a changed meaning of an unchanged
column.

### 2.5 Backend selection

All database access goes through **SQLAlchemy Core** -- not the ORM. The
tables are flat and the queries are explicit; an object model would add a
layer without removing one. Existing report SQL moves to `sqlalchemy.text()`
with named binds: the `where_clause` helper's contract changes from
`(fragment with ?, list)` to `(fragment with :name binds, dict)`, and every
call site follows.

The backend is chosen by connection URL:

```text
sqlite:////data/nav-results/index.sqlite3
postgresql+psycopg://user@host/spindoctor
```

A `sqlite:` URL names a **local filesystem path** and is the one path in
this system that does not go through `FCPath` -- SQLAlchemy opens it with
the C library directly. A SQLite URL whose path is on a filesystem that
cannot honor its locking (probed at open with a `BEGIN IMMEDIATE` /
rollback) is refused, naming PostgreSQL as the cross-machine option.

That URL carries **no query string**. The driver derives the filename it
opens from the whole URL, so `.../index.sqlite3?mode=ro` passes an existence
check on `index.sqlite3` and then creates a second file whose name contains
the query -- neither the database the operator named nor an index anything
can read. Such a URL is refused, naming the URL and saying that a SQLite
index URL is a plain local path.

**A read-only database is a separate case from a locking failure**, and the
two are told apart rather than merged. A read-only mount honors locking
perfectly and a consumer cannot corrupt anything, so refusing one and
blaming its filesystem is a false diagnosis of a deployment the index is
meant to support -- an archived copy, or a file shipped to workers on
read-only media.

**Whether ingest can write is asked of the filesystem, not of SQLite**:
`os.access` on the file and on its parent directory, before the engine is
built. SQLite does not answer the question at open. A write-ahead-logged
database -- the shape this opener always leaves behind, having selected that
journal mode -- accepts both the journal-mode selection, which its header
already records, and the write lock `BEGIN IMMEDIATE` takes, on a file it
will never write; it refuses only the first real write, in the middle of an
ingest. A rollback-journal database refuses the journal-mode selection
instead, so no single SQLite answer covers both shapes. The directory is
asked about with the file, because SQLite writes the write-ahead log and its
shared-memory index beside the database, and a writable file in a directory
that permits nothing is still a database ingest cannot write. So:

- `create=False` **accepts** a read-only database and reads it.
- `create=True` refuses it before anything is opened, with a message naming
  read-only -- of the file, or of its directory -- as the cause and saying to
  ingest a writable copy, not the filesystem-locking message.
- A genuine `SQLITE_BUSY` or `SQLITE_IOERR` refuses in both modes, with the
  filesystem-and-PostgreSQL message.

The connect-time journal-mode selection still has to tolerate a refusal (any
`SQLITE_READONLY*` result code), because every connection to a read-only
rollback-journal database would otherwise fail; the refusal is ignored rather
than raised, and nothing is inferred from its absence.

**A probe failure is classified by SQLite's own result code before a message
is chosen.** One exception type covers several unrelated causes, and only one
of them is a reason to move the index to a server. `SQLITE_NOTADB` says the
file is not a SQLite database. `SQLITE_CANTOPEN` names the path and says
which of its causes applies: a directory that does not exist, a path that is
not a file, or a file this user cannot open -- prescribing PostgreSQL for an
operator whose results directory has not been created yet is a wrong remedy
for the most common first-run error there is. Only `SQLITE_BUSY`,
`SQLITE_IOERR` and a code that names no cause at all keep the
filesystem-and-PostgreSQL message. Every other code -- a full disk, a corrupt
file -- is reported as what SQLite said, naming the path and the code and
prescribing nothing, because inventing a remedy for an unclassified failure is
exactly how the wrong one gets prescribed. Codes are matched by prefix, since
SQLite refines several of them into extended forms naming the same cause more
precisely.

One read-only database cannot be read at all: SQLite reads a write-ahead-logged
database through a shared-memory index it creates beside the file, so a
write-ahead-logged copy in a directory that permits no writes is unreadable
by construction. That is refused with a message saying exactly that and to
copy the file somewhere writable, rather than letting a consumer's first
query fail with a write error on a read.

`sqlalchemy` becomes a runtime dependency in `[project] dependencies`. The
PostgreSQL driver ships as an optional extra, `rms-spindoctor[postgres]`,
declaring `psycopg[binary]`; a PostgreSQL URL with the driver absent must
fail with a message naming the extra to install, not with an import
traceback. SQLAlchemy's own internal use of stdlib `logging` is left
untouched and unconfigured; nothing in `spindoctor.results_index` may call
`logging.getLogger` or enable engine echo.

Constructs that are SQLite-only and must not survive into the Core layer:

- **`image_number` as a Python UDF.** `register_image_number_function`
  (defined in `report_common.py`, registered by `report.py`) installs a
  Python callable that queries then call in `WHERE` clauses. It becomes the
  ingested `image_number` column; the queries compare against the column.
- **`TOTAL(...)`** (in `report_sections.py`) becomes
  `COALESCE(SUM(...), 0)`.
- **Integer arithmetic on booleans.** `t.spurious = 0` becomes
  `NOT t.spurious`; `SUM(1 - t.spurious)` becomes
  `SUM(CASE WHEN t.spurious THEN 0 ELSE 1 END)`. Both current spellings are
  type errors under a native PostgreSQL boolean.
- **`executescript` and `PRAGMA`.** DDL is emitted by SQLAlchemy metadata.
  `PRAGMA foreign_keys = ON`, WAL, and `busy_timeout` become SQLite-dialect
  connect-time events.

### 2.6 Configuration and the command-line surface

A new `environment.results_db` configuration key holds the connection URL.
Resolution follows the pattern the other roots use, in
`spindoctor/config/config_helper.py`: the command-line value, then
`config.environment.results_db`, then the `NAV_RESULTS_DB` environment
variable. Absence is not an error -- it means "no index", the default mode
of every program. Add `get_results_db_url(arguments, config) -> str | None`
beside the existing getters.

Every consuming program accepts `--results-db URL`, and also
`--results-db none`: the literal sentinel `none` resolves to no index,
overriding the configuration key and the environment variable. Without an
explicit opt-out, an exported `NAV_RESULTS_DB` would make file-mode runs
impossible on that machine. The sentinel is recognized at whichever level
supplied the value, so a configuration file or an exported variable can opt out
the same way; it is matched as the exact string, so a URL that merely contains
the word is still a URL.

The codebase convention for an argument of this kind is that each program
defines its own, as it does for the results roots: the reprojection family
shares `add_common_env_args` in `spindoctor/cli/reproj/args.py`, and that is
the only grouping.

A program that resolves a URL and cannot open it fails immediately with that
error; it does not silently fall back to reading files. Falling back would
turn a misconfigured run into a slow, silently different one.

The `environment` section joins `HASH_EXCLUDED_SECTIONS` in
`spindoctor/config/config.py`, alongside `logging`: it holds deployment
locations (roots, and now a database URL), none of which can change a
navigation result, and a provenance hash that shifts when an operator moves
a results directory answers its question wrongly. This changes the stamped
`config_hash` once for operators who set `environment` keys in
configuration files; the change is provenance-only and is called out in the
PR body.

### 2.7 Ingest

`sd_stats_ingest` walks each root for `*_metadata.json`, reads each
document, and upserts its rows.

**One walk feeds everything.** The recursive listing collects both
`*_metadata.json` and `*_summary.png` names in a single pass (the pattern
`ResultsFilter._scan_volumes` already uses), so `has_summary_png` comes from
the walk, and each metadata file's `mtime_ns` and `size_bytes` come from the
same listing entries. There is no per-file `stat` call and no per-file
`exists` call; a backend whose listing does not supply size and mtime
degrades to `--force` behavior for that root, with a logged warning.

**Incremental.** A file whose `(mtime_ns, size_bytes)` matches the stored
pair is skipped without being read. `--force` re-reads everything.

**Per-root bookkeeping.** An `ingest_runs` table records, per root:
`root_url`, `started_utc`, `finished_utc` (NULL while running),
`files_seen` / `files_ingested` / `files_skipped` / `files_failed`, and
`schema_version`, under a surrogate `run_id` primary key (a root legitimately
has many runs, and a consumer reads the newest). The row is written at
start and updated at completion, in both the interactive and cloud paths. A
consumer treats a root whose newest row has `finished_utc IS NULL` -- or no
row at all -- as not ingested, and fails with a message saying so.

**Batched retrieval.** Metadata files are retrieved in batches through
`FCPath.retrieve()` with `exception_on_fail=False`, the pattern
`ResultsFilter.filter_batch` uses. The current code reads via
`get_local_path()`, which does not download on a cloud root and must be
replaced. Batch and chunk sizes are module constants,
`INGEST_RETRIEVE_BATCH_SIZE = 64` and `INGEST_COMMIT_CHUNK_SIZE = 512`,
independent of one another.

**Chunked transactions.** Commit every `INGEST_COMMIT_CHUNK_SIZE` images.
A crash costs one chunk, and concurrent writers are not serialized behind a
run-length lock.

**Per-image atomicity.** The delete-then-insert upsert for one image and its
child rows happens inside one transaction, so concurrent workers can never
interleave halves of one image's rows.

### 2.8 Cloud-task ingest

`sd_stats_ingest_cloud_tasks` mirrors the structure of the existing
cloud-task drivers (`spindoctor/cli/sd_backplanes_cloud_tasks.py` is the
reference): a `process_task(task_id, task_data, worker_data)` returning
`(retry, result)`, a `Worker` started from `async_main`, and cloud-task
logging isolation. Like every cloud-task driver, it adds **no** logging
arguments (and is added to the `_CLOUD_TASK_DRIVERS` list in
`tests/spindoctor/cli/test_logging_argument_surface.py`, whose test asserts
exactly that). A cloud ingest worker has no run log and no per-image scope;
its outcome -- counts of ingested, skipped, and failed files, with the
failing files named -- is **returned in the task result**, which the
enqueuer aggregates.

A task carries a list of metadata-file URLs or a stub prefix; the worker
ingests its share.

**Concurrency:**

- **Workers on one machine, SQLite backend.** They open the same local file.
  WAL, `busy_timeout`, and short transactions (section 2.7) make multiple
  local writer processes safe. There is one file, so there is no merge step
  and none must be written.
- **Workers across machines.** A shared SQLite file is not an option. They
  connect to PostgreSQL as ordinary clients. This is the case the backend
  abstraction exists for.

**Schema creation is not per-task.** The enqueuing program opens with
`create=True` before fan-out and writes the `ingest_runs` start row; workers
open with `create=False` and fail if `schema_meta` is absent. Per-task
counts return in task results and are aggregated and written to
`ingest_runs` by the enqueuer at completion; workers never touch
`ingest_runs`.

### 2.9 Consumers

Each consumer gains one code path behind the resolved URL; its existing path
is untouched and remains the default.

**`sd_stats_report`** (with ingest, Phase 2). Moves onto the Core layer and
the versioned schema. Its report must not change: the failure-reason tables
that read the previously merged reason column move to
`COALESCE(status_reason, status_error)`; every join and correlated subquery
(including the four in the CSV export) moves to the composite key; the CSV
export's `images.csv` columns become the section 2.3 set in schema order,
`root_url` through `size_bytes` included, and the user-guide column list is
updated to match. `--db` is **removed** from both statistics programs and
replaced by `--results-db` (a URL, not a bare path); the four documented
invocations in `docs/user_guide/user_guide_statistics.rst` change with it,
and that page's "plain SQLite" framing is rewritten around the URL scheme
with each direct-SQL example shown in both dialect spellings where they
differ (`json_each` vs `jsonb_array_elements_text`).

**`sd_backplanes`.** Reads `status`, `status_error`, and the top-level
`offset` for one stub. With an index it reads one row. A stub absent from
the index raises, exactly as a missing metadata file raises in the file
path -- the caller reports it -- with a message naming the stub and the
index URL.

**Reprojection offset lookup.** `load_offset_if_any` in
`spindoctor/cli/reproj/offsets.py` is the highest-volume reader in the
system. The seam is made explicit rather than overloaded: an `OffsetSource`
protocol with two implementations,

```text
FileOffsetSource(nav_results_root)          # today's body, unchanged
IndexOffsetSource(engine, root_url)         # one SELECT per lookup
```

both returning the existing `OffsetLookup`. Drivers construct one according
to the resolved URL and pass it where they pass `nav_results_root` today
(`generate_backplanes_image_files` and the mosaic pass take the source
object in place of the root-plus-implicit-reader they take now).

The reason vocabulary maps as follows, and the mapping table belongs in the
module docstring:

| File-path reason | Index-path equivalent |
|---|---|
| `no_metadata` | no row for the stub |
| `navigation_did_not_succeed` | row with `status != 'success'` |
| `null_offset` | row with `status = 'success'` and `offset_dv IS NULL` |
| `unusable_metadata_path` | unreachable: a stub is a key, not a path |
| `unreadable_metadata`, `invalid_json`, `metadata_not_an_object` | unreachable: ingest already refused such a file, so it has no row (surfaces as `no_metadata`) |
| `invalid_offset_type`, `non_finite_offset`, `malformed_offset` | unreachable: ingest coerces a malformed or non-finite offset to NULL (surfaces as `null_offset`) |

The last two rows are a real behavioral difference between the paths -- a
malformed document reports a different reason depending on the mode -- and
the docstring states it rather than papering over it. The index path logs
the same per-image `IMAGE_LOGGER` warnings for the three reachable reasons,
with the same message shapes.

**`ResultsFilter`.** When a URL is given, the presence, absence, and error
filters become one query per enumeration instead of a walk per volume plus
batched reads -- preserving the exact semantics of both existing modes (the
walked-set mode and the absence-only batched-`exists()` mode) and every
contradictory-pair rejection in the constructor. `ResultsFilter` lives in
`spindoctor.dataset`, which `sd_offset` imports on every run, so the
index-backed implementation lives in `spindoctor/results_index/selection.py`
and `results_filter.py` imports it **inside the branch where a URL was
given**. This is a recorded exception to the top-of-file import rule,
justified the same way the GUI imports are: it keeps SQLAlchemy off the
navigation critical path, and it is noted in the dev guide.

### 2.10 Logging

`sd_stats_ingest` gains a program identity and a logger. It becomes
infrastructure other programs depend on, and a partial or failed ingest must
appear in a run log rather than only in an exit code.

Concretely:

- Add `SD_STATS_INGEST = 'sd_stats_ingest'` to
  `spindoctor/config/program_names.py` and `PROGRAM_NAMES`; both drivers
  declare `PROGRAM_NAME`.
- The interactive driver adds
  `add_logging_arguments(parser, has_image_logger=False)` (ingest processes
  documents, not images) and wraps its configuration load in
  `reporting_logging_errors()` with the exact
  `with reporting_logging_errors():` / `load_default_and_user_config`
  adjacency -- a source-scanning test asserts that literal pattern. The
  cloud-task driver adds no logging arguments (section 2.8).
- `print()` calls in ingest become `MAIN_LOGGER` calls.
- Collateral that asserts the old exclusion, all updated in the same
  commit: `sd_stats_ingest` moves from `_WITHOUT_LOGGER` to
  `_WITH_ANY_LOGGER` in
  `tests/spindoctor/cli/test_logging_argument_surface.py`; the program
  table in `docs/user_guide/user_guide_logging.rst`; the `program_names.py`
  module docstring; and `.cursor/rules/logging_nav.mdc` section 1, whose
  statistics-programs sentence is replaced so that it names
  `sd_stats_report` alone as the print-only statistics program (the
  license for `sd_stats_report`'s `print()` must survive the edit).

`sd_stats_report` keeps `print()`. Its output *is* terminal text for a human
reading a report; a logger would wrap that in machinery it does not need.
The docstring says so, so the split reads as a decision rather than an
oversight.

---

## 3. Current state

What exists, and what each phase has to change.

**`spindoctor/cli/stats/schema.py`** defines three tables via a `_SCHEMA`
string applied with `executescript`. `images` is keyed on `image_name`
alone, so two images with the same basename in different volumes silently
overwrite each other. `open_stats_db` compares `PRAGMA table_info(images)`
against a frozen column set and raises with delete-and-re-ingest
instructions on a mismatch -- and silently creates a fresh database at a
nonexistent path, which section 2.4 deliberately changes for consumers.
`upsert_image` issues DELETE + INSERT with transaction management left to
the caller.

**`spindoctor/cli/stats/ingest.py`** flattens a document in
`rows_from_metadata`. It reads the rounded `navigation_result.offset_px`
rather than the top-level `offset`; merges `status_reason` and
`status_error` into one column; and drops `rotation_deg`,
`sigma_rotation_deg`, `covariance_px2`, and `sigma_along_unobservable_px`.
(The feature inventory is not dropped: it is aggregated into
`feature_sources`, and stays that way.) `ingest_metadata_files` wraps the
entire multi-root scan in a single transaction, has no incremental logic,
and reads files via `get_local_path()`, which does not download on a cloud
root. `main_ingest` takes `--db` defaulting to `nav_stats.sqlite3` in the
working directory, bypassing the configuration system, and reports via
`print()`.

**`spindoctor/cli/stats/report_common.py`** defines the `image_number`
Python UDF and a `where_clause` helper returning `?`-style fragments;
`report.py` registers the UDF and, with `report_sections.py`, issues the
`status_reason` taxonomy queries, the boolean integer arithmetic, the
`TOTAL(...)` aggregate, and `image_name`-keyed joins that all change per
sections 2.3 and 2.5. `report.py` is 782 lines; if the migration pushes it
past the 1000-line cap it splits into a package in the same commit.

**`spindoctor/dataset/results_filter.py`** implements the selection filters
with two distinct modes: presence/error filters walk the results tree once
per selected volume; absence-only filters skip the walk and use batched
`exists()` calls driven from `dataset_pds3.py`'s 64-image batching loop.
`_metadata_matches` requires `status == 'error'` and compares `status_error`
against `missing_spice_data` verbatim -- the behavior the split columns
preserve.

**`spindoctor/cli/backplanes/backplanes.py`** reads one document per image
by stub; a missing file raises and the caller reports it. (The module has no
docstring; Phase 4 adds one, which is also where the reason-mapping note
lives.) **`spindoctor/cli/reproj/offsets.py`** reads one document per image
and returns `OffsetLookup` with the ten-reason vocabulary of section 2.9.
**`spindoctor/cli/pds4/bundle_data.py`** serializes the whole document into
the supplemental product, which is why it is out of scope.

SQLAlchemy is not currently installed or declared.

---

## 4. Implementation phases

Each phase is one pull request against `rf_results_index`, and each must
leave `main`-equivalent behavior intact for every program not given a
results-db URL.

### Phase 1 — Core layer, schema, and version gate

Introduce `spindoctor/results_index/` as a library package (not under
`cli`; library consumers use it). Define the SQLAlchemy Core metadata for
`images`, `techniques`, `feature_sources`, `schema_meta` and `ingest_runs`
per sections 2.2-2.4; `open_index` with the `create` flag and version gate;
the SQLite dialect events (WAL, `busy_timeout`, foreign keys, the
lockability probe); and the missing-driver message for PostgreSQL URLs.

Declare `sqlalchemy` in `[project] dependencies` and the `postgres` extra in
`[project.optional-dependencies]`. Add `environment` to
`HASH_EXCLUDED_SECTIONS` with its rationale comment and test.

Tests: schema creation against SQLite; `create=False` raising on a missing
database, a missing `schema_meta` row, and a version mismatch (each message
asserted); a refused `create=True` open creating no table; the
`ValueError` guarantee on each route a driver exception takes (an absent
driver, an unparseable URL, an unknown URL scheme, a non-numeric port that
reaches the caller as a bare `ValueError`, a server that refuses the
connection, and an unexpected failure inside the engine factory), with the
driver's exception kept as the `__cause__` on each; a password absent from
every one of those messages while the rest of the URL survives, including on
the `schema_meta` and version gates and against a server that rejects the
password (`postgres` tier); the structural masking rule as a table of URLs and
exactly what masking each produces -- a slashed password, an at-sign in the
user name, a leading space, a hyphenated scheme, a two-line copy, and the
non-credential strings it must leave alone (a port with a later at-sign, a
user name with no password, a SQLite path, a scheme alone, the empty string)
-- with every credential-bearing row driven through `open_index` itself and
not only through the helper; the missing-driver message, with the driver
hidden by an import hook rather than by its happening to be absent from the
environment; the read-only cases of section 2.5 (accepted by a consumer,
refused by an ingest, and a write-ahead-logged copy that cannot be read),
each parametrized over both journal modes, plus an ingest into a read-only
directory, alongside a genuine lock failure in both modes; the result-code
classification of a probe failure (a file that is not a database, a path that
is a directory, a directory that does not exist, a file this user cannot
open, an exception carrying no result code at all, one carrying a code that is
not text, and an extended form of each classified family, since matching by
equality rather than by prefix loses the family's remedy); the connect handler
re-raising a driver error that is not a read-only refusal; a `sqlite:` URL with
a query string refused without leaving a file behind; the connect-time
settings read from a second connection held open beside the first, so a
one-shot application fails the test; a refused open disposing the pool it
built; Double-precision round-trip of a 15-significant-digit value; boolean
round-trip; a duplicate child row refused on both backends; the config-hash
exclusion.

The engine tests are three files, because one would run past the 1000-line
module cap: the opener's contract (`test_engine.py`), what it does with a
SQLite file (`test_engine_sqlite.py`), and how it names a URL without naming
its password (`test_engine_masking.py`).

### Phase 2 — Ingest and reporting onto the index

One PR, because the report reads the table ingest writes and neither is
usable mid-cutover.

Ingest: rewrite `rows_from_metadata` to the section 2.3 column set and
`ingest_metadata_files` per section 2.7 (single walk feeding presence, stat
and file list; batched retrieval replacing `get_local_path()`; incremental
skip; chunked transactions; per-image atomicity; `ingest_runs`). Root
resolution and normalization per section 2.2. The driver gains the
configuration surface (`--results-db`, `environment.results_db`,
`NAV_RESULTS_DB`, the `none` sentinel), the logging surface, and the program
identity, with every collateral edit of section 2.10.

Reporting: move `sd_stats_report` onto the Core layer per sections 2.5 and
2.9 -- named binds, composite-key joins, `COALESCE` for both the reason
taxonomy and `TOTAL`, boolean spellings, the column-backed `image_number`,
`--root`, and `--db` removed from both programs.

The source scan that enforces criterion 10 reads only
`src/spindoctor/results_index/`, which is the whole of the Core layer while
that is all there is. The statistics programs move onto the Core layer here,
so this phase widens the scan's `_SOURCE_ROOT` to cover
`src/spindoctor/cli/stats/` as well, and updates the test that pins which
modules the scan reaches. Left alone, criterion 10 would report green over a
package that no longer holds the queries it exists to check.

Tests: a two-volume tree with colliding basenames producing two rows; a
bare-basename stub with NULL `volume`; the unrounded offset; the separated
status vocabularies; an unchanged file skipped on a second ingest (proven by
counting retrievals *and* asserting no per-file stat call) and a touched one
re-read; `--force`; a chunk boundary crossed mid-run; `ingest_runs` written
at start and completed at end; a malformed document counted as an error
without aborting; a cloud-style root actually downloading (the test fails if
`retrieve()` reverts to `get_local_path()`). For the report: byte-identical
report and CSV output (modulo the documented new CSV columns) against the
same fixture tree ingested by the old and new ingest, which is what proves
the `COALESCE` rewrite; and a PostgreSQL run of the same assertions under
the `postgres` marker.

### Phase 3 — Cloud-task ingest

`sd_stats_ingest_cloud_tasks` per section 2.8, entry point in
`pyproject.toml`.

Tests: enqueuer creates the schema and workers refuse to; concurrent local
SQLite workers produce the same rows as a serial ingest; per-task counts
aggregate into `ingest_runs`; the worker writes zero bytes to stdout and
stderr from SpinDoctor code, asserted at file-descriptor level as the
existing cloud-task silence tests do; the driver appears in
`_CLOUD_TASK_DRIVERS` and passes the no-logging-flags assertion.

### Phase 4 — Backplanes and reprojection consume the index

The `OffsetSource` protocol and both implementations; `--results-db` (and
the `none` sentinel) on `sd_backplanes`, `sd_mosaic`, and their cloud-task
variants; the backplane single-row read with the missing-stub raise; the
root-url comparison failure of section 2.2.

Unit tests at the `OffsetLookup` level over a fixture tree: each reachable
reason with and without an index, including `null_offset` from a
success-with-NULL row; the unreachable-reason remapping of section 2.9
asserted (a malformed-offset document yields `malformed_offset` via files
and `null_offset` via the index); the same `IMAGE_LOGGER` warnings in the
index path; a program handed an unopenable URL failing rather than falling
back; a consumer refusing a root with no completed `ingest_runs` row.
Integration tests (marked `integration`): identical backplane and mosaic
products for the same images with and without an index, asserted on the
outputs.

### Phase 5 — Selection filters

`spindoctor/results_index/selection.py` and the `ResultsFilter` branch-local
import per section 2.9.

Tests: for every filter flag, both existing modes (walked and
absence-only-batched) against the index-backed answer over a fixture tree;
every contradictory-pair rejection unchanged; an import-time assertion that
`import spindoctor.dataset` does not import `sqlalchemy` (this is the
criterion 2 test).

### Phase 6 — Documentation

New `docs/user_guide/user_guide_results_index.rst` (added to the
`user_guide.rst` toctree): what the index is and that it is optional; the
snapshot semantics and staleness; the URL forms, the `postgres` extra, and
the SQLite-is-local rule; the ingest workflow, including that it is never
automatic; root normalization and the not-ingested failure; the version gate
and delete-and-re-ingest; which programs consume the index, that
`sd_stats_report` requires it, and which programs deliberately do not use
it. `docs/api_reference/api_results_index.rst` -- created with the package in
Phase 1, so the branch never carries an undocumented public package -- gains
the modules the later phases add. Updates: `user_guide_statistics.rst` (per section 2.9),
`user_guide_logging.rst` (program table), `introduction_configuration.rst`
(`environment.results_db`), and a dev-guide section covering the Core
layer, the concurrency model, the branch-local import exception, and how to
add a column (increment the version). No issue numbers in any of it.

---

## 5. Acceptance criteria

1. `sd_backplanes`, `sd_mosaic`, and the `ResultsFilter`-driven selections
   produce identical products and identical selections for the same inputs
   with and without an index, over a fixture tree exercising success,
   failure, error, missing-metadata and malformed-metadata images. Asserted
   by tests (unit tier at the `OffsetLookup`/selection level; integration
   tier on written products). "Identical" binds returned values, written
   products, and the reachable-reason warnings -- not incidental log text.
   `sd_stats_report`'s criterion is section 4 Phase 2's old-vs-new
   byte-identical report.
2. No pipeline program requires an index, and `import spindoctor.dataset`
   (the `sd_offset` critical path) does not import `sqlalchemy`. Asserted by
   a test.
3. A program given an unopenable, nonexistent, or version-mismatched index,
   or a root the index has not fully ingested, fails with a message naming
   the cause; it does not fall back to reading files and does not create an
   empty database.
4. Two images with the same basename in different volumes produce two rows
   and are independently retrievable; a multi-root index serves each
   consumer only rows from its own normalized root.
5. The stored offset round-trips the document's top-level `offset`
   bit-exactly at 15 significant digits, and `status_error` is retrievable
   verbatim.
6. A second ingest over an unchanged tree reads no metadata file and issues
   no per-file stat or exists call, proven by counting.
7. Ingest on a cloud-style root downloads its files; the test fails if the
   download call is reverted to `get_local_path()`.
8. Concurrent local SQLite ingest workers produce the same rows as one
   serial ingest over the same input.
9. `sd_stats_ingest_cloud_tasks` writes zero bytes to stdout and stderr
   from SpinDoctor code under a worker subprocess, and its counts arrive in
   the task result.
10. No SQLite-only construct remains in any query: no UDF registration, no
    `TOTAL(`, no `executescript`, no `PRAGMA` outside a dialect event, no
    integer comparison or arithmetic against a Boolean column. Enforced by a
    source-scanning test, whose root covers every package holding index
    queries: `src/spindoctor/results_index/`, and from Phase 2 also
    `src/spindoctor/cli/stats/`, where the report's queries live. A scan
    aimed at one package while the queries live in another reports green
    without reading them.
11. The suite's index tests pass against PostgreSQL under a `postgres`
    marker: registered in `[tool.pytest.ini_options].markers`, excluded by
    default via `addopts` (`-m "not integration and not postgres"`), given a
    `scripts/run-all-checks.sh` flag alongside `-i` (`-P` / `--postgres`), and
    named in the dev guide as a locally-runnable tier if CI has no service
    container.
12. `ruff check`, `ruff format --check`, `mypy --strict`, `sphinx-build -W`
    and `pymarkdown scan` all pass; suite coverage stays at or above 90%.

---

## 6. Risks and constraints

**The snapshot can be stale, by design.** A consumer reading an index that
predates the newest navigation run gets the older answer with no warning.
The mitigation is documentation and the `ingest_runs` timestamps, not
machinery. Anyone tempted to add staleness detection later should note that
it reintroduces per-image tree access, which is the cost this work removes.

**Absence is ambiguous unless bookkeeping is right.** "No row for this stub"
means "not navigated" only if the root was fully ingested. That is what
`ingest_runs` is for, and it is load-bearing rather than cosmetic -- hence
the hard failure on a root with no completed row, and the root
normalization rule, without which the comparison generates false failures.

**One writer assumption in SQLite.** WAL plus short transactions makes
multiple local writers safe; it does not make a shared network filesystem
safe. The cross-machine case is PostgreSQL, and the lockability probe
refuses the SQLite URL rather than corrupting the file.

**A new required dependency.** SQLAlchemy is imported by the statistics
programs and by `results_index` whether or not an index is used, and by
nothing on the navigation critical path -- criterion 2 pins that.

**Schema changes cost every user a rebuild.** That is the accepted trade for
having no migration machinery, and the reason the schema carries the
corrected-pointing columns in the shapes their producer writes rather than
leaving them for a later migration.

**The report is the regression surface.** Phase 2 touches every query the
statistics user guide documents. The old-vs-new byte-identical report test
is the only thing standing between "migrated" and "silently different";
treat a diff there as a defect, not as an acceptable drift.

---

## 7. Follow-ups

File as tracking issues alongside the implementation issue:

- **A document column for bundle generation.** Storing the raw metadata JSON
  per image (SQLite TEXT, PostgreSQL `jsonb`) is the only way bundle
  generation stops reading one document per image, since it needs the whole
  document. It roughly doubles the index size, which weakens shipping the
  SQLite file to workers, so it likely belongs in a separate optional
  database rather than the main index.
- **Serving the curation triage tool.** Needs `missing_frac`,
  `saturation_frac`, classifier flags, the per-technique list, result file
  paths, a name-to-stub lookup, and a second (rescue) root -- a schema
  widening for one out-of-package tool, wanted only if triage's ten
  rglobs-per-frame remain a practical pain after the pipeline consumers are
  converted.
- **Shipping the index to cloud workers.** Publishing the SQLite file to the
  results bucket and having each worker download it once is the alternative
  to a PostgreSQL instance, and needs a documented workflow either way.
- **A `--since` selector for ingest.** The stat-pair skip makes a re-scan
  cheap in reads but not in listings; a time-bounded scan would cut the
  listing too.
- **The lockability probe takes a write lock on a consumer's open** (#462).
  Section
  2.5 has it refuse in both modes, so a consumer opening a SQLite index while
  an ingest holds a write transaction waits out the busy timeout and can then
  fail with the filesystem-and-PostgreSQL message though nothing is wrong. It
  is the plan's own rule and is left as written here; whether a `create=False`
  open should probe with a read instead belongs with the concurrent-ingest work
  of Phase 3, which is what makes the collision likely.

---

## 8. Execution protocol

1. Branch `rf_results_index` off current `main`; one commit series per
   phase.
2. Per phase: dispatch an **implementer subagent** (Opus-class) whose prompt
   embeds that phase's section of this plan verbatim plus sections 1-3, so
   the subagent needs no other briefing and does not have to locate this
   file. Then dispatch an **independent, fresh-context adversarial
   reviewer** (also Opus-class) with the diff, the same plan sections, and
   instructions to (a) verify each normative statement of section 2 against
   the code line by line, (b) run the phase's tests plus
   `ruff check src tests`, `ruff format --check src tests`, and
   `mypy src tests`, (c) confirm the no-index path is genuinely unchanged,
   and (d) hunt for convention violations and unstated deviations from this
   plan. Fix rounds until the review is clean; the controller, not the
   implementer, judges cleanliness.
3. The reviewer must verify each guarantee by **breaking the source and
   confirming a test fails**. A test that passes against a deliberately
   broken implementation is a defect in the test, and is reported as one.
4. Deviations discovered mid-phase -- an API that does not exist, a wrong
   assumption about a consumer -- are recorded in the phase commit message
   and reconciled into this plan file in the same commit, so the document
   the next reviewer holds is never stale. Scope changes go to the operator
   instead.
5. Final sweep before the pull request to `main`:
   `./scripts/run-all-checks.sh -i` plus the `postgres` tier, and a run of
   every consuming program over a real fixture tree with and without an
   index, with every output difference accounted for.
6. One pull request to `main`: summary, phase map, evidence, `Closes` per
   issue, and the plan and guide reconciliation included.
