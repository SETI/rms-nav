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

**Absent is not empty, in every JSON column.** The JSON type is declared
`none_as_null=True` (and its PostgreSQL variant likewise), so a Python `None`
is stored as SQL NULL rather than as the JSON value `null`. Without that,
`WHERE cmatrix IS NOT NULL` matches every row ever written and the CSV export
carries the literal text `null` in the cell. Which columns are ever absent is
a property of the mapping rather than of the type: `cmatrix` and
`cmatrix_original` are, and are NULL. `excluded_from_consensus`, `source_names`
and `diagnostics` are not -- an empty list or object there is a statement
(nothing was excluded, the technique named no source, it reported no
diagnostics), and is stored as the empty container.

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

**The URL a message names is rendered with its credentials masked.** These
messages are written to run logs and pasted into bug reports, and a database
password belongs in neither. Everything else about the URL survives, because
naming the URL is what tells a reader which of the three resolution levels
supplied the value.

One structural rule does this for every URL, not only for the ones the parser
rejects. A parsed URL renders itself with its password hidden but renders a
`?password=` query parameter verbatim, and that parameter authenticates exactly
as the authority form does; adopting the parser for the URLs it accepts would
adopt its blind spot with it. The rule is stated rather than pattern-matched:

1. If the scheme -- the text before the first `:`, with any `+driver` suffix
   and surrounding space removed -- is `sqlite`, the string is returned
   unchanged. A SQLite URL names a local path, which has no credentials and is
   free to carry the colons, at-signs and question marks that would otherwise
   read as some.
2. The authority begins after the scheme's `:` **only when a `/` follows that
   colon**, and then after **every** slash of that run, however many there are.
   Without that slash, `postgresql:svc:pw@host/db` and `svc:pw@host/db` are the
   same shape, and reading the leading word as a scheme leaves the second one's
   password visible; it is read as a user name in both, which hides the first
   one's user name along with its password. That is one word of a message about
   a URL no driver would have accepted; the other reading loses a working
   password. One slash and two are what a hand-edited setting arrives as, three
   is `postgresql:///db` omitting the host to name a local socket, and a rule
   that stopped counting at two left the authority start on a slash -- which
   reads as a path beginning before any password and returns such a URL whole.
3. Within the authority the user name runs to the first `:`; only a `:`
   introduces a password. The credentials end at the **last** `@` in the
   string. A user name is free to carry one -- `user@servername` is the login
   form of a managed server -- and so is a password: `p@ssword`, `p@ss/word`
   and `pw:with@both` are all things an operator types. Every narrower bound
   stops inside a password that carries the character it stops at: ending at
   the `@` before the first `/` leaves the tail of `p@ss/word` in the message,
   and ending at the last `@` before a `#` leaves the tail of `pw@part#rest`.
   The last `@` is the only bound that cannot stop early, because the span it
   produces contains every other candidate span. What it costs is over-masking
   a URL whose credentials end sooner and whose tail carries an `@` -- a
   fragment such as `...?password=x#note@host` is masked to its last character
   -- which is a mangled message about a URL no driver accepts, against a
   working password in a run log. If the first `/` precedes the first `:`, the
   authority ended before any colon and there is no password.
4. `host:5432/path@name` is genuinely ambiguous -- a host with a port and a
   path, or a user name with a password that carries a slash. It is read as
   credentials, which is how the URL parser itself reads it: for the spelling
   of that shape a parser accepts, this rule and `render_as_string()` hide the
   same characters, and a test asserts that agreement over a hand-picked set of
   URLs both can read, which is the only comparison available and cannot reach
   the unparseable shapes where this rule is the sole defense. Reading it as a
   port instead -- deciding by whether the text before the slash is digits --
   leaves a password that opens with digits, `123/secret`, visible in full,
   which is the failure this rule exists to prevent; the cost of the reading
   taken is a mangled host and database name in a message about a URL that was
   already unusable.
5. A query parameter whose name carries `password`, `passwd`, `pwd`, `secret`,
   `token` or `credential` has its value replaced. A driver accepts any
   parameter its library knows, so the name is matched by what it contains
   rather than against a fixed list: over-hiding a setting whose name says
   credential costs a word of a message, and under-hiding one puts a working
   password in a run log. One parameter is separated from the next by `&` or by
   `;`, both of which libpq and the drivers built on it accept, so both are
   split on and each is put back as it was written.
6. Only credential material is replaced. Nothing outside it is touched.

The rule is asked about a **corpus** rather than a list of remembered shapes:
every combination of scheme, of the slashes after it, of credentials, and of
what a password may contain, asserted in both directions -- the secret is gone,
and the result is exactly the URL with its credentials replaced.

Each dimension of that corpus is **covered rather than sampled**, which is not
the same thing and is where the leaks kept living. A corpus carrying no slash,
one and two stopped one value short of the three-slash spelling; a corpus
varying one special character of a password at a time could not reach a
password carrying an at-sign *and* a slash. So the slash count runs from none
to four, and the passwords carry every ordered pair of the characters that mean
something to a URL -- order included, since `p@ss/word` and `pw/part@rest` stop
a rule reading by eye at different places. Tests assert that the dimensions are
still crossed, because a corpus quietly narrowed proves less than it says.

The rule is a named function of the Core layer rather than a private helper of
the opener, because a run log records the command line a program was given and
one of those words can be a connection URL. `sd_stats_ingest` masks the value of
`--results-db` in the command line it logs, in both spellings argparse accepts.

**A results root is never masked.** It is not a connection URL, it has no
credentials to hide, and it is the one string an operator reads a run log to
correct, so the ambiguity rule 4 accepts would corrupt it for nothing. Every
message that names both -- `require_ingested_roots`'s refusal above all -- masks
the index URL and prints the roots as they were given.

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
- `create=True` asks the filesystem first, and a database or directory that
  `os.access` reports unwritable is refused there, before an engine is built,
  with a message naming read-only -- of the file, or of its directory -- as the
  cause and saying to ingest a writable copy, not the filesystem-locking
  message. A path the filesystem permits still passes through the engine build
  and the open probe, and a refusal from either is diagnosed by what the driver
  said, including the `SQLITE_READONLY*` a rollback-journal database gives to
  the journal-mode selection.
- A genuine `SQLITE_BUSY` or `SQLITE_IOERR` refuses in both modes, with the
  filesystem-and-PostgreSQL message. A refusal carrying no result code at all
  is neither, and is reported as what SQLite said with no remedy prescribed for
  it: the locking remedy is a deployment rebuild, and answering an
  unclassifiable failure with one is a false diagnosis.

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
pair, and beside which the walk saw the same summary PNG the row records, is
skipped without being read. The summary flag is part of the comparison because
it comes from the walk rather than from the document: a PNG written after the
document was ingested changes the row that ought to be stored while changing
nothing about the document. `--force` re-reads everything.

**A refused file is bookkeeping, not a row.** A file that is not a
current-schema navigation document is recorded in a `failed_files` table --
`root_url`, `results_path_stub`, `reason`, `mtime_ns`, `size_bytes` -- and is
skipped on the next pass on the same evidence as an ingested one, so a tree
whose non-navigation files outnumber its results does not pay to download and
parse every one of them on every run. It is a table of its own rather than a
marked `images` row: absence of an `images` row is what every consumer reads as
"this image was never navigated", and a file with no usable data must leave
that answer alone. `--force` re-reads a refused file too. A document that
ingested on an earlier pass and no longer reads has its `images` row deleted as
the refusal is written, since a row nothing backs would answer for an image
nothing produced; and a file that was refused and now reads has its refusal
deleted as its rows are written.

**A root the walk cannot list is not an empty root.** The walk reports whether
the root itself could be listed. When it could not -- a mistyped root, an
unmounted share -- the run's `ingest_runs` row keeps its NULL finish time, so
every consumer treats the root as one nobody has ingested rather than one that
holds nothing, and no row of it is touched. A root that exists and is genuinely
empty completes normally.

**Every way a directory refuses to be listed is one way.** The walk treats any
`OSError` as "could not be listed": not there, not a directory, unreadable by
this user, a share that stopped answering. A permission error is the commonest
of them on a shared tree, and enumerating only some of the others ends the pass
on it -- skipping every later root of a multi-root run and never reaching the
closing summary. A directory that could not be listed costs the files under it
and nothing else; the pass continues over the rest of the root, and the prune is
refused for that root because the listing no longer covers all of it.

**A directory the walk did not list is counted, not merely logged.** The pass
carries a `directories_missed` count, reports it in the closing summary, and
records it on the `ingest_runs` row. Absence of an `images` row is the
load-bearing claim of the whole design -- every consumer reads it as "this image
was never navigated" -- and under a directory nobody enumerated that reading is
simply false. A run that missed a directory still completes, because the rows it did
write are as good as any other run's, so the count is the only place the gap
shows; a consumer that means to read absence as an answer has it on the run row
rather than in a log file nobody kept. A pass whose count is zero listed the
whole root and absence means what it says everywhere under it.

**A directory already walked is not walked again.** The walk records each local
directory's device and inode as it enters it and skips one it has already
listed. A link from a subdirectory back to an ancestor otherwise writes the same
document under a new stub at every level, until the filesystem's own limit on
link traversal stops it -- forty-one rows for one document, forty of them
answering for images at paths no consumer will ask about, and the count of
navigated images wrong by all of them. A directory skipped this way is counted
as missed, since the walk did not enumerate it there, which is also what refuses
the prune for that pass. The identity is taken only for a local directory: a
cloud location has no links to go round in, and asking a bucket about a prefix
is a paid round trip per directory per run.

**Rows of documents that have left the tree are removed.** Presence has to mean
what absence means, so the stubs recorded for a root that this walk did not
find are deleted, and the count is reported and recorded in `ingest_runs` as
`files_removed`. The delete cascades to the child tables. This is sound only on
the evidence of a **complete listing of the root**: the prune reads the walk's
own listing and refuses one that does not cover the whole root, which is the
case whenever the root could not be listed or any directory under it could not.
Section 2.8 states the consequence for a worker that covers a share of a root.

**Per-root bookkeeping.** An `ingest_runs` table records, per root:
`root_url`, `started_utc`, `finished_utc` (NULL while running),
`files_seen` / `files_ingested` / `files_skipped` / `files_failed` /
`files_removed` / `directories_missed`, and `schema_version`, under a surrogate
`run_id` primary key
(a root legitimately has many runs, and a consumer reads the newest). The row
is written at
start and updated at completion, in both the interactive and cloud paths. A
consumer treats a root whose newest row has `finished_utc IS NULL` -- or no
row at all -- as not ingested, and fails with a message saying so.

**The normalized root is what is walked.** `normalize_root_url` renders the
root absolute once, and the walk and the retrievals are handed that rendering
rather than the string as typed. A relative root is a documented spelling of
the option, and the storage layer refuses a relative local URL outright; walking
the typed form would also record a relative `source_file` beside an absolute
`root_url` for the same file.

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

**A row the database refuses costs its own file.** A document can read cleanly
and still not go in -- an identifier too large for a `bigint`, a value a
backend's type will not hold -- and the failure arrives from the driver at the
insert rather than from any check the converter makes. Such a document would
otherwise take its whole chunk down and then the run, leaving the root's ingest
unfinished and every consumer refusing it. So a chunk whose write fails is
rolled back and written again one image at a time: every writable document goes
in, and the one that does not is counted as a failure for its own file. Nothing
is recorded in `failed_files` for it, exactly as nothing is for a retrieval that
failed -- the document read, so nothing about it says the next pass will not
store it, and a recorded refusal would be skipped for as long as the file did
not change. The one failure that is *not* isolated is a connection the driver
reports as invalidated: every remaining image would fail the same way, and a run
that "completed" without them leaves a consumer reading the absence of their
rows as "never navigated", so that one is allowed out.

**Names inside a document are checked with its shapes.** A `per_technique` entry
carries a `technique_name` and no two entries carry the same one, because that
name is half the primary key of `techniques`; an entry with none has no identity
and two entries with one name have the same one. A document that breaks either
is refused as a document of another shape. Standing a nameless entry in under a
placeholder name manufactured the collision out of the absence, and numbering
duplicates apart would put a technique nobody ran into the operator's report.

**The driver's exit status says whether the pass completed**, not what it found:
0 when every named root was walked, whatever mix of documents was read, skipped
and refused, and 1 when the run could not complete -- no index or no root
resolvable, the index unopenable, or a root that could not be listed. A status
read from a count of ingested documents flips between two passes over one
unchanged tree, since what one pass ingests or refuses the next one skips, and a
scheduled run would then see a failure once and never again. **The driver always
exits rather than raising.** The pass charges every failure it enumerates to one
file or one root; anything still escaping is a failure nobody enumerated, and a
console entry point owes its caller a message and a status for one rather than a
traceback. The roots such a run never reached keep their NULL finish times, so
no consumer reads absence under them as an answer.

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
enqueuer aggregates. The result carries the per-reason tally of section 2.7
beside the names, with one example file per reason, because the enqueuer's
closing summary is the only place a divided ingest can report why files were
refused and a bare count of refusals reads the same whether a tree holds many
documents that were never navigation results or the ingest went wrong.

A task carries the files of its share as the enqueuer's own walk reported them
-- each with its stub, `mtime_ns`, `size_bytes` and summary-PNG flag, plus the
run identifier, the root, `force`, and whether the listing reported metrics at
all -- and the worker ingests exactly those. It stats nothing and checks for
nothing: the one listing of the pass is the enqueuer's, and everything a worker
would otherwise ask the tree travels with the task. The worker applies the
incremental skip of section 2.7 to its own share, against what the index records
for its own stubs, so a retried task reads no document at all and costs a lookup
over its own stubs instead.

**A worker never prunes.** Removing the rows of documents that have left the
tree (section 2.7) is licensed by a complete listing of the root, and a worker
holding a share of one has no evidence about the stubs outside its share:
pruning on it would delete its peers' rows. The prune takes the walk's listing
and raises unless that listing covers the whole root, so the restriction is a
property of the seam rather than a rule a worker has to remember. Nothing hands
a worker a listing at all, so there is nothing for it to offer.

**The enqueuer prunes at fan-out, not at completion.** Fan-out is the one moment
of the pass that holds a complete listing, and it is the listing the shares were
cut from, so the prune is licensed exactly as the single-process one is. It
cannot race the workers either: every stub a worker writes is one the listing
held and the prune deletes only stubs it did not hold, so the two sets are
disjoint by construction however the workers are scheduled. Pruning at
completion instead would mean listing the whole root a second time -- the most
expensive thing an ingest does, and a paid round trip per directory on a cloud
root -- to act on evidence no share ever came from. The window between the two
is not a hazard: the run is unfinished throughout, so no consumer reads the root
either way.

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

**The enqueuer is `sd_stats_ingest` in two further modes**, since the fan-out
resolves the same roots and the same index URL as the pass it replaces:
`--output-cloud-tasks-file` lists, prunes, records what the walk found on each
run row, and writes the shares out; `--complete-cloud-tasks-file` reads the
`cloud_tasks` event log, adds the tallies up, and stamps the runs. The two are
mutually exclusive. Completion opens with `create=False`: the runs it means to
finish are in the index the fan-out wrote, and creating an empty one would report
every root as never fanned out.

**A run is stamped only when its shares account for the whole listing.** The
fan-out records `files_seen` on the run row, because no worker sees more than a
share; completion sums `files_ingested + files_skipped + files_failed` over that
run's own shares and refuses to stamp a run the sum does not match exactly. A
task that failed, timed out, or was never run read none of its documents, and a
run stamped without them tells every consumer that absence of their rows means
those images were never navigated -- the one claim the run bookkeeping exists to
license. A worker that reported an error rather than a share is likewise a
shortfall, so an unopenable index or a malformed task leaves the root unfinished
rather than silently shrinking it. An account that runs *past* the listing is
refused from the other side of the same rule: each task counts once, so the sum
can only exceed the listing on a report that is not this run's, and a run is not
stamped on an account that cannot be right.

Three rules are what make that sum mean what it says.

**Each task's report counts once**, taken under the `task_id` its event carries,
which the fan-out mints uniquely per share. A queue redelivers a task whenever
it could not see the delivery acknowledged, and an operator re-runs a task file
after a partial failure; a retried task reads nothing and reports its share a
second time, as skipped. Added twice, that report covers for a share that never
ran at all: over- and under-accounting cancel, the sum reaches the number the
walk found, and the run is stamped with its documents unread. The later report
of a task supersedes the earlier one, since a task that failed and was re-run
reports its failure first. A result carrying no task identity cannot be told
from a repeat of another and so counts toward no run.

**A share counts toward a run only when it names that run's root**, which is why
the worker returns the root beside the run identifier and completion compares
both. The identifier is a surrogate that starts again at 1 in a fresh index --
which is exactly what the remedy for a schema-version mismatch produces, and
what a mistyped `--results-db` names -- so a task file that outlived the index it
was cut from carries the run number of whatever was built next. Its shares then
add up to that run's listing while their rows sit under a different root, and a
run stamped on them is a root with nothing under it: every consumer reads absence
there as "this image was never navigated". A result naming a run being completed
but another root is counted and reported rather than credited, and is told apart
from one belonging to a fan-out nobody here is completing.

**A run whose listing was never recorded is never stamped**, whatever its shares
say. No files seen is not zero files seen: zero is what a root that was listed
and holds nothing records, and only zero can be accounted for by no shares at
all. A root the walk could not list keeps a run with no `files_seen`, exactly as
section 2.7 requires, and so does a pass that died between starting its run and
listing its root; reading either as zero completes a root nobody ever listed and
hands every consumer a tree of images to read as never navigated.

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

Details settled during execution, none of them a change of intent:

- **The root normalization lives in `spindoctor/results_index/roots.py`**,
  alongside the `ingest_runs` reads that decide whether a root has been
  ingested at all. Section 2.2 asks every consumer to spell a root the same
  way and to refuse a root with no completed run; both are one function
  there rather than one per consumer. `sd_stats_report` already uses the
  refusal for `--root`; the pipeline consumers of Phase 4 use the same one.
- **The old ingest's output is frozen rather than re-derived.** The
  byte-identical comparison needs the previous implementation, which this
  phase deletes, so the fixture tree and the `report.md` / `images.csv` the
  previous implementation produced from it are committed as test data in
  their own commit, ahead of the rewrite.
- **The CSV's two feature-count aggregates become integers.** `TOTAL(...)`
  always returned a float, including a `0.0` for an image with no features;
  `COALESCE(SUM(...), 0)` returns the count. The regression test asserts the
  numbers are equal rather than the text, and asserts it for those two
  columns alone.
- **The CSV's `status_reason` column no longer carries a fatal error.** That
  is the merge the split columns exist to undo, and `status_error` is beside
  it now. The regression test asserts that the pair reproduces what the one
  column held.
- **`sd_stats_ingest` and the statistics package leave the print-only list**
  in `tests/spindoctor/config/test_logging_static_invariants.py`, which
  section 2.10's collateral list does not name. The list is narrowed to the
  report modules, which keep `print()`.
- **The criterion-10 scan skips Markdown table rows.** The report writes its
  tables as literal rows, one of them headed `total (s)`, which the `TOTAL(`
  pattern reads as the SQLite aggregate. Nothing beginning with a table
  separator is SQL, and a test pins that the exclusion does not reach a
  statement -- against a widened exclusion as well as a blanked one, since a
  pattern that still excludes something is what would quietly empty the scan.
- **The column set changed, so the schema version is 4.** The JSON columns
  gained `none_as_null` (section 2.3), `ingest_runs` gained `files_removed` and
  then `directories_missed`, and `failed_files` was added (section 2.7). There
  are no migrations, so this is one version bump covering all four. The last of
  them arrived after the version had already been raised once in this phase, and
  it was raised again rather than reused: an index built from an earlier state
  of this phase would otherwise pass the version gate and then fail on a column
  that is not there, which is exactly what the gate exists to prevent.
- **The CSV export states its line terminator.** `csv.writer` defaults to CRLF;
  the export now names LF. The frozen `images.csv` blobs are LF, so what the
  export writes matches them byte for byte, which the previous implementation's
  output did not -- the blobs were normalized when they were frozen and the
  commit that froze them did not say so. The regression comparisons read fields
  rather than lines and would not notice either way, so the terminator has a
  test of its own reading the file as bytes.
- **Masking became one rule over a corpus, and stopped touching results
  roots.** Section 2.4 above is the rule as it now stands: one structural pass
  covering the query-parameter form of a password as well as the authority
  form, applied to every URL rather than only to the ones the parser rejects,
  and asserted over a generated corpus instead of a list of remembered shapes.
  `require_ingested_roots` masks the index URL itself, so a consumer cannot
  reintroduce the leak by forgetting; the roots in that message, and the results
  roots in the ingest driver's log, are printed exactly as they were given.
- **Section 2.4 rule 4 decides the ambiguous shape the other way round.** The
  base plan read `host:5432/path@name` by whether the text before the slash is
  digits -- "a port is digits and a password is not". It is now read as
  credentials outright. This is the one plan edit of this phase that reverses a
  decision rather than adding or tightening a requirement, so it is listed here
  on its own: digits-decide leaves a password that opens with digits,
  `123/secret`, visible in full, and the cost of the reading taken -- a mangled
  host and database name -- no longer reaches anything an operator needs, since
  a results root is never passed through masking at all. The URL parser reads
  that shape the same way, so for every spelling of it a parser accepts, this
  rule and `render_as_string()` hide the same characters.
- **The credentials end at the last `@` of the string** (section 2.4 rule 3),
  not at the last one before the first `/` and not at the last one before a
  `#`. Both narrower bounds stop inside a password that carries the character
  they stop at, and each was a leak. The accepted cost is over-masking a URL
  whose tail carries an at-sign after the real credentials, which is a fragment
  on a connection URL and therefore a URL no driver would have taken.
- **A failed write costs one file, not the run.** Section 2.7's isolation of a
  chunk whose write fails. SQLAlchemy savepoints were tried first and are not
  usable here: pysqlite's transaction handling commits a released savepoint
  independently of the enclosing transaction, which silently breaks the chunk
  boundary. Rolling the chunk back and rewriting it one image at a time costs
  nothing when nothing fails.
- **The ingest driver's exit status reports completion rather than counts**
  (section 2.7), because the count-based status flipped between two passes over
  one unchanged tree. It also always exits rather than raising, which is what
  its own documented contract said and what a traceback out of a console entry
  point breaks.
- **The walk is handed the normalized root** (section 2.7). Absolutizing the
  root for the key while walking the string as typed made a relative root -- a
  documented spelling -- a traceback out of the driver, with the run row already
  written and its finish time left NULL.
- **A directory the walk did not list is counted** (section 2.7), reported in
  the summary and recorded on the `ingest_runs` row, and the walk skips a
  directory it has already listed rather than descending into it again. The
  first is what keeps "absence means never navigated" honest for the consumers
  of Phase 4; the second stops a link back into a tree from writing one
  document as forty-one rows.
- **Ingest is a package, on the treatment section 3 names for the report.**
  Everything section 2.7 asks of one pass carries `spindoctor/cli/stats/ingest`
  past the 1000-line cap, so it is `ingest/` split along the stages a pass runs
  through: `counts` (the tally the summary is read from), `walk` (the single
  listing of a root), `store` (what the index already holds, and how rows go
  back in), `chunks` (batched retrieval, reading a document, the per-chunk
  write), `runs` (the record that makes absence of a row readable), and
  `driver` (the pass itself). The package re-exports the whole surface, so
  every consumer imports the names it always did from
  `spindoctor.cli.stats.ingest`. `report.py` stays inside the cap and stays one
  module. The source scan of criterion 10 finds the split modules for itself,
  and the floor it asserts its own reach against names them.

### Phase 3 — Cloud-task ingest

`sd_stats_ingest_cloud_tasks` per section 2.8, entry point in
`pyproject.toml`, with the enqueuer's two modes on `sd_stats_ingest`.

Tests: enqueuer creates the schema and workers refuse to; concurrent local
SQLite workers produce the same rows as a serial ingest; per-task counts
aggregate into `ingest_runs`; the worker writes zero bytes to stdout and
stderr from SpinDoctor code, asserted at file-descriptor level as the
existing cloud-task silence tests do; the driver appears in
`_CLOUD_TASK_DRIVERS` and passes the no-logging-flags assertion.

Details settled during execution, none of them a change of intent:

- **The enqueuer prunes at fan-out**, which is section 2.8's open question
  decided. The reasoning is in section 2.8 above; the short form is that fan-out
  is the only moment of the pass holding the complete listing the prune is
  licensed by, and it is the listing the shares were cut from, so the prune and
  the workers' writes cannot touch the same stub. Within one fan-out this is an
  ordering guarantee rather than only a set argument: the prune runs before the
  task descriptions are built, so no worker of that pass exists while it runs.
  Two limits are worth recording. The disjointness is a claim about **one**
  fan-out: two overlapping fan-outs against one root can leave a stale row, when
  a worker of the first writes a stub after the second's snapshot of what is
  recorded and before its delete, for a document that left the tree between the
  two listings. It is narrow, and the next pass removes the row. And the prune
  is destructive before any document has been read, so a fan-out that is
  abandoned shrinks the index -- but only by rows whose documents have genuinely
  left the tree, and the run is unfinished throughout, so no consumer reads the
  root either way.
- **Each task's report is counted once, under its `task_id`**; a share counts
  toward a run only when it names that run's root; the account must match the
  listing exactly rather than merely reach it; and a run whose listing was never
  recorded is never stamped. All four are in section 2.8 above. Summing files
  alone lets a share reported twice cover for a share that never ran; crediting
  by run identifier alone lets a task file that outlived its index stamp a root
  with nothing under it, since the identifier restarts at 1 in a fresh one;
  accepting an account that runs past the listing stamps a run on evidence that
  cannot be right; and reading an unrecorded listing as zero files stamps a root
  nobody listed. Each one hands consumers a tree of images to read as never
  navigated, which is the claim the run bookkeeping exists to license.
- **Two spellings of one root are one root**, at the fan-out and at the
  completion. Listed twice, every document is handed out in two shares and read
  twice, the first of the two runs is left unfinished for good, and the
  completion stamps the newer run and then reports the root it has just finished
  as one nobody divided up.
- **A count no share could report is not a share's tally.** `_share_tally`
  bounds the magnitude of each count as well as its type and its sign. The
  counts are written to the run row on a shortfall, and one larger than that
  column holds ends the whole completion in the database driver's own error --
  for one corrupt or foreign line of a concatenated event log, which is exactly
  the input class the guards either side of it exist for.
- **The seam lives in `spindoctor/cli/stats/ingest/tasks.py`**, beside the pass
  it divides: fan-out, one share, and the completion that adds them up are the
  same three stages `driver.py` runs in one process, and both read the same
  walk, store and chunk modules. The package re-exports them, so the drivers
  import from `spindoctor.cli.stats.ingest` as they do everything else.
- **Two Phase 2 helpers were widened rather than copied.** `_files_to_read`
  takes the files, their summary stubs and the metrics flag instead of a whole
  `_RootListing`, so a share selects by exactly the rule a root does and there
  is nothing listing-shaped for a worker to reach for; `_recorded_files` takes
  an optional set of stubs, so a share reads what the index holds about its own
  files rather than about every row of an archive-scale root.
- **A share names every file it could not read**, which a pass over a whole
  root does not: the fan-out bounds a share, and a worker has no run log to name
  them in instead. The whole-root pass keeps one example per reason, as before.
  The share's result carries that per-reason tally as well as the names, and the
  completion folds it into the summary it writes, so a divided ingest reports
  why files were refused exactly as a single-process pass does. The names stay
  in the event log: a summary that listed several hundred thousand of them would
  read as a broken ingest rather than as the ordinary thing it is, which is why
  the whole-root pass keeps one example per reason in the first place.
- **A run row carries what the fan-out found before it is finished.**
  `files_seen`, `files_removed` and `directories_missed` are written at fan-out
  with the finish time left NULL, because nothing later in the pass can find
  them out again and the completion step must not have to list the root to learn
  them.
- **`sd_stats_ingest` and `sd_stats_ingest_cloud_tasks` joined the program
  identity tests** in `tests/spindoctor/config/test_logging_keys.py`, which named
  neither. The interactive driver has declared `PROGRAM_NAME` since Phase 2 and
  section 2.10's collateral list did not reach that file.
- **The developer guide's script table gained the statistics family.** It is
  headed as the full set of `[project.scripts]` and named none of them, so
  adding one program to it meant naming its siblings too.
- **Every root-keyed delete is exercised with a second root present.** A
  fixture holding one root cannot tell a query keyed by the pair from one keyed
  by the stub alone, which is how a root-blind query ships. Both arms of the
  share's own lookup, both deletes of an image write, both of a refusal, and
  both of the prune are each pinned by a test that fails when its root half is
  dropped (`tests/spindoctor/cli/stats/test_ingest_two_roots.py`, and the two
  share-lookup tests beside it). What each break costs differs -- a navigated
  image made invisible under another root, a refusal cleared so its file is
  downloaded again on every pass -- and the tests are named for it.
- **The suite resolves no results index it did not name.** A URL comes from an
  argument, the `environment.results_db` configuration variable, or
  `NAV_RESULTS_DB`, and a test of the no-index path names none of them. Both
  ambient levels are closed for every test by one fixture in `tests/conftest.py`
  rather than by a line each test author has to remember: it unsets the variable
  and runs from a directory holding no `nav_default_config.yaml`. Run from a
  directory that names a live index, the suite had opened it -- for SQLite, a
  write-lock probe against a file an ingest may be holding.

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
`import spindoctor.dataset` does not import `sqlalchemy`. **That assertion is
criterion 2's only test and this phase owns it**: no earlier phase writes it,
because the branch-local import it protects is added here, so it must not be
assumed to exist already.

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

- **A document column for bundle generation** (#464). Storing the raw metadata JSON
  per image (SQLite TEXT, PostgreSQL `jsonb`) is the only way bundle
  generation stops reading one document per image, since it needs the whole
  document. It roughly doubles the index size, which weakens shipping the
  SQLite file to workers, so it likely belongs in a separate optional
  database rather than the main index.
- **Serving the curation triage tool** (#465). Needs `missing_frac`,
  `saturation_frac`, classifier flags, the per-technique list, result file
  paths, a name-to-stub lookup, and a second (rescue) root -- a schema
  widening for one out-of-package tool, wanted only if triage's ten
  rglobs-per-frame remain a practical pain after the pipeline consumers are
  converted.
- **Shipping the index to cloud workers** (#466). Publishing the SQLite file to the
  results bucket and having each worker download it once is the alternative
  to a PostgreSQL instance, and needs a documented workflow either way.
- **A `--since` selector for ingest** (#467). The stat-pair skip makes a re-scan
  cheap in reads but not in listings; a time-bounded scan would cut the
  listing too.
- **Two overlapping ingest passes over one root can leave a stale row** (#479).
  A worker of the first writes a stub after the second has read what is
  recorded and before its delete, for a document that left the tree between the
  two listings. Narrow, self-healing on the next pass, and invisible to
  consumers while it is open, since both runs are unfinished; what is undecided
  is whether a fan-out over a root whose newest run is unfinished should be
  refused, warned about, or left as it is.
- **An abandoned fan-out has already removed rows** (#480). The prune runs
  before any document is read, so a pass that is given up on after step 1 has
  shrunk the index. Only rows whose documents have genuinely left the tree go,
  and the run is unfinished throughout, so nothing valid is lost and no consumer
  reads the root; the way back is a full ingest.
- **The lockability probe takes a write lock on a consumer's open** (#462).
  Section 2.5 has it refuse in both modes, so a consumer opening a SQLite index
  while an ingest holds a write transaction waits out the busy timeout and can
  then fail with the filesystem-and-PostgreSQL message though nothing is wrong. It
  is the plan's own rule and is left as written here. Cloud-task ingest makes
  the collision routine rather than occasional: a worker opens the index once
  per task, so a local SQLite run of a thousand tasks takes a thousand write-lock
  probes against peers holding chunk transactions. Nothing here measures what
  those probes wait, and no test does; the question of whether a `create=False`
  open should probe with a read instead is tracked on its issue, with what this
  work changed about it recorded there.

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
