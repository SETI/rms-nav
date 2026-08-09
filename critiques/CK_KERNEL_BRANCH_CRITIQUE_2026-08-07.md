# CK Kernel Branch Critique

**Date:** 2026-08-07
**Baseline:** `rf_ck_kernels` @ 4b26ae9, the eight phases of the
corrected-kernel program as bundled in PR #457, code and documentation
both (~21k added lines across 87 files).
**Question:** is the branch correct, complete, and honest as it stands --
and does every open review thread (including the 37 CodeRabbit comments
on the PR) resolve to a fix, a reasoned decline, or a refutation?
**Method:** six parallel adversarial reviews (SPICE/C-matrix core, the
writer package in two halves, the navigation integration, the
documentation, the plan/issue reconciliation), each verifying its claims
against the source and, for every SPICE behavioral claim, against the
installed toolkit by execution. Every finding below was then re-verified
independently before being acted on; two review claims died in that step.
The full gate suite was run before and after. This is a frozen snapshot;
the fixes it describes were applied on the branch after 4b26ae9, and
nothing in this file is maintained.

---

## Verdict

The branch is in substantially better shape than its size suggests: the
C-matrix math is right (sign convention, conjugation order, quaternion
convention, and the OffsetFOV replication were each re-derived
independently and pinned by discriminating tests), the writer's
malformed-input posture is exemplary, the round-trip evidence is real,
and the documentation is unusually accurate against the code. But the
review surfaced five genuine majors -- four in code, one a documentation
contract -- every one of them of a shape the phases' own testing could
not see: a correct rule keyed on the wrong invariant, a guard testing a
property SPICE does not share, and a contract honored on the main path
and dropped on an early exit.

## Majors (all fixed on the branch)

1. **BOTSIM pairing keyed on exposure start; the invariant is shutter
   close.** Cassini's `IMAGE_TIME` is the shutter close; oops derives
   `tstart = tstop - texp`; the dataset layer pairs on close time within
   2.0 s. `botsim_losers` compared `start_et` within 1.0 s, so a genuine
   pair with unequal exposure durations (NAC 0.18 s / WAC 2.6 s -> starts
   2.42 s apart) escaped detection and both corrections were written into
   one file for one bus, where segment priority silently shadows one.
   Every BOTSIM test used equal exposures, which is why it survived.
   Fixed: keyed on `stop_et`, window 2.0 s matching the dataset layer,
   unequal-duration case pinned.

2. **The comment-area guard tested `isprintable`; `dafac` requires
   printable ASCII.** An accented letter in a `status_reason` passes the
   up-front check, every segment writes, `ckcls` succeeds, and then the
   comment write raises -- leaving the complete, furnishable,
   provenance-less kernel the check exists to prevent, with the overwrite
   guard then blocking the rerun. Measured directly: `dafac` raises
   `SPICE(ILLEGALCHARACTER)` on byte 195. The same gap existed for output
   paths and meta-kernel paths. Fixed at all three sites
   (`isascii() and isprintable()`), with the free-text reason now elided
   to fit its line rather than allowed to refuse the whole file.

3. **No bound on the joined meta-kernel path; SPICE's is 255.** The
   continuation logic bounds each 80-character piece but not the total,
   so a deep output tree produces a meta-kernel every consumer's `furnsh`
   refuses -- after the kernels listed before it have loaded, which is
   precisely the partially-loaded-pool state the module's own docstring
   warns about. Measured directly. Fixed: paths over 255 characters are
   refused at writing time.

4. **A non-finite navigated offset was absorbed as an environment
   failure.** `NaN` defeats the degenerate-axis guard (`NaN < 1e-12` is
   False), flows to `axisar`, and surfaces as the one exception class
   (`NavPointingError`) the orchestrator deliberately absorbs per image
   -- so a regressed technique emitting NaN offsets would silently drop
   pointing from a whole batch while every image reported success, the
   exact failure mode the propagate-by-design contract exists to prevent
   (and the twelfth instance of the branch's own eleven-defect
   malformed-input catalog). Fixed: offsets are validated at the entry
   point (arity and finiteness) and refused as `ValueError`, which
   propagates.

5. **The empty-selection early exit dropped the exit-status contract.**
   A run whose metadata was entirely unreadable selected nothing and
   exited 0 -- the worst instance of exactly the condition the non-zero
   exit exists to signal. Fixed; the no-images path now carries the
   unreadable count, and the user guide says so.

## Notable minors (fixed)

- A same-mission metadata document with no readable
  `observation.instrument` vanished silently -- in neither the report nor
  the unreadable count, defeating the every-image-appears-once guarantee
  for the malformed-input class this project has been bitten by.
- A WAC frame yielded to its NAC partner even when the partner's baseline
  failed reproduction and wrote nothing; pairing now runs after the
  reproduction test, so a winner that writes nothing suppresses nothing.
- The corrected-kernel `_nav` marker was tested case-sensitively while
  class patterns match case-blind, so an upper-cased copy of a corrected
  kernel could re-enter the index as a predicted baseline candidate.
- An inverted `--start-time`/`--stop-time` pair silently selected nothing
  and exited clean; now refused by name. A malformed time string now
  names its flag instead of surfacing as a raw SPICE traceback.
- `ckcls` ran outside `write_ck_file`'s guarded region, so a close-time
  failure leaked the handle and left the partial file the docstring
  promises never survives; cleanup now covers it, and cleanup's own
  failure cannot replace the error worth reading.
- A non-finite `exposure_s` (the one `times` field that passes through no
  SPICE conversion) serialized as a bare NaN token that strict JSON
  readers reject far from the cause; all four time fields are now refused
  at the source. The public `AttitudeBaseline`/`PointingSolution`
  dataclasses now validate their matrices at construction.
- The stale `navigate_image_files` module docstring still promised that
  no exception escapes navigation, contradicting the (correct,
  deliberate, tested) propagate-on-defect design added by this branch.
- Four byte-identical copies of the furnish/unload context manager
  collapsed into one leaf module (`spindoctor.cli.ck.pool`); duplicate
  frame-label spelling collapsed into one helper; the duplicate-image
  refusal now names the images.
- Documentation: the dev guide still described the pre-#453
  directory-based classification in its tie-break passage; the "size
  proportional to the original" claim (guide and plan) described a copy
  mechanism the writer does not have; the "report is always written"
  claim contradicted both the code and the guide's own refusal section;
  a `:func:` reference into the un-autodoc'd `sd_create_ck` never
  resolved; the no-oops claim is now scoped to the writer package, since
  the driver imports oops through the shared logging surface (and its
  `MISSIONS` docstring claimed otherwise -- both ends fixed).

## Review claims refuted by measurement (no change)

- **"`cspyce.dafec` reads one chunk; long comment areas truncate"**
  (CodeRabbit, Major): cspyce wraps the CSPICE chunk loop internally; a
  2000-line / ~140 kB comment area reads back complete in one call.
- **"`obs.texp` can disagree with `obs.time` and skew Voyager snapped
  tolerances"** (CodeRabbit, Major): the writer refuses any document
  whose `exposure_s` differs from `stop - start` by more than 1 ms
  (`ImagePointing`), and oops defines `texp` as exactly that span.

## Declined with reason

- **Pinning the Voyager snapped-lookup constants (800, 48) to oops**:
  oops exposes them only as inline literals in a function body, and the
  writer package is barred from importing oops by acceptance criterion 7,
  test-enforced. The real-frame round trip pins the behavior.
- **Removing the `spindoctor.cli.ck` nitpick-ignore**: it follows the
  identical convention of every sibling CLI subpackage, and whether any
  of them belongs in the API reference is the recorded, still-open
  decision of #443.
- **`--dist loadgroup` for the round-trip module**: the project mandates
  `--dist=loadfile` everywhere for PyQt6 worker safety; the per-worker
  cache rebuild cost is noted in the module instead.

## Not fixed here, tracked

- Everything already filed on the PR (#433-#437, #440, #443, #444, and
  the #446-#448, #452, #455 set) stands as filed; nothing new needed
  filing -- every remaining review thread resolved to a fix on the
  branch or a decline above.

## Gates at this snapshot

`ruff check`, `ruff format --check`, `mypy` strict (src and tests),
`sphinx-build -W`, `pymarkdown`: all clean. Unit suite at the CI
invocation (`-n 4 --dist=loadfile`): green before the fixes (5714) and
green after with the review's added tests. Plans reconciled as if PR #457
merged: #449 removed from the four lists still carrying it (it is fixed
by this branch; `Closes #449` added to the PR), #455 and #459 indexed,
the Voyager zero-AV acceptance wording corrected to what Phase B
actually pins.
