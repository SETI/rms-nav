========================
Developer Guide: Logging
========================

The autonomous-navigation pipeline routes every per-image log line
through ``pdslogger`` (:data:`~spindoctor.config.log_scope.IMAGE_LOGGER`). The standard
library ``logging`` module is intentionally **not used** anywhere in the
``spindoctor.feature``, ``spindoctor.nav_model``, ``spindoctor.nav_orchestrator``,
``spindoctor.nav_technique``, or ``spindoctor.support`` packages.

Why pdslogger
=============

* It produces a single per-image log file alongside the JSON metadata
  the curator writes; operators can read the narrative of one image
  without correlating multiple files.
* It supports nested sections via ``logger.open(...)``, which the
  ``NavTechnique`` base class uses to delimit each technique's
  contribution.
* Tests capture pdslogger output via ``capsys`` (it writes through its
  own stream handler that does not feed the standard ``logging``
  propagation, so ``caplog`` sees nothing).

Log structure
=============

Every navigation produces a top-level INFO line per per-image phase
plus a final ``status_reason``-keyed verdict:

* **Per technique.** Each technique opens a section with
  ``with self.logger.open(f'TECHNIQUE: {self.name}'):`` so per-image
  logs delimit each technique's contribution unambiguously.
* **Per status reason.** The orchestrator emits one INFO line per item
  in :data:`spindoctor.nav_orchestrator.status_reason_info.STATUS_REASON_INFO_TEMPLATE`
  for the final verdict's ``status_reason``. Operator-readable text
  describes both the outcome (``Final: status=ok``,
  ``Final: status=failed``) and the proximate cause
  (``Image classifier: blank / dark frame``,
  ``No technique's is_feasible returned True``, etc.).
* **Hard failures.** Image-classifier hard-failures
  (``blank``, ``fully_overexposed``, ``mostly_missing_data``,
  ``corrupt``) short-circuit before any extractor runs; the per-image
  log shows the section header followed by the matching INFO lines so
  an operator can tell at a glance why the run failed.

Tests
=====

Tests assert log content via ``capsys`` not ``caplog``:

.. code-block:: python

    def test_orchestrator_emits_blank_status_info(
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # ... drive the orchestrator with a blank image ...
        out = capsys.readouterr().out
        assert 'Image classifier: blank / dark frame' in out

The pdslogger templates avoid the ``%`` character because pdslogger
interprets ``%`` as a positional-format placeholder. Plain prose ("most
pixels at full-well DN") stands in for percent-encoded numbers like ">80%".

Log levels
==========

pdslogger exposes the standard six-level ladder. Pick the level that matches the
audience and the consequence of the line, not the call site's depth:

* **DEBUG** — pixel-level intermediate values, per-iteration LM diagnostics, per-vertex
  Tukey weights, and any other quantity an operator only consults while reproducing a
  single image's behaviour. DEBUG output is not routed to the per-image log file by
  default; enable it through the per-driver ``--log-level`` flag when needed.
* **INFO** — the per-image narrative every operator should see by default: phase
  headers (extraction begin / end, pass-1 ensemble verdict, pass-2 ensemble verdict),
  the final ``status_reason`` line, and one summary line per technique with its
  consumed feature count and reported confidence. INFO is the default verbosity for the
  per-image log file.
* **WARNING** — recoverable anomalies that do not fail the image but should bias an
  operator's review: a feature dropped by the reliability gate, a technique's
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` or
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` flag firing,
  a fall-back path triggering, a per-instrument override missing where the default
  applies. WARNING lines surface in the curator's per-image JSON sidecar so the
  operator-curated regression library can flag them.
* **ERROR** — a per-image failure that the orchestrator could downgrade to a failed
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` rather than propagate as a Python
  exception: a model whose ``create_model`` returned no usable state, an ensemble that
  cannot reconcile any technique result. ERROR is reserved for failures whose remediation
  is operator-side (re-run with different inputs, file a bug); the run continues to
  emit a JSON sidecar.
* **EXCEPTION** — emitted by ``self.logger.exception(...)`` from inside the
  orchestrator's broad ``except Exception`` blocks around every model and technique
  callback. Carries a full Python traceback; the offending model or technique is
  treated as if it produced no output, the rest of the pipeline continues, and the
  surfaced :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status_reason` records
  what fell over. Never raise EXCEPTION from non-orchestrator code; let the
  orchestrator's sandbox catch it.
* **FATAL** — process-level failures (a corrupt config file, a missing kernel, an
  un-importable extension) that abort the whole run before any image is processed.
  Reserve for setup errors that no per-image fallback can recover from.

Writing a component that logs
=============================

Which logger a class writes to is declared, not inferred. Most components work
on one image and keep the default :attr:`LogRole.IMAGE
<spindoctor.config.log_scope.LogRole>`; one whose work spans a run -- enumerating a
dataset, tallying totals -- sets ``log_role = LogRole.MAIN`` on the class, and
:class:`~spindoctor.support.nav_base.NavBase` binds ``self.logger`` accordingly.

Open sections with :meth:`~spindoctor.support.nav_base.NavBase.log_section`
rather than ``self.logger.open``. A level is applied when a section is opened,
so the section is where a component's configured level takes effect; calling
``open`` directly silently ignores it::

    with self.log_section(f'TECHNIQUE: {self.name}'):
        ...

A component that is a module-level function rather than a class gets the same
treatment from the :func:`~spindoctor.config.log_scope.logged_section`
decorator, which is what makes it independently configurable at all.

The name a component is configured under is its ``log_key``, defaulting to the
snake_case form of the class name with the ``Nav`` prefix and suffix stripped
(``TitanHazeNav`` becomes ``titan_haze``). A family that should share one key
-- every body model, say -- declares it once on their base, since ``log_key``
is inherited normally. Adding a technique or model therefore adds a
configuration key automatically; adding a function-shaped component means
adding its key to ``OTHER_LOG_KEYS`` in
:mod:`spindoctor.config.logging_keys`, or the configuration will reject it.

Every dispatch module that has a logger declares ``PROGRAM_NAME`` from
:mod:`spindoctor.config.program_names`. It names the program's main log
directory and selects its block under ``logging.programs``, so a program
without one has no way to be configured separately and no place to put its
main log.

The scope rule
==============

An image-role component that logs when no image scope is open is a bug. The
record is routed to the main logger so it is never lost, and a warning names
the call site, deduplicated so a loop cannot flood the log. Under
``logging.strict_scope`` it raises instead.

There is no legitimate case for it in production code: a component logging
about one image should be running inside that image's section, and one whose
work spans the run belongs on the main logger. Strict scope is opt-in per test
rather than on suite-wide, because a unit test that drives a model or technique
directly is correct isolation testing, not a mis-binding -- request the
``strict_log_scope`` fixture from a test that drives a real pipeline.

Cloud tasks
===========

``sd_offset_cloud_tasks``, ``sd_backplanes_cloud_tasks`` and
``sd_mosaic_cloud_tasks`` write nothing to the terminal. A worker's console
belongs to ``cloud_tasks``, which reports task progress there under its own
configuration; per-image processing detail goes to the per-image log file,
under the same ``{log_root}/{backend}/`` tree the interactive driver writes
to. Levels resolve identically, so an image's log reads the same whichever
driver produced it.

:func:`~spindoctor.config.logging_config.build_cloud_task_logging` is what a
task calls in place of
:func:`~spindoctor.config.logging_config.build_run_logging`. It builds no main
logger, and it refuses a console for either logger however the configuration
or the command line asked for one. Two details are worth knowing before
changing anything here:

* It is called **inside** each worker's task handler, not once at startup.
  Workers are spawned rather than forked, so a worker process does not inherit
  what the parent configured.
* Both loggers are bound to ``pdslogger.NULL_HANDLER`` and have
  ``propagate`` turned off. Neither is redundant: a ``PdsLogger`` with no
  handlers at all prints every record to stdout regardless of level, and one
  that propagates reaches the root handler ``logging.basicConfig`` installs in
  each worker, which re-emits every line a second time on stderr.

Because a cloud task has no main log, a record about one image must be logged
to :data:`~spindoctor.config.log_scope.IMAGE_LOGGER` rather than to
:data:`~spindoctor.config.logger.MAIN_LOGGER`. A per-image failure reported to
the main logger is reported nowhere at all: the task carries on to the next
image, and nothing is left to say what happened to that one.

Conventions
===========

* Never ``import logging`` in ``nav.*`` core code.
* Never ``print(...)`` in library code; route through ``self.logger``.
* Every :meth:`~spindoctor.nav_technique.nav_technique.NavTechnique.navigate` body
  wraps its work in
  ``with self.logger.open(f'TECHNIQUE: {self.name}'):`` for log
  scoping.
* The orchestrator captures every per-technique exception and emits an
  ``EXCEPTION``-level pdslogger line via ``self._logger.exception(...)``;
  the technique's failure surfaces on the returned
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`,
  never as a propagating Python exception.
