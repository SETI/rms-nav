========================
Developer Guide: Logging
========================

The autonomous-navigation pipeline routes every per-image log line
through ``pdslogger`` (``nav.config.logger.IMAGE_LOGGER``). The standard
library ``logging`` module is intentionally **not used** anywhere in the
``nav.feature``, ``nav.nav_model``, ``nav.nav_orchestrator``,
``nav.nav_technique``, or ``nav.support`` packages.

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
  in :data:`nav.nav_orchestrator.status_reason_info.STATUS_REASON_INFO_TEMPLATE`
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
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` or
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` flag firing,
  a fall-back path triggering, a per-instrument override missing where the default
  applies. WARNING lines surface in the curator's per-image JSON sidecar so the
  operator-curated regression library can flag them.
* **ERROR** — a per-image failure that the orchestrator could downgrade to a failed
  :class:`~nav.nav_orchestrator.nav_result.NavResult` rather than propagate as a Python
  exception: a model whose ``create_model`` returned no usable state, an ensemble that
  cannot reconcile any technique result. ERROR is reserved for failures whose remediation
  is operator-side (re-run with different inputs, file a bug); the run continues to
  emit a JSON sidecar.
* **EXCEPTION** — emitted by ``self.logger.exception(...)`` from inside the
  orchestrator's broad ``except Exception`` blocks around every model and technique
  callback. Carries a full Python traceback; the offending model or technique is
  treated as if it produced no output, the rest of the pipeline continues, and the
  surfaced :attr:`~nav.nav_orchestrator.nav_result.NavResult.status_reason` records
  what fell over. Never raise EXCEPTION from non-orchestrator code; let the
  orchestrator's sandbox catch it.
* **FATAL** — process-level failures (a corrupt config file, a missing kernel, an
  un-importable extension) that abort the whole run before any image is processed.
  Reserve for setup errors that no per-image fallback can recover from.

Conventions
===========

* Never ``import logging`` in ``nav.*`` core code.
* Never ``print(...)`` in library code; route through ``self.logger``.
* Every :meth:`~nav.nav_technique.nav_technique.NavTechnique.navigate` body
  wraps its work in
  ``with self.logger.open(f'TECHNIQUE: {self.name}'):`` for log
  scoping.
* The orchestrator captures every per-technique exception and emits an
  ``EXCEPTION``-level pdslogger line via ``self._logger.exception(...)``;
  the technique's failure surfaces on the returned
  :class:`~nav.nav_orchestrator.nav_result.NavResult`,
  never as a propagating Python exception.
