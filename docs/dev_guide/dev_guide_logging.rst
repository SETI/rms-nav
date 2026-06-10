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
pixels at full-well DN") replaces percent-encoded numbers like ">80%".

Conventions
===========

* Never ``import logging`` in ``nav.*`` core code.
* Never ``print(...)`` in library code; route through ``self.logger``.
* Every ``NavTechnique.navigate`` body wraps its work in
  ``with self.logger.open(f'TECHNIQUE: {self.name}'):`` for log
  scoping.
* The orchestrator captures every per-technique exception and emits an
  ``EXCEPTION``-level pdslogger line via ``self._logger.exception(...)``;
  the technique's failure surfaces on the returned ``NavResult``,
  never as a propagating Python exception.
