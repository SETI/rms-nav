=======
Testing
=======

Overview
========

The test suite has three tiers, separated by the ``integration`` and
``postgres`` markers. ``addopts`` in ``pyproject.toml`` carries ``-m "not
integration and not postgres"``, so a plain ``pytest`` runs the default tier
alone:

- The **default tier** runs on a plain ``pytest``. Everything in it is fast and
  self-contained: it needs no spacecraft holdings, no SPICE kernels, and no
  network. Unit tests live here, and so do the in-process simulator tests --
  every simulated frame is rendered and navigated in memory, so the whole
  simulator-driven invariant and structural coverage runs without external data.
- The **integration tier** is opted into with ``-m ""`` or ``-m integration``.
  It holds the slow and the archive-backed tests: the real-image
  regression cohort (which fetches PDS holdings and resolves SPICE geometry) and
  the heavier or jitter-prone in-process simulator tests.
- The **postgres tier** is opted into with ``-m postgres`` (or ``-m ""``). It
  re-asks the results-index questions against a real PostgreSQL server, because
  SQLite accepts spellings a server rejects: an integer compared against a
  boolean, a single-precision float where a double was meant, a foreign key that
  is only enforced when asked. The tests read their connection URL from
  ``SPINDOCTOR_TEST_POSTGRES_URL`` and skip themselves when it is unset, so the
  tier is runnable on any machine with a server and harmless on any machine
  without one. Each test creates and drops a PostgreSQL schema of its own, so
  repeat runs and parallel workers do not collide. CI has no service container
  for it; run it locally before changing anything in
  :mod:`spindoctor.results_index`
  or the statistics programs.

The simulator (:doc:`dev_guide_simulator`) is the engine behind several tiers: it
lets the suite grow algorithmic-invariant and sensitivity coverage on frames
whose true offset is known by construction, without operator labour and without
real data.

Running the suite
=================

.. code-block:: bash

   pytest                              # default tier (fast, no holdings)
   pytest -m ""                        # full suite, every tier
   pytest -m integration               # only the integration tier
   pytest -m postgres                  # only the postgres tier
   pytest -n auto --dist=loadfile      # parallel, matching CI (loadfile avoids
                                       #   PyQt6 worker crashes)
   pytest tests/spindoctor/sim/test_sim_noise.py            # one file
   pytest tests/spindoctor/sim/test_sim_noise.py::test_foo  # one test
   pytest --cov                        # with coverage

   ./scripts/run-all-checks.sh         # ruff + mypy + pytest + docs + markdown
   ./scripts/run-all-checks.sh -i      # the same, including integration tests
   ./scripts/run-all-checks.sh -P      # the same, including the postgres tier

The postgres tier additionally needs a server to point at:

.. code-block:: bash

   export SPINDOCTOR_TEST_POSTGRES_URL=postgresql+psycopg://USER@HOST:5432/spindoctor

Supply the credentials the way the server expects them -- a ``~/.pgpass`` entry,
``PGPASSWORD``, or a peer-authenticated local socket -- rather than writing a
password into a shell profile or a checked-in file. A password inside the URL
works, and the tier masks it out of every message, but the URL itself is then a
secret in the environment of every process the shell starts.

``pytest-xdist`` must run with ``--dist=loadfile``; the default scheduling
crashes PyQt6 workers when tests from one file split across processes. Multi-test
integration runs should always use ``-n auto --dist=loadfile``.

Archive-backed tests additionally require the holdings and catalog environment
(set by CI; see :doc:`dev_guide_introduction`):

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=https://pds-rings.seti.org/holdings
   export PDS4_HOLDINGS_DIR=https://pds-rings.seti.org/pds4
   export OOPS_RESOURCES=https://storage.googleapis.com/rms-node-oops-resources
   export UCAC4_PATH=https://storage.googleapis.com/rms-node-star-catalogs/UCAC4
   export YBSC_PATH=https://storage.googleapis.com/rms-node-star-catalogs/YBSC
   # plus SPICE kernels at $SPICE_PATH for any real navigation run

Test kinds
==========

The suite is layered by what a test proves and what it needs. The simulator-only
tests need none of the archive environment above.

.. list-table::
   :widths: 28 14 16 42
   :header-rows: 1

   * - Kind (path)
     - Tier
     - Requires
     - What it proves
   * - Unit tests (``tests/spindoctor/**``)
     - default
     - nothing
     - One component in isolation (config, feature, dataset, obs, model,
       technique, orchestrator, reproj, support).
   * - Results index on a server
       (``tests/spindoctor/results_index/test_postgres.py``)
     - postgres
     - PostgreSQL
     - The schema, the version gate, and the type discipline against a backend
       that enforces them, rather than against SQLite's permissive typing.
   * - Simulator unit tests (``tests/spindoctor/sim/**``)
     - default
     - nothing
     - The renderer's contracts: determinism, noise, saturation, PSF, stray
       light, instrument coupling, camera roll, irregular-body rendering and the
       ``nav_override`` channel, scene-schema validation, and the information
       boundary (``test_information_boundary.py``: no truth key is reachable
       through ``obs.nav_params``).
   * - GUI smoke (``tests/main/test_create_simulated_image.py``,
       ``tests/main/test_sim_editor_round_trip.py``)
     - default
     - PyQt6
     - Each GUI control wires to the right ``sim_params`` field and a YAML
       scene round-trips through the editor without loss.
   * - Scene structural (``test_sim_scenes.py``)
     - default
     - nothing
     - Every catalog scene validates, sits in a declared class, has a unique
       name, and renders.
   * - Algorithmic invariants (``test_sim_algorithmic_invariants.py``)
     - default
     - nothing
     - Each technique recovers its planted offset / roll on a clean scene --
       correct by construction, so no baseline.
   * - End-to-end sim nav (``test_sim_navigation.py``)
     - default
     - nothing
     - A simulated frame navigates through the full orchestrator and recovers a
       planted offset.
   * - Sim bug regression (``test_sim_regression.py``)
     - default
     - nothing
     - Fast bug-specific scenes guarding defects the sweeps surfaced.
   * - Sim regression baselines (``test_sim_baselines.py``)
     - integration
     - nothing
     - Every catalog scene re-navigates to its recorded rounded outcome (a
       tripwire). Integration-marked because the solvers carry sub-millipixel
       cross-process jitter.
   * - Sensitivity sweeps (``test_sim_sweeps.py``)
     - integration
     - nothing
     - Each single-variable sweep responds as expected (a technique transition, a
       degradation to failure, recovery within tolerance). Heavier -- each sweep
       navigates several frames.
   * - Pose behavioral (``test_sim_irregular_pose.py``)
     - integration
     - nothing
     - On a wrong-pose irregular body the limb degrades far off while the
       pose-free blob stays accurate (asserted per technique).
   * - Real-image structural (``test_image_library.py``)
     - default
     - nothing
     - The operator-curated sidecar catalog validates structurally (schema,
       classes, uniqueness) without fetching images.
   * - Real-image regression (``test_autonomous_nav.py``,
       ``test_baselines.py``)
     - integration
     - holdings + SPICE
     - Each curated real image navigates to its expected status / tier / offset
       and matches its recorded baseline -- the calibration tripwire.

The simulator's role across these tiers is summarized in
:doc:`dev_guide_simulator`. The operator-curated real-image cohort is documented
in :doc:`dev_guide_image_library`.

Characterization runners and updaters
=====================================

These are not pytest tests; they are ``python -m`` scripts that produce the
report figures, the example images, and the regression baselines. They render and
navigate in-process and need no holdings.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Command
     - Produces
   * - ``python -m tests.integration.sim_sweep_runner``
     - Per-sweep response-curve JSON under ``sim_sweeps/results/`` (gitignored)
       and the offset / star / roll / mesh figures in the report. Add
       ``--dump-images DIR`` to also write every sweep frame as a PNG.
   * - ``python -m tests.integration.technique_snr_characterization``
     - The per-technique accuracy-vs-SNR and accuracy-vs-offset report figures.
   * - ``python -m tests.integration.star_snr_characterization``
     - The star-field centroiding (moment / PSF / adaptive) report figures.
   * - ``python -m tests.integration.sim_doc_images``
     - The developer-guide scene gallery and the report scene images.
   * - ``python -m tests.integration.update_sim_baselines``
     - Regenerates the simulator regression baselines under ``sim_baselines/``.
   * - ``python -m tests.integration.update_baselines``
     - Regenerates the real-image regression baselines (needs holdings).

After a deliberate change that shifts a baseline or a figure, rerun the relevant
updater or runner and review the diff before committing -- the baselines are
tripwires, so an unexpected change is a regression to investigate, not to bless
blindly. A new invariant scene also needs a baseline (``update_sim_baselines``);
verify no existing baseline shifts.
