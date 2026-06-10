==========
Provenance
==========

Overview
========

:py:class:`~nav.nav_orchestrator.provenance.Provenance` is the frozen reproducibility envelope
attached to every :py:class:`~nav.nav_orchestrator.nav_result.NavResult`.  It records the exact
inputs that produced a navigation: the package version and git SHA, the SPICE kernels that were
actually loaded, a content hash of every shipped static-data YAML, the registered technique and
extractor class names, the observation midtime, and the wall-clock run timestamp.

The envelope is populated at navigate time by the orchestrator's ``_make_provenance``, which gathers
the runtime-derived fields through the free helper
:py:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata` and combines them with the
version, midtime, and registry names.  The consumers are the curator (which writes the envelope
into the per-image JSON metadata) and the regression-baseline comparator (which compares two
envelopes for byte equality after stripping the run timestamp).

Theory
======

The envelope encodes one reproducibility guarantee: two navigations of identical inputs produce
byte-identical provenance except for the wall-clock run timestamp, which varies by construction.
Determinism is enforced by normalising every sequence field to a sorted tuple, by reducing SPICE
kernel paths to bare basenames so install-root differences do not leak in, and by sorting the
static-data hash mapping by filename.  Beyond that normalisation the envelope is an inert record;
it runs no navigation algorithm.

Configuration
=============

:py:class:`~nav.nav_orchestrator.provenance.Provenance` has no configuration.  It is populated at
navigate time by :py:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata`; no YAML
knobs apply.  The one fixed list it depends on is the set of filename prefixes that mark a config
file as static data for hashing, defined as a module-level constant.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/provenance.py``.  Public class
:py:class:`~nav.nav_orchestrator.provenance.Provenance`, a frozen
:py:func:`dataclasses.dataclass`.  The module also exposes the helper class
:py:class:`~nav.nav_orchestrator.provenance.ProvenanceMetadata` (the runtime-derived subset) and the
free function :py:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata`.

The public fields are :py:attr:`~nav.nav_orchestrator.provenance.Provenance.rms_nav_version`,
:py:attr:`~nav.nav_orchestrator.provenance.Provenance.image_et` (observation midtime ET),
:py:attr:`~nav.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601` (the run timestamp
excluded from baseline comparison),
:py:attr:`~nav.nav_orchestrator.provenance.Provenance.rms_nav_git_sha` (short SHA, ``'dirty'``
marker, or ``None``), :py:attr:`~nav.nav_orchestrator.provenance.Provenance.spice_kernels` (sorted
basenames), :py:attr:`~nav.nav_orchestrator.provenance.Provenance.static_data_hashes` (filename to
sha256-hex), :py:attr:`~nav.nav_orchestrator.provenance.Provenance.technique_names`,
:py:attr:`~nav.nav_orchestrator.provenance.Provenance.extractor_names`, and the non-init derived
field :py:attr:`~nav.nav_orchestrator.provenance.Provenance.spice_kernel_count`.

The :py:meth:`~nav.nav_orchestrator.provenance.Provenance.__post_init__` invariant normalises the
three sequence fields to sorted tuples, derives the kernel count from the kernel tuple length, and
freezes the static-data hash mapping into a read-only :py:class:`types.MappingProxyType`.
:py:class:`~nav.nav_orchestrator.provenance.ProvenanceMetadata` carries its three runtime fields
without further invariants.  The public helper
:py:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata` assembles a
:py:class:`~nav.nav_orchestrator.provenance.ProvenanceMetadata` from the module's private resolvers
for the git SHA (via ``git rev-parse`` and ``git status``), the loaded SPICE kernel basenames (via
the ``cspyce`` binding when available), and the static-data hashes (sha256 over the raw bytes of
every config file whose name matches a recognised prefix).  Each resolver is best-effort: an
unavailable git, an unavailable SPICE binding, or a per-file read error yields an empty or partial
result rather than aborting the run.

Examples
========

A populated envelope for one navigation looks like the following field values:

.. code-block:: yaml

    rms_nav_version: "0.5.2"
    rms_nav_git_sha: "8a6b607"
    image_et: 487105349.2
    pipeline_run_iso8601: "2026-06-10T17:42:03Z"
    spice_kernels:
      - "cas00171.tsc"
      - "naif0012.tls"
      - "sat441.bsp"
    spice_kernel_count: 3
    static_data_hashes:
      config_220_body_shape.yaml: "9f2c..."
      config_400_inst_coiss.yaml: "ab14..."
    technique_names:
      - "BodyLimbNav"
      - "BodyTerminatorNav"
    extractor_names:
      - "NavModelBody"

Re-running the same image produces an identical envelope except for
:py:attr:`~nav.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601`, which advances with the
wall clock; the baseline comparator strips that field before asserting equality.
