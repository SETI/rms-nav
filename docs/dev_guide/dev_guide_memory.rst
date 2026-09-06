=================
Bounding a Run
=================

A navigation's memory is dominated by the backplanes it evaluates, and a backplane
is sized by the meshgrid it is handed rather than by the answer it returns. ``oops``
materializes its intermediates over that meshgrid at roughly a kilobyte per pixel,
so a stage that hands it the extended frame pays for the extended frame whatever
the question was. The extended frame is the detector plus twice the instrument's
search margin, which is what makes the wide-margin instruments the expensive ones:
a Voyager frame extends to 1800 x 1800, or 3.24 megapixels, against a 1000 x 1000
detector.

Two mechanisms keep that bounded. They are independent, and a stage needs both.

Striping
========

A backplane is a per-pixel function of the ray through each pixel, so a band of
rows can be evaluated on its own and the bands stacked into the array a whole-frame
evaluation would have returned. Only one band's intermediates exist at a time, so
the live heap is set by the strip height rather than by the frame.

The stacked array is the whole-frame array exactly, not an approximation of it:
strip boundaries fall on whole rows, and no per-pixel quantity depends on a
neighbouring row. Tests assert the identity rather than a tolerance.

Three places stripe: ``NavModelRings._striped_backplane`` for the ring
quantities, ``nav_model_body._body_strips`` for a body's oversampled box, and
``titan_geometry._striped_occlusion`` for both occlusion masks over one set of
strips. Each caps a strip at 128 rows.

A caller must take everything it needs from a strip while that strip is the one in
hand. Asking again afterwards rebuilds the whole box and gives back nothing.

Releasing
=========

Striping bounds the live heap. It does not on its own bound the process's resident
size, and resident size is the number that matters: it is what the kernel's
out-of-memory killer reads and what a recorded peak reports.

Two things sit between the two, and neither is sufficient alone. The intermediates
are held in reference cycles, so dropping the last name bound to a strip frees
nothing until a collection runs, and a collection of the oldest generation is not
otherwise due within the handful of allocations a strip costs. Once freed, the C
allocator keeps the arenas rather than returning them, so the address space is
still charged to the process.

Left alone the two compound, and a run's resident size grows by the *sum* of its
strips rather than by the largest of them -- which is the quantity striping exists
to reduce. A striped pass that does not release is a pass whose striping cannot be
observed from outside.

:func:`~spindoctor.support.memory.release_transient_memory` does both halves.
Measured over one striped ring pass on a Voyager Saturn frame:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Released with
     - Resident growth
     - Wall time
   * - neither
     - 2.84 GB
     - 47.5 s
   * - collection only
     - 1.39 GB
     - 41.9 s
   * - arena release only
     - 2.45 GB
     - 45.9 s
   * - both
     - 0.00 GB
     - 45.3 s

The results are identical in every row; only the resident size differs. The release
costs nothing worth counting, because a strip is expensive enough that a collection
between strips does not register against it.

Where it is called
------------------

After each strip, in each of the three striped loops; and once more where the
model stage ends, after the caches it filled have been dropped.

A release on its own reclaims only what has stopped being referenced, so where
it is called is a claim about what has just stopped being needed. Inside a
striped loop that is the strip. At the end of the model stage it is the
backplanes, and only because they are dropped in the same breath -- a release at
that boundary that dropped nothing was measured at 0.04 GB, which is what a
release finds when everything around it is still held.

This is deliberately not wired into a general allocation path. A collection is
cheap against a backplane evaluation and expensive against a small one, so it
belongs only where a large unit of work has just ended.

What the models leave behind
============================

A ring render leaves several gigabytes resident on a Voyager Saturn frame after a
collection and an arena release have both run, and all of it is held.

Ask the C library and it says so plainly. On the heaviest measured frame, once
:func:`~spindoctor.support.memory.release_transient_memory` has run, ``mallinfo2``
reports 0.23 GB free and retained against 3.16 GB handed out and 3.46 GB in
mappings, and CPython's own small-object allocator holds 0.10 GB across 99 arenas.
Almost none of it is fragmentation. It is live.

An ``oops`` ``Backplane`` caches every event, intercept and computed surface it has
ever been asked for, and it is sized by the meshgrid rather than by the answer, so
on the extended frame each entry is megabytes and there are hundreds. Emptying them
one kind at a time on that frame:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Cache
     - Entries
     - Returns
   * - computed backplanes
     - 237
     - 1.89 GB
   * - unmasked intercept events
     - 2
     - 1.26 GB
   * - observation events, with and without line-of-sight derivatives
     - 2
     - 1.88 GB

Settled resident size falls from 6.80 GB to 1.77 GB. The observation builds each
backplane lazily and caches it, so dropping them all -- which is what
:meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.reset_all` does -- costs only the
recomputation of whatever is asked for next.

Nothing asks. The models are the only stage that reads a backplane; no technique,
and nothing in the orchestrator below them, touches one. So the release goes where
the model stage ends, after the features have been extracted and the annotations
drawn, and the techniques run against an observation holding nothing.

Two dead ends, recorded so they are not tried again. Pinning the C library's mmap
threshold, so large arrays are served by mappings returned on free rather than from
the heap, moves about a third of a gigabyte for about six percent more runtime.
Walking ``gc.get_objects()`` for live arrays reports approximately zero against any
residue at all: numeric NumPy arrays are not tracked by the collector, so that walk
cannot see them, and a probe built on it will report an empty heap under six
gigabytes of live data.

Correlation
===========

Once the models have given their memory back, the largest thing left in the process
is the masked normalized cross-correlation in
:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`. It
correlates the extended frame against a template of the same size, zero-padded for
linear rather than circular correlation, so the surfaces it works on are twice the
extended frame on each axis: 3600 x 3600 for a Voyager frame, or about a tenth of a
gigabyte each. It needs six of them at once for the normalization, and everything on
the way to them is a transform of the same size.

Three things keep that down, none of which changes what is computed.

The transforms are of real fields, so half of a full spectrum is the conjugate of
the other half. ``_correlate_from_spectra`` uses half-spectrum transforms
throughout, which halves every spectrum and is also about twice as fast.

The result of a correlation is real by construction, and ``np.real`` of a complex
array is a strided *view* -- so a surface obtained that way keeps a complex array of
twice its size alive for as long as the surface is needed, six times over. A
half-spectrum inverse returns the contiguous real array directly.

Each spectrum is built where it is first needed and dropped at its last use: the
mask spectrum is finished after the second shift-wise sum, the image spectrum after
the third, the model-mask spectrum after the fourth. The normalization that follows
writes into surfaces that have just stopped being needed rather than allocating one
per step, because a step allocates as much as a whole surface and there are a dozen
of them.

Measured on one call at Voyager's padded size, against random ring-like fields:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Correlation surfaces built with
     - Resident growth
     - Wall time
   * - full spectra, real parts as views
     - 2.16 GB
     - 18.5 s
   * - full spectra, real parts copied out
     - 1.75 GB
     - 17.9 s
   * - half spectra
     - 1.07 GB
     - 8.6 s

The middle row is bit-identical to the first. The half-spectrum row agrees with it
to 9e-16 on the correlation surface, marks exactly the same shifts invalid, and puts
the peak in the same place; the discarded imaginary part was rounding noise. Tests
check the result against a transform-free evaluation of the same sums.

Declining early
===============

The cheapest backplane is the one never built. A body whose disc reaches past all
four corners of the extended frame leaves no sky around it, no limb inside the
frame, and no measurable extent, so there is nothing a shape-based technique could
match. :func:`~spindoctor.nav_model.nav_model_body.body_fills_extfov` decides that
from the inventory alone -- an ellipse against the frame corners, costing no
backplane -- and the model declines before building anything. The navigation records
``body_fills_fov`` as its reason, which separates a frame that is unnavigable by
that body from one that merely failed.

:func:`~spindoctor.nav_model.nav_model_rings.NavModelRings._sparse_visibility_skip`
is the same idea for the rings: a 16 x 16 evaluation rules out the two common cases
-- no ring-plane intersection anywhere in the frame, and a visible radial range
entirely outside the catalogue's outermost feature -- without paying for a dense
backplane.
