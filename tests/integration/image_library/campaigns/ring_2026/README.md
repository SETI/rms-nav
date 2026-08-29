# Ring truth campaign, 2026

Staging area for operator-verified ring sidecars from the
`util/ring_curation` shortlist.  Sidecars here carry final `ground_truth`
and use the standard sidecar schema, but they are NOT auto-enrolled in the
per-image regression test: discovery scans `images/*/*.yaml` only, so this
directory is invisible to it -- deliberately.

Why staging exists at all: enrollment under `images/<class>/` adds each
frame to `test_autonomous_nav.py`, growing suite runtime and, until the
ring-navigation routing work lands, adding standing red pins at scale.
Truth is preserved and versioned here immediately; enrollment into
`images/<class>/` happens in small reviewed batches (one PR per batch),
with `expected` pins set honestly at enrollment time.

Calibration and validation tooling may read this directory directly; the
regression suite must not.
