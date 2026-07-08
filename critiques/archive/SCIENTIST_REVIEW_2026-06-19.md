# A Research Scientist's Impressions of RMS-NAV

*Written from the point of view of a planetary scientist who would use this
software to navigate spacecraft images and build science-ready products. I am
comfortable with SPICE, photometry, ring and body geometry, and the realities
of Cassini/Voyager/Galileo/New Horizons data, but I am not a software engineer.
These are my reactions after reading the user guide, the developer guide, and
the simulator performance & sensitivity report. I have changed nothing in the
code or the documentation.*

---

## Short version

This is a genuinely impressive piece of work, and it is clearly built by people
who understand both the instruments and the science. It does the thing I most
need a navigation system to do: it takes a real archival image, predicts what
*should* be in the frame from kernels, and tells me the pointing correction
plus an honest covariance and confidence. The breadth — bodies, rings, stars,
multiple missions, backplanes, mosaics, PDS4 bundles — is exactly the toolchain
I would otherwise have to assemble myself out of half a dozen scripts. The
simulator report, in particular, gave me more confidence in five minutes than
most navigation packages earn in a month, because it states real numbers and is
candid about where things degrade.

It is not, however, a "press the button and trust the answer" black box yet, and
the documentation is honest enough that I can see exactly where the soft spots
are. Below is what I appreciate, what I would want before I published results
off it, and the things that would trip me up day to day.

---

## What I like a lot

**It reports uncertainty, and the uncertainty is principled.** Every technique
hands back a 2×2 (or 3×3 with roll) covariance and a [0,1] confidence, and the
ensemble fuses them in information form. The fact that rank-deficient cases — a
straight ring edge, a single matched star — are carried as *infinite variance in
the unconstrained direction* rather than silently dropped or faked to zero is
exactly right, and it is the kind of thing that separates a tool I can quote in a
paper from one I can't. A straight F-ring edge genuinely only constrains me
radially, and the system says so and then lets an orthogonal feature (a moon
limb, a star) supply the other axis. That matches how I actually think about the
problem.

**The simulator report is the document that sold me.** Sub-pixel recovery of a
planted offset to ~0.006 px for disc correlation, ~0.005 px for a bright star
field, ~0.03 px for ring edges, ~0.09–0.13 px for the limb — with the limb's
residual *explained* as a PSF/distance-transform effect rather than waved away —
is the level of honesty I want. The technique-selection ladder (limb → disc →
blob as a body shrinks from 130 px to 20 px, then a clean failure at 12 px) is
precisely the behavior I'd hope for, and seeing it demonstrated on ground truth
that is correct *by construction* is far more convincing than a regression
baseline someone blessed once. The noise-cliff and phase sweeps tell me where
the tool stops working, which is more useful to me than another table of where
it works.

**The technique ladder maps onto real observing geometry.** As a user I don't
want to pick an algorithm; I want to point the thing at a Mimas encounter frame,
a distant ring ansa, or a star-only calibration image and have it choose. The
autonomous orchestrator running every feasible technique and combining them is
the right default, and `--nav-techniques` / `--nav-models` globbing gives me an
escape hatch when I know better (e.g. forcing `BodyLimbNav,RingEdgeNav` on a
LORRI frame). That is a good division of labor between the tool and the human.

**The end products are the ones I actually want.** Per-pixel backplanes
(lon/lat, incidence/emission/phase, ring radius, resolution, a body-ID map),
offset-*corrected* from the navigation result rather than raw SPICE, with a
distance-aware merge between body and ring sources — that is a science-ready
product, not a demo. The mosaic side (lat/lon body grids, sparse radius/longitude
ring grids, co-rotating frames for eccentric rings so the F ring lies straight,
photometric correction options, BEST_RESOLUTION merge) shows real cartographic
thought. And PDS4 bundles out the back means I could actually archive what I
produce instead of emailing FITS files around.

**The documentation is unusually honest.** Reserved-but-unwired config keys are
labeled as such. The limb bias is described mechanistically. The confidence
numbers are explicitly flagged as *uncalibrated on clean frames*. Disabled
features (body-shadow removal, gradient-ridge refinement, ring polarity) are
called out rather than hidden. This honesty is itself a feature: it tells me what
I can lean on and what I can't.

---

## What I would need to settle before I trusted numbers in a paper

**Confidence is not yet calibrated, and the docs admit it.** The simulator report
says plainly that the per-technique confidence coefficients are uncalibrated on
clean frames and that the confidence column is flat across the navigable range —
it verifies *geometry* and *technique selection*, not the confidence tier. The
developer guide confirms the sigmoid coefficients and the A/B/C/D tiers are
hand-tuned against the simulator, with real-image tuning still to come. So today
I would treat the covariance as meaningful (it comes from the fit) but the
confidence/tier as a *relative* sorting key, not an absolute probability. For
publication I would want the calibration anchored to the operator-curated real
image library, with a statement of what tier C actually means in pixels. This is
the single thing most likely to bite a naive user who sees "confidence 0.30" and
assumes it's a 30% anything.

**The limb carries a ~0.1 px systematic, and the limb is exactly what I use for
resolved icy moons.** The report and dev guide are upfront that the body-limb fit
sits at a ~0.09–0.13 px bias floor (up to ~0.25 px two-axis worst case) because
the model predicts the geometric silhouette while the image gradient peaks
slightly inside it, and that the cleaner fix (gradient-ridge refinement) is
implemented but held *off* by default because it currently worsens limb fits.
0.1 px is below most science needs and well inside the stated bound, but it is a
*bias*, not noise, so it won't average down over a sequence. For ring-radius work
keyed off a navigated moon limb I'd want to keep that systematic in my error
budget.

**Simulated accuracy is not real-image accuracy.** Everything in the sensitivity
report is on synthetic frames with ground truth by construction. That is the
right way to test the *algorithms*, but it does not include the things that
actually wreck real navigation: stray light, ghosting, imperfect flat fields,
SPK/CK error beyond the modeled envelope, mismodeled PSF wings, and (notably) the
I/F render path is explicitly "noise-light" — no Poisson shot noise, no full-well
saturation, no bias pedestal. So the headline sub-pixel numbers are a *best case*
ceiling. I'd want to see the same accuracy story told against the curated
real-image cohort before quoting any of it.

**Stars and small rolls.** Good to know up front: a camera roll below ~0.75°
(or ~0.5° with the two-star path) is not separable from a translation, and the
star pattern matcher will return a spurious zero roll there. If I'm relying on
star fields for absolute pointing on a near-zero-roll frame, I need to know the
roll is essentially unconstrained, not measured-as-zero.

---

## Ease of use — mostly good, a few friction points

**The CLI is clean and the mental model is simple.** `nav_offset coiss N...` and
go; volume/range/list/random selectors; dry-run; a sensible JSON-metadata +
summary-PNG output pair per image. The summary PNG with model overlays is exactly
the first thing I look at to sanity-check a result, and I'm glad it's a
first-class output rather than a debug afterthought. The cloud-tasks variants and
the two-pass mosaic workflow (reproject, then combine, resumable) show this was
designed for real batch campaigns, not just one-off images.

**Things that would slow me down as a non-programmer:**

- **The instrument appendices are empty.** All four (Cassini, Galileo, New
  Horizons, Voyager) are "placeholder ... content will be added in future
  updates." For me these are the most important pages in the whole user guide —
  the per-instrument gotchas, the kernel sets I need, the known quirks of WAC vs
  NAC, the LORRI plate scale, Voyager's geometric distortion. Their absence means
  the per-instrument knowledge currently lives only in config files and in the
  developer's head. This is the biggest *usability* gap for a scientist.
- **A lot of important behavior lives in YAML I'd have to learn to read.** The
  limiting-magnitude model, ring feature definitions, body-shape uncertainties,
  the noise/edge thresholds — these are all config-file knobs. The docs describe
  them well, but the layered numeric-prefix config system (`config_03_stars.yaml`,
  `config_510_techniques.yaml`, per-instrument overrides, user overrides) is a
  developer's mental model. A short "here are the ten knobs a scientist actually
  touches and how" cheat-sheet would save me a lot of spelunking.
- **Setup is non-trivial.** SPICE kernels, PDS3 holdings, star catalogs (UCAC4,
  Tycho-2, YBSC), oops resources — each via its own env var. Reasonable for the
  domain, and remote URLs are supported, but a first run is a configuration
  exercise, not a one-liner.
- **Manual navigation reports a fixed σ = 1.0 px and confidence 1.0.** Useful as
  an operator override and for building the test library, but I shouldn't mistake
  the hand-placed offset's "confidence 1.0" for an actual 1-pixel-good answer; the
  reported uncertainty is a placeholder, not a measurement.

---

## Metadata — close to what I'd want, with one wish

What's recorded per image is rich: the offset and covariance, per-technique
results with their own covariance/confidence/diagnostics, the feature IDs each
technique consumed (so I can trace *which* limb or *which* star drove the
answer), the image-classifier output, and per-model geometry (sub-solar/sub-
observer, phase, epochs). Provenance is good and the run is deterministic given
the same kernels/image/config. The backplane and PDS4 metadata carry per-body and
per-ring min/max inventories, which is what I'd need for searchable indices.

My one real wish: **the exact config and kernel provenance should be pinned into
the per-image metadata.** The docs describe deterministic behavior *given the same
config and SPICE kernels*, but reproducibility a year later depends on knowing
*which* kernels (which CK/SPK versions) and *which* config overrides produced a
given offset. If that's captured, it should be highlighted; if it isn't, it's the
first thing I'd add, because "navigated with which pointing kernel?" is a question
I will absolutely be asked.

---

## Coverage — broad, with honest holes

- **Solidly there:** Cassini ISS, Voyager ISS, Galileo SSI, New Horizons LORRI;
  body limb/terminator/disc/blob, ring edge/annulus, star field/unique-match/
  refine; backplanes; body and ring mosaics; PDS4 bundles (with the Cassini ISS
  Saturn path as the complete reference implementation).
- **Honestly incomplete:** Titan (and atmospheric/hazy bodies generally) is a
  registered placeholder that emits no features — so no haze-limb navigation yet.
  Several config keys are reserved but unwired (curvature/roughness limb filters,
  ring fiducial promotion, body-shadow removal on rings). PDS4 templates beyond
  Cassini ISS may raise `NotImplementedError`. None of this is hidden, which I
  appreciate, but a Titan-focused or non-Cassini-PDS4 user should know going in.

---

## Would I use it?

Yes — and for a lot of my workflow I'd reach for it over rolling my own. For
geometric reconnaissance, building offset-corrected backplanes, and assembling
ring/body mosaics from Cassini and Voyager, it already does what I need and the
covariances are trustworthy. The design is coherent: the model/feature/technique/
ensemble split is the right abstraction, and it shows in how cleanly the system
degrades and how legible the results are.

Before I put a navigated *number* with a stated uncertainty into a manuscript, I
would want three things the documentation itself tells me are still open:
(1) confidence/tier calibration anchored to real images, not just simulated
frames; (2) a real-image accuracy report alongside the simulator report, so I know
how much of the sub-pixel performance survives stray light and pointing error;
and (3) the per-instrument appendices filled in, plus kernel/config provenance
pinned in the metadata. None of those are design flaws — they're the maturing
that any navigation system goes through — and the fact that the docs already name
all three is the best evidence that the people building this know exactly what's
left to do.

In short: well-architected, scientifically honest, broadly capable, and pleasant
to drive for the common cases. The covariances I'd trust today; the confidence
tiers I'd treat as relative until calibrated; the limb's 0.1-px systematic I'd
keep in my error budget; and I'd badly want those instrument appendices written.
