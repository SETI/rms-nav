# RMS-NAV: A Skeptic's Teardown

*Written deliberately as a hostile reviewer — the referee who wants to reject the
paper, the PI deciding whether to bet a mission's calibration pipeline on this
code. I went into the source, not just the docs. Nothing below is invented; every
claim is anchored to a file. But I have given everything the least charitable
reading it can honestly bear, because that is the job. If a number looks good, I
asked what it is hiding.*

---

## The one finding that should stop you

**The headline accuracy numbers are very close to circular. The simulator
"validates" the navigator against images drawn from the navigator's own forward
model.**

The sensitivity report's marquee results — disc correlation recovering a planted
offset to ~0.006 px, a bright star field to ~0.005 px, ring edges to ~0.03 px —
are presented as accuracy benchmarks. They are not. Look at what generates the
test image versus what the navigator fits:

- The test-image renderer, `src/nav/sim/render.py`, builds frames by calling
  `create_simulated_body`, `render_mesh_body_image`, and `render_ring`
  (`render.py:14,18,20`).
- The navigator's *prediction* for those same frames,
  `src/nav/nav_model/nav_model_body_simulated.py` and
  `nav_model_rings_simulated.py`, builds its model by calling the **same
  functions** — `create_simulated_body`, `render_mesh_body_image`
  (`nav_model_body_simulated.py:26-27`) and `render_ring`
  (`nav_model_rings_simulated.py:29`).

So the "image" and the "model" come out of one piece of code. The only thing
that differs between truth and prediction is a planted sub-pixel shift. The
experiment therefore measures one thing: *can a registration routine recover a
known translation between two nearly identical synthetic arrays?* Of course it
can, to a few thousandths of a pixel. There is **no PSF mismatch, no shape
mismatch, no albedo or limb-darkening mismatch, no ephemeris error, and no
realistic detector noise** between model and data, because they share a parent.

This is not a navigation accuracy result. It is a self-consistency check wearing
the costume of one. The report's own hedging ("treat the recovery errors as
comfortably within the stated bound") quietly concedes the numbers aren't even
reproducible across machines — but the deeper problem is that they would be
near-perfect *by construction* even if the underlying physics model were wrong,
because the same wrong model is on both sides of the comparison.

**Consequence:** every quantitative performance claim in the project — the whole
`simulator_report.rst` — tells you about the fitter's internal numerics and
nothing trustworthy about how well RMS-NAV will point a real Cassini frame.

---

## There is no real-image accuracy assessment anywhere

Follow the obvious next question — "fine, how does it do on *real* images?" — and
you hit a void. The `docs/` tree contains exactly one performance document, the
simulator report. There is no `docs/*real*`, no `*accuracy*` against archival
data, nothing. The entire quantitative validation of a *spacecraft image
navigation system* is run on synthetic frames it generated itself.

And the real-image safety net is threadbare:

- The "operator-curated image library" — the regression cohort the whole project
  leans on for real-world correctness — is **thirteen YAML sidecars**
  (`tests/integration/image_library/`, counted directly). Its `README.txt` is
  empty. A dozen-odd hand-blessed frames is the sum total of real-image ground
  truth.
- **CI never runs it.** `.github/workflows/run-tests.yml:94` runs
  `pytest ... -m "not integration"`. The integration suite — i.e., the only tests
  that touch a real image — is explicitly excluded on every PR and every push.
  The comment says to "run the integration suite locally."

Translation: every merge goes green without a single real image being navigated.
Correctness on archival data rests on a human remembering to run a slow,
holdings-dependent, self-described "flaky" suite by hand "before merging anything
that could plausibly regress accuracy." That is not a regression guarantee; it is
an honor system. The advertised "90% line coverage" is a vanity number — it
covers the unit suite, which by construction cannot exercise navigation accuracy
on real data.

---

## It ships a "confidence" it admits is meaningless

Every `_metadata.json` carries a per-image `confidence` in [0,1] and an A/B/C/D
`confidence_tier`. A scientist will read those as probabilities or quality
grades. They are neither. The report states outright that "the per-technique
confidence coefficients are uncalibrated on these clean frames" and that "the
confidence column is flat across the navigable range." The dev guide confirms the
sigmoid coefficients and tier thresholds are hand-tuned — and hand-tuned against
the *circular simulator* described above. So the project tunes a confidence model
to a synthetic benchmark that can't measure accuracy, then emits the result as a
headline field. Shipping an authoritative-looking number you know is
uncalibrated is worse than shipping nothing: it manufactures false trust.

---

## This is an unfinished rewrite wearing a finished manual

The polished, comprehensive documentation describes a system more complete than
the code behind it.

- The branch is `core_rewrite_phase10_sim`. The recent log is full of
  `feat:`/`docs:` churn ("Phase 10," "core rewrite phase 10"). This is a project
  *mid-rewrite*, not a settled release.
- The repository's own canonical guidance contradicts the shipped docs. The
  top-level `CLAUDE.md` states "the remaining correlation / star techniques are
  **pending**," while the user guide presents `StarFieldFromCatalogNav`,
  `StarUniqueMatchNav`, and `StarRefineNav` as shipping features. Either the
  authoritative project file is stale or the docs are ahead of reality; either way
  the project cannot keep its own story straight about what exists.
- The documentation is padded with forward-looking vapor. Across `docs/` there
  are *hundreds* of instances of "placeholder," "reserved for," "pending,"
  "deferred," "not yet implemented," and "future enhancement." A manual that has
  to say "reserved for a future enhancement" that many times is documenting
  aspirations, not software.

---

## Capability gaps that matter to actual Saturn science

- **Titan is a no-op.** `nav_model_titan.py` is a registered placeholder whose
  `to_features()` literally `return []` (`nav_model_titan.py:48`). Titan — one of
  the most-imaged, most-studied targets in the entire Cassini archive — gets
  *zero* navigation support. The docstring waves at "a haze-aware limb-fit
  technique ... out of scope." For a system pitched at Saturn-system imagery, that
  is a gaping hole, not a footnote.
- **PDS4 is largely fictional.** `dataset_pds4.py` raises
  `NotImplementedError('PDS4 datasets are not yet implemented')` for every method
  — there is no PDS4 *input* path at all. And PDS4 *bundle generation* works only
  for Cassini ISS: Voyager, Galileo, and New Horizons dataset classes raise bare
  `NotImplementedError` for the `pds4_*` hooks
  (`dataset_pds3_voyager_iss.py:226+`, `dataset_pds3_galileo_ssi.py:266`,
  `dataset_pds3_newhorizons_lorri.py:222+`). "Generates PDS4 bundles" is a
  Cassini-only claim dressed as a general one.
- **The per-instrument appendices — the pages a user needs most — are empty.**
  All four (Cassini, Galileo, New Horizons, Voyager) are one-line "placeholder ...
  content will be added in future updates." Every per-instrument quirk (NAC vs WAC,
  LORRI plate scale, Voyager geometric distortion, kernel sets) lives only in YAML
  and the author's head. A new user on anything but the author's daily-driver
  instrument is on their own.

---

## The numbers a scientist would actually quote are built on hand-picked constants

The covariance is the one output I'd put in a paper. It is shaped throughout by
magic numbers:

- `ROTATION_UNOBSERVABLE_VARIANCE = 1.0e15` (`nav_technique.py:46`) — a "finite
  proxy for infinity" sentinel threaded straight into the covariance algebra.
- `DEFAULT_PINVH_RCOND = 1.0e-9` (`dt_fitting.py:94`) — one project-wide
  pseudoinverse cutoff that decides which directions of every fit are declared
  rank-deficient.
- Star detectability and covariance derive from a **fabricated** SNR:
  `snr_eff = SNR_REF * 2.512 ** (mag_limit - vmag)` with `SNR_REF = 8.0`,
  `SNR_FLOOR = 0.1` (`nav_model_stars.py:77-78,246-247`). The star "signal-to-
  noise" that gates stars in and out and feeds their error bars is not measured
  from the photometry in the frame — it is synthesized from catalog magnitude
  against a hand-chosen reference constant. Get the per-instrument limiting
  magnitude slightly wrong and the whole star error budget is wrong, silently.

None of these are inherently illegitimate, but collectively they mean the error
bars are as much a product of tuning choices as of the data. A referee is
entitled to ask which.

---

## Known systematic, no working fix

The body limb — the workhorse for resolved icy satellites — carries a
**systematic bias of ~0.09–0.13 px, up to ~0.25 px** on two axes, because the
model predicts the geometric silhouette while the image gradient peaks inside it.
The implemented remedy, gradient-ridge refinement, is **shipped disabled**:
`config_510_techniques.yaml:237` sets `gradient_ridge_refine: 0` for the limb
(and `:343` for ring edges), with the comment that it is "Held OFF" because it
makes limb fits worse. So the team has identified the bias, written a fix, and
turned the fix off because it doesn't work yet. The current accuracy on the most
important resolved-body technique is "good enough, with a known bias we can't
cleanly remove" — and that bias does not average down over a sequence.

Worse, the dev guide's own explanation is that integer DT quantization plus Tukey
reweighting "accidentally" pulls the fit back toward truth. Relying on a partial
cancellation between two unrelated effects is not a property you want underneath
a published radius measurement.

---

## Operational realities that will hurt

- **It is slow.** A single 1024-px navigation costs ~35 s (`simulator_report.rst:71`)
  — and the report dodges this by running its sweeps at 220 px "for tractable
  runtime." A full Cassini ISS campaign is hundreds of thousands of NAC frames;
  at 35 s each that is months of wall-clock. Star matching is O(M^3) in source
  count on top of that.
- **The obvious fix for slowness is mined.** The reprojection path mutates
  `oops` global precision and builds shared `Backplane` objects; the dev guide
  warns it is *not thread-safe* on a shared observation. So naive parallelization
  — the first thing anyone reaches for — corrupts results.
- **Small rolls are silently wrong, not flagged.** Below ~0.75° the star pattern
  matcher returns a *spurious zero roll* (per the report's own roll table), not an
  "unobservable" flag. A user near zero roll gets a confident-looking 0.0° that is
  simply incorrect.
- **The calibrated (I/F) path is essentially untested.** The report concedes the
  I/F render is "noise-light" — no Poisson shot noise, no full-well saturation, no
  bias pedestal. Many users navigate calibrated products; that regime has no
  realistic test behind it.
- **Garbage in, garbage out, with no audit trail.** The offset is purely relative
  to nominal SPICE pointing; the system never validates the kernels and (as far as
  I can find) does not pin the CK/SPK versions or config overrides into the output
  metadata. Absolute accuracy is entirely hostage to inputs the tool doesn't
  record, so a result a year old may not be reproducible or even attributable to a
  pointing kernel.

---

## Bottom line

Strip away the genuinely good engineering hygiene and the unusually pleasant
prose, and what is left is a **mid-rewrite system whose only quantitative
validation is circular, whose real-image testing is a dozen frames that CI never
runs, that ships an uncalibrated "confidence" as if it meant something, that
cannot navigate Titan, that supports PDS4 for one instrument, that documents four
empty instrument appendices, and that carries a known, unfixed sub-pixel bias on
its primary resolved-body technique.** None of that is hidden — the docs are
admirably candid — but candor about a weakness is not the same as not having it.
The honesty makes the project pleasant to audit; it does not make the system
ready to anchor science.

I would not put a navigated offset and its error bar from this tool into a
manuscript today without independently re-deriving the uncertainty, and I would
not stand up a production pipeline on it until (1) there is a real-image accuracy
study against ground truth the navigator did **not** generate, (2) the
integration cohort is large enough and runs in CI, and (3) the confidence output
is either calibrated or removed. Until then, treat the impressive numbers as what
they are: a renderer admiring its own reflection.
