# Simulator (`src/nav/sim`) — findings for the rewrite branch

This is a hand-off summary of every simulator-related finding from the source
code critique (`critiques/CODE_CRITIQUE.md`, IDs `CODE-SIM-*`). The simulator is
being rewritten on a separate branch, so all of these are **deferred** in the
critique rather than fixed in place. They are collected here so the rewrite can
address (or consciously discard) each one.

Scope: `src/nav/sim/render.py`, `src/nav/sim/sim_body.py`,
`src/nav/sim/sim_ring.py`. Determinism note from the review: there is **no
time-based seeding and no module-level RNG mutation** anywhere in `src/nav/sim`
— all randomness flows through local `np.random.RandomState(seed)` seeded from an
explicit `random_seed` (default 42) or a stable `hash((axes, center))` fallback.
Seeding is correct; the findings below are about model quality / self-consistency,
not seed nondeterminism.

Line numbers are from the reviewed revision and will have drifted; treat them as
hints, not exact locations.

---

## Medium

### CODE-SIM-1 — All bodies in a combined scene share one crater seed
- **Files:** `render.py` `_render_combined_model_cached` (~line 766,
  `seed=random_seed`), `_render_single_body` (~line 384),
  `_render_bodies_positioned_cached` (~line 275).
- **Problem:** Every body is rendered with `seed=random_seed`, and
  `_render_body_shape_cached` is keyed by that seed. Two bodies with the same
  axes/shape but distinct identities get (a) the *identical* crater pattern and
  (b) a shape-cache **collision** (same key → same array returned). A per-body
  `params['seed']` is only honored when the global seed is `None`.
- **Why it matters:** Multi-body simulated scenes have correlated/degenerate
  surface texture, which can bias correlation-based navigation tests.
- **Recommended fix:** Mix the body identity into the seed (e.g.
  `seed ^ hash(body_name)` or `seed + body_index`) for both the crater RNG and
  the shape-cache key.

### CODE-SIM-2 — Craters disable limb anti-aliasing (hard step)
- **Files:** `sim_body.py` `_add_craters_and_shading` (~line 479,
  `intensity_out[~ellipse_mask_nz] = 0.0`) vs the `create_simulated_body` AA
  path (~lines 135-138) / `_lambertian_shading`.
- **Problem:** With `crater_fill > 0` the intensity is hard-masked to
  `ellipse_dist_sq < 1.0` (strict), zeroing the soft anti-aliased limb rim the
  no-crater path preserves. So `anti_aliasing` only has effect when there are no
  craters; with craters the limb is a hard step even at high `aa_scale` (the
  supersampling downsample gives only partial mitigation). The docstring's AA
  contract ("only affects the edge") is silently violated for the crater path.
- **Why it matters:** Inconsistent limb sharpness between cratered and
  non-cratered bodies; DT/limb techniques consuming simulated limbs see a
  different edge profile depending on an unrelated parameter.
- **Recommended fix:** Preserve the soft AA rim in the crater path (apply the
  same edge model as `_lambertian_shading`); unify the two shading paths (see
  SIM-3).

### CODE-SIM-3 — Crater vs no-crater shaders use divergent illumination conventions
- **Files:** `sim_body.py` `_lambertian_shading` (~lines 249-250) vs
  `_add_craters_and_shading` (~lines 459-460).
- **Problem:** The in-plane illumination unit vector is built with swapped axis
  assignments, and — critically — the crater path computes the surface normal
  from a height-field gradient with **no `rotation_z` back-rotation** applied to
  the lighting, while the smooth path rotates the normal back through
  `cos_rz/sin_rz`. For a body with non-zero `rotation_z` the lit hemisphere
  (and terminator placement) of a cratered body does not match the same body
  rendered without craters.
- **Why it matters:** For `rotation_z != 0` cratered bodies the terminator is
  wrong relative to the smooth model; any test comparing a simulated body
  against a `NavModel` limb/terminator could mis-locate. Repro: render the same
  body with `crater_fill=0` vs `crater_fill>0` at `rotation_z=pi/2`, same
  illumination — the bright sides differ.
- **Recommended fix:** Unify the two shaders behind one illumination convention;
  apply `rotation_z` to the lighting/gradient frame in the crater path.

### CODE-SIM-4 — GAP/RINGLET composition overwrites the scene instead of compositing
- **Status:** Tracked by issue **#84** ("Fix simulated ring edges and gaps").
  Listed here for completeness; the rewrite should subsume #84.
- **Files:** `render.py` `_render_combined_model_cached` GAP branch
  (~lines 741-758) and RINGLET branch (~lines 724-740).
- **Problem:** For range-ordered composition the code does
  `img[ring_mask] = ring_img[ring_mask]` (RINGLET) or
  `img[ring_mask] = temp_bg[ring_mask]` (GAP, `temp_bg` all-ones), **replacing**
  whatever was in `img` (background noise, stars, farther rings). The GAP path
  writes `1.0 - gap_coverage` over the real scene, so a partial gap leaves a
  near-white patch instead of darkening. The single-ring `render_ring`
  (`sim_ring.py` ~lines 413, 440) correctly adds/subtracts; the combined
  compositor does not use that additive path for range ordering.
- **Why it matters:** Background/underlying features vanish under rings and gaps
  brighten instead of darken in multi-layer simulated scenes (rings + noise +
  stars).
- **Recommended fix:** Composite additively/subtractively (reuse the single-ring
  add/subtract path) rather than overwriting pixels under the ring footprint.

---

## Low

### CODE-SIM-5 — Inner `_render_*` caches are `lru_cache(maxsize=1)`
- **File:** `render.py` (`_render_stars_cached`, `_render_bodies_positioned_cached`,
  `_render_background_noise_cached`, `_render_background_stars_cached`,
  `_render_combined_model_cached`), all `maxsize=1`.
- **Problem:** Each inner cache holds one entry across a multi-stage pipeline, so
  alternating between two scenes (GUI toggling `ignore_offset`, an offset sweep)
  evicts on every call — the caches rarely hit and add `json.dumps(..., sort_keys=True)`
  overhead on the hot path.
- **Recommended fix:** Increase `maxsize` (or drop the ineffective inner caches);
  reconsider the JSON-key serialization cost.

### CODE-SIM-6 — `render_stars` / `render_bodies` are dead public API that leak cached mutable objects
- **File:** `render.py` `render_stars` (~138-152), `render_bodies` (~435-491).
- **Problem:** Only `render_combined_model` is imported outside `render.py`.
  `render_stars` returns `cached_star_list` straight from the `lru_cache`d
  `_render_stars_cached` **without copying**, so an external caller would mutate
  shared cached `MutableStar` objects. `render_bodies` is similarly unused and
  duplicates logic already inlined in `_render_combined_model_cached`.
- **Recommended fix:** Delete both (or route the combined path through them and
  return copies). In the rewrite, ensure any public render entry point returns
  defensive copies of cached mutable state.

### CODE-SIM-7 — Body inventory bbox uses `max(axis1,axis2,axis3)/2` for both axes
- **File:** `render.py` `_render_single_body` (~417-426),
  `_render_bodies_positioned_cached` (~314-323).
- **Problem:** `max_dim = max(axis1, axis2, axis3) / 2.0` is used as the
  half-extent for *both* v and u, so for a non-circular or tilted ellipsoid the
  reported inventory bbox does not match the rendered silhouette.
- **Why it matters:** Diagnostic only — downstream consumers of `inventory`
  (hit-testing, fixtures) get a coarse bbox.
- **Recommended fix:** Compute a proper per-axis projected bounding box that
  accounts for anisotropy and tilt.

### CODE-SIM-8 — Star-flux zero-point comment vs default `vmag` mismatch
- **File:** `render.py` (~lines 59, 116).
- **Problem:** `star.dn = 2.512 ** -(star.vmag - 4.0)`; `scale_factor =
  star.dn / (2.512**4.0)` reduces to `2.512**-vmag`, so peak=1 at vmag=0 — but
  the comment says "vmag=4 -> peak=1", and the default `vmag=8` then renders a
  near-black star (peak ~= 0.0006). The model is internally consistent but the
  documented zero-point and the default are inconsistent.
- **Why it matters:** Confusing photometry; default stars render essentially
  invisible.
- **Recommended fix:** Decide the intended zero-point/dynamic range, fix the
  comment to match the math, and pick a default `vmag` that renders visibly (or
  document why the default is faint).

### CODE-SIM-9 — `e >= 1.0 -> e = 0.99` silently clamps invalid ring eccentricity
- **File:** `sim_ring.py` `compute_edge_radius_at_angle` (~96-97),
  `_compute_edge_radii_array` (~137-138).
- **Problem:** When `ae/a >= 1` the eccentricity is silently clamped to 0.99
  with no warning, so a caller that mis-specifies `ae`/`a` gets a
  plausible-but-wrong ellipse instead of an error.
- **Why it matters:** Wrong simulated ring radius with no diagnostic (sim-only).
- **Recommended fix:** Raise (or at least warn) on a physically impossible
  eccentricity rather than silently clamping.

---

## Cross-cutting suggestions for the rewrite

- **Single shading path.** SIM-2 and SIM-3 both stem from having two
  near-duplicate body shaders (crater vs no-crater) with subtly different
  illumination conventions and edge handling. A single shading routine that
  optionally adds a crater height-field would eliminate both classes of bug.
- **Per-body identity in all seeds and cache keys** (SIM-1) — thread the body
  name/index everywhere a seed or shape-cache key is derived.
- **Additive/subtractive compositing** for the combined scene (SIM-4) instead of
  pixel replacement, so layers (noise, stars, rings, bodies) coexist.
- **Defensive copies** out of any cached render result that returns mutable
  objects (SIM-6).
- **Cache sizing / serialization cost** review (SIM-5).
