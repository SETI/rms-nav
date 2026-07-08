Simulator scene catalog
=======================

Each YAML here is a synthetic frame the navigator can be run against, laid out
as:

    sim_scenes/<scene_class>/<scene_name>.yaml

The directory is the registry: <scene_class> is the immediate parent directory
and must be one of the declared classes; <scene_name> must equal the filename
stem.  The schema is defined and validated by src/spindoctor/sim/scene.py; the
structural invariants are enforced by tests/integration/test_sim_scenes.py.

The fields are the flat runtime sim_params names the renderer consumes, so a
validated scene file is the sim_params dict directly (load_sim_scene returns it
unchanged; schema_version and scene_name are metadata the renderer ignores).

This mirrors the operator-curated image library (images/README.txt) but the
scenes are generated on demand by the simulator, so they augment -- never
replace -- the real-data calibration target.

Scene classes
-------------

  phase_sweep_regular_body    - an ellipsoid body at varying phase angle
  phase_sweep_irregular_body  - a mesh (irregular) body at varying phase angle
  noise_sweep                 - a fixed scene at varying read-noise level
  smear_sweep                 - varying star smear (rendering pending B3)
  range_sweep                 - a body at varying apparent size / distance
  multi_body_geometry         - controlled multi-body arrangements
  algorithmic_invariants      - clean planted-offset scenes for unit recovery

Fields
------

  schema_version   (int, required)   must be 1
  scene_name       (str, required)   must equal the filename stem
  instrument       (str, required)   a sim instrument (coiss_nac, coiss_wac,
                                     coiss_calib_nac, coiss_calib_wac, gossi,
                                     nhlorri, vgiss) or 'generic'
  size_v, size_u   (int, required)   image height/width in pixels
  random_seed      (int, required)   scene seed (drives all sim randomness)
  exposure_sec     (float, opt)      exposure seconds (default 1.0)
  offset_v, offset_u (float, opt)    planted pointing offset (px) the navigator
                                     must recover (default 0.0)
  offset_rotation_deg (float, opt)   planted boresight roll (deg, default 0.0)
  midtime_utc      (str, optional)   ISO timestamp, informational
  closest_planet   (str, optional)   ring-model planet (default SATURN)
  time, ring_epoch (float, opt)      TDB seconds for ring calculations
  shade_solid_rings (bool, optional) shade solid rings (default false)
  bodies           (list, optional)  per-body params (see below)
  rings            (list, optional)  per-ring params
  stars            (list, optional)  explicit star dicts (name, v, u, vmag, ...)
  background_stars_num (int, opt)    random background-star count (default 0)
  background_stars_psf_sigma (float) background-star PSF sigma (px)
  background_stars_distribution_exponent (float)  background-star brightness slope
  noise            (mapping, opt)    poisson, read_noise_dn, cosmic_ray_rate_per_sec,
                                     missing_data_rate, bloom_length, signal_full_scale_frac
  stray_light      (mapping, opt)    amplitude, direction_deg, model (linear|radial)
  instrument_config (mapping, opt)   per-instrument config overrides deep-merged over
                                     the named instrument's block (star_psf_sigma,
                                     data_units, noise.*, extfov_margin_vu, ...).  Omit
                                     to inherit everything; override individual keys to
                                     pin them to the scene; name 'generic' and override
                                     everything to fully self-specify so a later
                                     camera-config change cannot shift the scene.  (The
                                     top-level noise block still wins over
                                     instrument_config.noise for rendering.)
  fit_camera_rotation (bool, opt)    force whether navigation solves a camera roll

Body params follow the renderer: shape_model (ellipsoid | polyhedral_mesh),
center_v, center_u, axis1, axis2, axis3, illumination_angle, phase_angle, range,
and -- for polyhedral_mesh -- mesh_lumpiness, mesh_seed, pose_euler_deg.

A body may also carry an optional nav_override mapping.  The renderer ignores it
(it always draws the true geometry), but the navigator builds its predicted body
from the body params with nav_override overlaid -- the channel that makes the
navigation geometry diverge from the render geometry.  Use it to render an
irregular mesh yet predict its smooth (ellipsoidal) limit (mesh_lumpiness 0.0)
for a shape mismatch (B7 scenario 2), or to predict the same body at a different
pose_euler_deg for a pose disagreement (B7 scenario 3).  The override never
changes the centre, so the predicted body stays at the unshifted position the
planted offset is measured from.

The planted offset_v/offset_u is applied as the rendered offset, so a navigator
predicting the unshifted geometry must recover it (see Phase T4).

Adding a scene
--------------

Drop a new <scene_name>.yaml under the appropriate class directory.  Run
``pytest tests/integration/test_sim_scenes.py`` to confirm it validates and
renders.
