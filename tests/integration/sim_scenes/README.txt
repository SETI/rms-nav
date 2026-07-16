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
  smear_sweep                 - varying star smear
  range_sweep                 - a body at varying apparent size / distance
  multi_body_geometry         - controlled multi-body arrangements
  algorithmic_invariants      - clean planted-offset scenes for unit recovery
  regression                  - pinned scenes reproducing fixed defects

Fields
------

  schema_version   (int, required)   must be 2
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
  noise            (mapping, opt)    poisson, read_noise_dn, bias_dn,
                                     cosmic_ray_rate_per_sec, missing_data_rate,
                                     bloom_length, signal_full_scale_frac, pixel_area_cm2,
                                     dark_current_e_per_sec, hot_pixel_fraction,
                                     hot_pixel_amplitude_e, hot_pixel_column_factor,
                                     banding_amplitude_e, banding_period_px,
                                     bias_pedestal_sigma_dn, bias_row_gradient_dn,
                                     bias_col_gradient_dn, vidicon {read_noise_line_dn,
                                     read_noise_pixel_dn, coherent_amplitude_dn,
                                     coherent_period_px}; unknown keys fail validation
  oversample       (int, opt)        radiance oversampling factor (default 4 when a PSF
                                     is active, else 1)
  optics           (mapping, opt)    whole-scene optical effects: psf (sigma_v, sigma_u,
                                     w, r0, n; or match_navigator, preserved as authored
                                     and resolved by the renderer), smear (list of
                                     dv_px/du_px/object_class), distortion (k1, k2,
                                     center_v, center_u, nonradial_rms_px), ghosts (list
                                     of dv_px/du_px/amplitude/defocus_sigma), stray_light
                                     (amplitude, direction_deg, model linear|radial)
  detector         (mapping, opt)    detector-chain override: gain_state (must be
                                     catalogued for the instrument), detector_model
                                     (ccd | vidicon), exposure_ref_sec, quantization
                                     (exact | 8bit | uneven_12bit | sqrt_lut); omitted
                                     keys track the instrument catalog
  artifacts        (mapping, opt)    instrument_defaults (bool): opt into the emulated
                                     camera's physical signal chain at catalog values
                                     (PSF, distortion residual, shot noise, dark / hot /
                                     bloom / banding / bias); loss modes stay 0
  spk_error        (mapping, opt)    planted spacecraft-ephemeris parallax: dv_px, du_px,
                                     reference_range_km (needs range_km on each body/ring)
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
center_v, center_u, axis1, axis2, axis3, illumination_angle, phase_angle,
range_km, and -- for polyhedral_mesh -- mesh_lumpiness, mesh_seed,
pose_euler_deg.

A body may also carry an optional nav_override mapping.  The renderer ignores it
(it always draws the true geometry), but the navigator builds its predicted body
from the body params with nav_override overlaid -- the channel that makes the
navigation geometry diverge from the render geometry.  Use it to render an
irregular mesh yet predict its smooth (ellipsoidal) limit (mesh_lumpiness 0.0)
for a shape mismatch, or to predict the same body at a different pose_euler_deg
for a pose disagreement.  The override never changes the centre, so the
predicted body stays at the unshifted position the planted offset is measured
from.

The planted offset_v/offset_u is applied as the rendered offset, so a navigator
predicting the unshifted geometry must recover it.

Adding a scene
--------------

Drop a new <scene_name>.yaml under the appropriate class directory.  Run
``pytest tests/integration/test_sim_scenes.py`` to confirm it validates and
renders.
