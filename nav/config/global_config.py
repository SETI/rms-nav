import numpy as np


ALL_PLANETS = ['MERCURY', 'VENUS', 'EARTH', 'MARS',
               'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE',
               'PLUTO']


# python nav_main_offset.py --force-offset --png-blackpoint 0 --moons-only  N1658884355_1 --bodies-label-font /usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf,30 --metadata-label-font /usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf,25

PNG_INCLUDE_IMG = True
PNG_INCLUDE_MODEL = True
PNG_FORCE_MODEL_WHITE = False
PNG_INCLUDE_MODEL_TEXT = True
PNG_BODY_OUTLINE = True
PNG_INCLUDE_METADATA = True
CORR_ALLOW_SUBMODELS = False


########################
# BODY CHARACTERISTICS #
########################

# These are bodies large enough to be picked up in an image.
LARGE_BODY_LIST = {
    'JUPITER': ['JUPITER', 'AMALTHEA', 'IO', 'EUROPA', 'GANYMEDE', 'CALLISTO'], # XXX
    'SATURN': ['SATURN', 'PAN', 'DAPHNIS', 'ATLAS', 'PROMETHEUS',
               'PANDORA', 'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
               'TETHYS', 'TELESTO', 'CALYPSO', 'DIONE', 'HELENE',
               'RHEA', 'TITAN', 'HYPERION', 'IAPETUS', 'PHOEBE',
               'VENUS', 'EARTH', 'MARS', 'JUPITER', 'URANUS', 'NEPTUNE'],
}

# These are bodies large enough to be picked up in an image.
# LARGE_BODY_LIST_TITAN_ATMOS = [x if x != 'TITAN' else 'TITAN+ATMOSPHERE'
#                                for x in LARGE_BODY_LIST] # XXX

# These are bodies that shouldn't be used for navigation because they
# are 'fuzzy' in some way or at least don't have a well-defined orientation.
FUZZY_BODY_LIST = ['HYPERION', 'PHOEBE'] # XXX

# These are bodies inside the rings that should not be used to compute
# ring occlusions.
RINGS_BODY_LIST = ['PAN', 'DAPHNIS'] # XXX


##################
# CONFIGURATIONS #
##################

STARS_CONFIG = {
    # True if data is already calibrated as I/F and needs to be converted back
    # to raw DN.
    'calibrated_data': True,

    # True if we want to apply stellar aberration to the star positions in the
    # catalog.
    'stellar_aberration': True,

    # Allow the PSF Gaussian size to float during PSF fitting instead of using
    # the official PSF size.
    'float_psf_sigma': False,

    # The order of multipliers to use to gradually expand the search area.
    'search_multipliers': [0.25, 0.5, 0.75, 1.],

    # Maximum number of stars to use.
    'max_stars': 30,

    # Verify offset with photometry?
    'perform_photometry': True,

    # If using photometry, try again at the end without using it?
    # This can be overriden later if there are nothing but stars in the FOV,
    # in which case we always try without photometry.
    'try_without_photometry': False,

    # Minimum number of stars that must photometrically match for an offset
    # to be considered acceptable and the corresponding confidence.
    # Also the minimum number of stars that must match to give a confidence
    # of 1.0.
    'min_stars_low_confidence': (3, 0.75),
    'min_stars_high_confidence': (6, 1.0),

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # PSF size for modeling a star (must be odd). The PSF is square.
    # This will be added to the smearing in each dimension to create a final
    # possibly-rectangular PSF.
    'psf_boxsizes': ((100000,15),
                     ( 50000,13),
                     (   500,11),
                     (     0, 9)),

    # The DN at which the PSF gets expanded, and how much to expand it by.
    'psf_gain': (5000, 4),

    # The maximum number of steps to use when smearing a PSF. This is really
    # only a suggestion, as the number will be clipped at either extreme to
    # guarantee a good smear. We limit the number of steps for performance.
    'max_movement_steps': 50,

    # The maximum amount of smearing to tolerate before giving up on star
    # navigation entirely.
    # OLD: This is currently set low because the smear angles
    #      are wrong thanks to SPICE inaccuracies.
    # NEW: This is currently set high because we have access to the
    #      predicted kernels.
    'max_smear': 100,

    # The default star class when none is available in the star catalog.
    'default_star_class': 'G0',

    # The minimum DN that is guaranteed to be visible in the image.
    'min_brightness_guaranteed_vis': 200.,

    # The minimum DN count for a star to be detectable. These values are pretty
    # aggressively dim - there's no guarantee a star with this brightness can
    # actually be seen. But there's no point in looking at a star any dimmer.
    ('min_detectable_dn', 'NAC'): 7,
    ('min_detectable_dn', 'WAC'): 10,

    # The amount of slop allowed during photometry. The measured intensity
    # can be this factor more or less than the predicted intensity.
    ('photometry_slop', 'NAC'): 2.0,
    ('photometry_slop', 'WAC'): 2.0,

    # The range of vmags to use when determining the dimmest star visible.
    'min_vmag': 5.,
    'max_vmag': 15.,
    'vmag_increment': 0.5,

    # The size of the box to analyze vs. the predicted integrated DN.
    # This is also used as the threshold where we let the Gaussian PSF sigma
    # float and the refine fit look further away.
    'photometry_boxsizes': ((100000,15),
                            ( 50000,11),
                            (   500, 9),
                            (     0, 7)),

    # The maximum DN a star can have for us to trust the photometry.
    'max_star_dn': 100000.,

    # The minimum DN the brighest star can have for us to try to match
    # a single star even if there are other objects in the FOV.
    'min_dn_force_one_star': 25000.,

    # How far (in pixels) a star has to be from a major body before it is no
    # longer considered to conflict.
    'star_body_conflict_margin': 3,

    # If star navigation fails, get rid of the brightest star if it's at least
    # this bright. None means don't apply this test.
    'too_bright_dn': 1000,

    # If star navigation fails, get rid of the brightest star if it's at least
    # this much brighter (factor) than the next dimmest star. None means don't
    # apply this test.
    'too_bright_factor': None,

    # The font and font size for star labels (font filename, size)
    'font': {
         256: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
         512: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
        1000: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18),
        1024: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18)
    }
}

BODIES_CONFIG = {
    # The minimum number of pixels in the bounding box surrounding the body
    # in order to bother with it.
    'min_bounding_box_area': 3*3,

    # For a body that lives inside the rings, this is the minimum emission angle
    # allowed to navigate on that body. Otherwise the body will be hidden by
    # the rings.
    'min_emission_ring_body': 20,

    # Do enough oversampling of a small body to fill a box approximately this
    # size. This allows nice anti-aliasing of edges.
    'oversample_edge_limit': 512,

    # But...don't do more than this amount of oversampling, because there's
    # really no point in being this precise.
    'oversample_maximum': 8,

    # The fraction of the width/height of a body that must be visible on either
    # side of the center in order for the curvature to be sufficient for
    # correlation.
    # Set both 'curvature_threshold_frac' and 'curvature_threshold_pixels' to
    # eliminate the check for curvature and mark all bodies as OK.
    'curvature_threshold_frac': 0.02,

    # The number of pixels of the width/height of a body that must be visible
    # on either side of the center in order for the curvature to be sufficient
    # for correlation. Both curvature_threshold_frac and
    # curvature_threshold_pixels must be true for correlation to be trusted.
    # The _pixels version is useful for the case of small moons.
    'curvature_threshold_pixels': 20,

    # The maximum incidence that can be considered a limb instead of a
    # terminator.
    # Set to oops.TWOPI to eliminate the check for limbs and mark
    # all bodies as OK.
    'limb_incidence_threshold': np.radians(88.), # cos = 0.05

    # What fraction of the total visible limb needs to meet the above criterion
    # in order for the limb to be marked as OK. This is only used in the case
    # where curvature is bad.
    'limb_incidence_frac': 0.4,

    # What resolution is so small that the surface features make the moon
    # non-circular when viewing the limb? If the center resolution is
    # higher than this, we need to blur the body before correlating.
    'surface_bumpiness':
        {'SATURN':      50.00, # This is really a measure of atmospheric haze
         'PAN':          7.00, # Synchronous; this is irregularity in surface
         'DAPHNIS':      3.00, # Synchronous; this is irregularity in surface
         'ATLAS':       12.00, # Synchronous; this is irregularity in surface
         'PROMETHEUS':  25.00, # Synchronous; this is irregularity in surface
         'PANDORA':     20.00, # Synchronous; this is irregularity in surface
         'EPIMETHEUS':  13.00, # Synchronous; this is irregularity in surface
         'JANUS':       20.00, # Synchronous; this is irregularity in surface
         'MIMAS':        5.00, # Relief of Herschel crater
         'ENCELADUS':    0.50, # Lacks high-relief
         'TETHYS':       5.00, # Relief of Odysseus crater
         'TELESTO':      5.00, # Synchronous; this is irregularity in surface
         'CALYPSO':      7.00, # Synchronous; this is irregularity in surface
         'DIONE':        0.75, # Lacks high-relief craters
         'HELENE':      45.00, # Very irregular and tumbles
         'RHEA':         0.75, # Lacks high-relief craters
         'TITAN':        0.00, # We never do Titan with a Lambert model
         'HYPERION':   350.00, # Highly elongated and tumbles
         'IAPETUS':     35.00, # This is the height of the 'walnut' equatorial bulge
         'PHOEBE':      40.00, # A real mess with lots of deep craters
         },

    'geometric_albedo':
        {'SATURN':     0.342, # https://doi.org/10.1016%2F0019-1035%2883%2990147-1
         'PAN':        0.500, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'DAPHNIS':    0.500, # Wikipedia
         'ATLAS':      0.800, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'PROMETHEUS': 0.500, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'PANDORA':    0.700, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'EPIMETHEUS': 0.800, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'JANUS':      0.900, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'MIMAS':      0.600, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'ENCELADUS':  1.000, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'TETHYS':     0.800, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'TELESTO':    1.000, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'CALYPSO':    1.000, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'DIONE':      0.700, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'HELENE':     0.700, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'RHEA':       0.700, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'TITAN':      0.220, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'HYPERION':   0.300, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'IAPETUS':    0.275, # Mean of light+dark sides https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         'PHOEBE':     0.080, # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
         },

    # Whether or not Lambert shading should be used, as opposed to just a
    # solid unshaded shape, when a cartographic reprojection is not
    # available.
    'use_lambert': True,

    # Whether or not the albedo should be use to scale the model's brightness
    'use_albedo': False,

    # The minimum number of pixels in the bounding box surrounding the body
    # in order to compute the reprojection for bootstrapping if this is going
    # to be a seed image (and thus we need pretty good resolution in the data).
    'min_reproj_seed_area': 200*200,

    # The minimum number of pixels in the bounding box surrounding the body
    # in order to compute the reprojection for bootstrapping if this is going
    # to be a candidate image (and thus we don't need very good resolution
    # but it has to be big enough to be worth the effort).
    'min_reproj_candidate_area': 50*50,

    # The resolution in longitude and latitude (radians) for the bootstrap
    # reprojection.
    'reproj_lon_resolution': np.radians(1.),
    'reproj_lat_resolution': np.radians(1.),

    # The latlon coordinate type and direction for the metadata reprojection
    # and sub-solar and sub-observer longitudes.
    'reproj_latlon_type': 'centric',
    'reproj_lon_direction': 'east',

    # A body has to take up at least this many pixels in order to be labeled.
    'min_text_area': 0.003,

    # The font and font size for body labels (font filename, size)
    'font': {
         256: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
         512: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
        1000: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18),
        1024: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18)
    }
}

TITAN_DEFAULT_CONFIG = {
    # The altitude of the top of the atmosphere (km).
    'atmosphere_height': 700,
}

RINGS_DEFAULT_CONFIG = {
    # The source for profile data - 'voyager', 'uvis', or 'ephemeris'.
    'model_source': 'ephemeris',

    # There must be at least this many fiducial features for rings to be used
    # for correlation.
    'fiducial_feature_threshold': 3,

    # The RMS error of a feature must be this many times less than the
    # coarsest resolution of the feature in the image in order for the feature
    # to be used. This makes sure that the statistical scatter of the feature
    # is blurred out during correlation.
    'fiducial_rms_gain': 2,

    # A full gap or ringlet must be at least this many pixels wide at some
    # place in the image to use it.
    'fiducial_min_feature_width': 2,

    # Assume a one-sided feature is about this wide in km. This is used to
    # determine if the local resolution is high enough for the feature to be
    # visible.
    'one_sided_feature_width': 30.,

    # When manufacturing a model from an ephemeris list, each one-sided feature
    # is shaded approximately this many pixels wide.
    'fiducial_ephemeris_width': 100,

    # There must be at least this much curvature present for rings to be used
    # for correlation.
    'min_curvature_low_confidence': (0.0, 0.5),
    'min_curvature_high_confidence': (0.17, 1.0), # ~10 degrees

    # If there is at least this much curvature, then we can reduce the number
    # of required fiducial features accordingly.
    'curvature_to_reduce_features': np.radians(90.), # 90 degrees visible
    'curvature_reduced_features': 1,

    # The minimum ring emission angle in the image must be at least this
    # many degrees away from 90 for fiducial features to be used; otherwise
    # we just use a plain model.
    'emission_fiducial_threshold': 0.75,

    # The minimum ring emission angle in the image must be at least this
    # many degrees away from 90 to be used at all.
    'emission_use_threshold': 0.2,

    # Remove the shadow of Saturn from the model
    'remove_saturn_shadow': True,

    # Remove the shadow of other bodies from the model
    'remove_body_shadows': False,

    # The font and font size for ring labels (font filename, size)
    'font': {
         256: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
         512: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
        1000: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18),
        1024: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18)
    },

    # The font and font size for ring labels (font filename, size)
    'parameter_font': {
         256: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
         512: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
        1000: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 12),
        1024: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 12)
    },

    # The starting position and step size for ring parameter labels
    'parameter_overlay_start': {
         512: (10, 30),
        1000: (10, 30),
        1024: (10, 30)
    },
    'parameter_overlay_step': {
         512: (100, 100),
        1000: (150, 100),
        1024: (150, 100)
    },
    'parameter_overlay_line_len_thickness': {
         512: (10, 1),
        1000: (20, 1),
        1024: (20, 1)
    },
    'parameter_overlay_box_size': {
         512: (1,0),
        1000: (2,1),
        1024: (2,1)
    },
}

OFFSET_DEFAULT_CONFIG = {
    # A body has to be at least this many pixels in area for us to pay
    # attention to it for bootstrapping purposes.
    'bootstrap_min_body_area': 100,

    # By default, each image and model is Gaussian blurred by this much
    # before correlation. This can be overridden if the rings model requests
    # additional blurring.
    'default_gaussian_blur': 0.25,

    # This is the maximum we allow a model to be blurred because of rings
    # or bodies before we give up.
    'maximum_blur': 100.,

    # This is the maximum blur we allow while still trying to compute the
    # error bars on the model result.
    'maximum_blur_error_bars': 60.,

    # By default, the median filter looks at this many pixels.
    'median_filter_size': None, # 11,

    # By default, the median filter is Gaussian blurred by this much before
    # being subtracted from the image or model. If the median filter is turned
    # off, we still blur by this much and subtract it from the image or model.
    'median_filter_blur': 1.2,

    # Do we want to use a circular or square footprint for the median filter?
    'median_filter_footprint': 'circle',

    #vvv
    # If there are at least this many bodies in the image, then we trust the
    # body-based model correlation result.
    'num_bodies_threshold': 3,

    # OR

    # If the bodies cover at least this fraction of the image, then we trust
    # the body-based model correlation result.
    'bodies_cov_threshold': 0.0005,
    #^^^

    #vvv
    # If the total model covers at least this number of pixels the given
    # distance from an edge, then we trust it.
    'model_cov_threshold': 25,
    'model_edge_pixels': 5,
    #^^^

    # The number of pixels to search in U,V during secondary correlation.
    'secondary_corr_search_size': (15,15),

    # The lowest confidence to allow for models
    'lowest_confidence': 0.01,

    # If the stars-based and bodies/rings-based correlations differ by at
    # least this number of pixels, then we need to choose between the stars
    # and the bodies.
    'stars_model_diff_threshold': 2,

    # If there are at least this many good stars, then the stars can override
    # the bodies/rings model when they differ by the above number of pixels.
    # Otherwise, the bodies/rings model overrides the stars.
    'stars_override_threshold': 6,

    # The maximum number of times to iterate on secondary correlation while
    # trying to zero in on the maximum answer.
    'secondary_max_attempts': 10,

    # If secondary correlation is off by at least this number of pixels, it
    # fails (or continues to iterate).
    'secondary_corr_threshold': 1,

    # If the secondary correlation peak isn't at least this fraction of the
    # primary correlation peak, correlation fails.
    'secondary_corr_peak_threshold': 0.2,

    # If the best secondary correlation peak found during offset tweaking isn't
    # at least this fraction of the previous worst secondary correlation peak,
    # correlation fails.
    'secondary_corr_tweak_threshold': 0.7,

    # The maximum number of exhaustive search correlations that can be done
    # if secondary correlation doesn't converge after max_attempts attempts.
    'secondary_max_num_exhaustive': 3*3,

    # The font and font size for the main descriptive block
    # (font filename, size)
    'font': {
         256: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
         512: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 10),
        1000: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18),
        1024: ('/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', 18)
    }
}

BOOTSTRAP_DEFAULT_CONFIG = {
    # These bodies can be used for bootstrapping.
    'body_list': ['DIONE', 'ENCELADUS', 'IAPETUS', 'MIMAS', 'RHEA', 'TETHYS'],

    # The minimum offset confidence for an image to be considered good.
    'min_confidence': 0.1,

    # The minimum square size of a moon to be used to be considered good.
    'min_area': 128*128,

    # The maximum lighting angles that can be used to be considered good.
    'max_phase_angle': np.radians(135.),
    'max_incidence_angle': np.radians(70.),
    'max_emission_angle': np.radians(70.),

    # The resolution in longitude and latitude (radians) for reprojections and
    # mosaics.
    'lon_resolution': np.radians(0.5),
    'lat_resolution': np.radians(0.5),

    # The latlon coordinate type and direction for reprojections and mosaics.
    'latlon_type': 'centric',
    'lon_direction': 'east',

    # The minimum fraction of a moon's pixels in an image that are available
    # from cartographic data in order for a bootstrapped offset to be attempted.
    'min_coverage_frac': 0.1,

    # The maximum difference in sub-solar illumation angle permitted.
    'max_subsolar_dist': np.radians(45.),

    # The maximum difference in resolution allowable between the mosaic
    # and the image to be bootstrapped.
    'max_res_factor': 3,
}
