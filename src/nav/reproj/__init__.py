"""nav.reproj -- reprojection and mosaicing utilities for bodies and rings.

This package provides tools for reprojecting planetary body images onto
latitude/longitude grids (BodyMosaic) and ring images onto radius/longitude
grids (RingMosaic), along with supporting photometric models, merge strategies,
and ring orbit models. A standalone cartographic model utility projects a
body mosaic back onto image coordinates for navigation correlation.

Public API:

    BodyMosaic                   -- accumulates body reprojections onto lat/lon grids
    BodyMosaicMergeStrategy      -- enum controlling how BodyMosaic conflicts are resolved
    BodyReprojResult             -- data returned by BodyMosaic.reproject()
    BodyMosaicData               -- mosaic data returned by BodyMosaic retrieval methods
    CartographicModelResult      -- result from create_cartographic_model()
    create_cartographic_model    -- project a body mosaic onto image coordinates
    RingMosaic                   -- accumulates ring reprojections onto sparse rad/lon grids
    RingReprojResult             -- data returned by RingMosaic.reproject()
    RingMosaicData               -- mosaic data returned by RingMosaic retrieval methods
    RingMosaicMergeStrategy      -- enum controlling how RingMosaic conflicts are resolved
    PhotometricModel             -- protocol for photometric correction implementations
    LambertModel                 -- Lambertian photometric correction
    LommelSeeligerModel          -- Lommel-Seeliger photometric correction
    MinnaertModel                -- Minnaert photometric correction
    RingOrbitModel               -- Keplerian ring orbit with precession
    FRING_CORE                   -- pre-defined F ring core orbit model
    BRING_OUTER_EDGE             -- pre-defined B ring outer edge orbit model
"""

from nav.reproj.bodies import BodyMosaic, BodyMosaicData, BodyMosaicMergeStrategy, BodyReprojResult
from nav.reproj.cartographic_model import CartographicModelResult, create_cartographic_model
from nav.reproj.photometric_model import (
    LambertModel,
    LommelSeeligerModel,
    MinnaertModel,
    PhotometricModel,
    photometric_model_from_name,
)
from nav.reproj.ring_orbit_model import (
    BRING_OUTER_EDGE,
    FRING_CORE,
    RingOrbitModel,
)
from nav.reproj.rings import RingMosaic, RingMosaicData, RingMosaicMergeStrategy, RingReprojResult

__all__ = [
    'BRING_OUTER_EDGE',
    'FRING_CORE',
    'BodyMosaic',
    'BodyMosaicData',
    'BodyMosaicMergeStrategy',
    'BodyReprojResult',
    'CartographicModelResult',
    'LambertModel',
    'LommelSeeligerModel',
    'MinnaertModel',
    'PhotometricModel',
    'photometric_model_from_name',
    'RingMosaic',
    'RingMosaicData',
    'RingMosaicMergeStrategy',
    'RingOrbitModel',
    'RingReprojResult',
    'create_cartographic_model',
]
