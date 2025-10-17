from typing import Any, Optional, Sequence, cast

import matplotlib.pyplot as plt
from oops import Observation
import numpy as np

from nav.annotation import Annotations
from nav.config import Config
from nav.nav_model import (NavModel,
                           NavModelBody,
                           NavModelRings,
                           NavModelStars,
                           NavModelTitan)
from nav.nav_technique import (NavTechniqueCorrelateAll,
                               NavTechniqueStars)
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayFloatType, NDArrayUint32Type, NDArrayUint8Type


class NavMaster(NavBase):
    """Coordinates the overall navigation process using multiple models and techniques.

    This class manages the creation of different navigation models (stars, bodies, rings,
    Titan), combines them appropriately, and applies navigation techniques to determine
    the final offset between predicted and actual positions.
    """

    def __init__(self,
                 obs: Observation,
                 *,
                 nav_models: Optional[list[str]] = None,
                 nav_techniques: Optional[list[str]] = None,
                 config: Optional[Config] = None) -> None:
        """Initializes a navigation master object for an observation.

        Parameters:
            obs: The observation object containing image and metadata.
            nav_models: Optional list of navigation models to use. If None, uses all models.
                Each entry must be one of 'stars', 'bodies', 'rings', 'titan'.
            nav_techniques: Optional list of navigation techniques to use. If None, uses all
                techniques. Each entry must be one of 'stars', 'correlate_all'.
            config: Optional configuration object. If None, uses the default configuration.
        """

        super().__init__(config=config)

        nav_techniques = nav_techniques or ['stars']  # XXX
        self._obs = obs
        if nav_models is not None and not all(
                x in ['stars', 'bodies', 'rings', 'titan'] for x in nav_models):
            raise ValueError(f'Invalid nav_models: {nav_models}')
        self._nav_models_to_use = nav_models
        if nav_techniques is not None and not all(
                x in ['stars', 'correlate_all'] for x in nav_techniques):
            raise ValueError(f'Invalid nav_techniques: {nav_techniques}')
        self._nav_techniques_to_use = nav_techniques
        self._final_offset: tuple[float, float] | None = None
        self._final_confidence: float | None = None
        self._offsets: dict[str, Any] = {}  # TODO Type
        self._star_models: list[NavModelStars] | None = None
        self._body_models: list[NavModelBody] | None = None
        self._ring_models: list[NavModelRings] | None = None
        self._titan_models: list[NavModelTitan] | None = None

        self._closest_model_index: NDArrayUint32Type | None = None

        self._metadata: dict[str, Any] = {}
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initializes the metadata dictionary."""
        obs_metadata = self.obs.inst.get_public_metadata()
        # kernels
        # RA/DEC corners and center, un nav and nav

        self._metadata = {
            'observation': obs_metadata,
        }

        try:
            spice_kernels = self.obs.spice_kernels
        except Exception:
            self._metadata['spice_kernels'] = 'Not supported by instrument'  # TODO
            pass
        else:
            self._metadata['spice_kernels'] = spice_kernels

        self._metadata['models'] = {}

    @property
    def obs(self) -> Observation:
        """Returns the observation object associated with this navigation master."""
        return self._obs

    @property
    def final_offset(self) -> tuple[float, float] | None:
        """Returns the final computed offset between predicted and actual positions."""
        return self._final_offset

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary."""
        return self._metadata

    @property
    def star_models(self) -> list[NavModelStars]:
        """Returns the list of star navigation models, computing them if necessary."""
        self.compute_star_models()
        assert self._star_models is not None
        return self._star_models

    @property
    def body_models(self) -> list[NavModelBody]:
        """Returns the list of planetary body navigation models, computing them if necessary."""
        self.compute_body_models()
        assert self._body_models is not None
        return self._body_models

    @property
    def ring_models(self) -> list[NavModelRings]:
        """Returns the list of ring navigation models, computing them if necessary."""
        self.compute_ring_models()
        assert self._ring_models is not None
        return self._ring_models

    @property
    def titan_models(self) -> list[NavModelTitan]:
        """Returns the list of Titan-specific navigation models, computing them if necessary."""
        self.compute_titan_models()
        assert self._titan_models is not None
        return self._titan_models

    @property
    def all_models(self) -> Sequence[NavModel]:
        """Returns a sequence containing all navigation models."""
        return (cast(list[NavModel], self.star_models) +
                cast(list[NavModel], self.body_models) +
                cast(list[NavModel], self.ring_models) +
                cast(list[NavModel], self.titan_models))

    def compute_star_models(self) -> None:
        """Creates navigation models for stars in the observation.

        If star models have already been computed, does nothing.
        """

        if self._nav_models_to_use is not None and 'stars' not in self._nav_models_to_use:
            self._star_models = []
            return

        if self._star_models is not None:
            # Keep cached version
            return

        stars_model = NavModelStars(self._obs)
        stars_model.create_model()
        self._star_models = [stars_model]
        self._metadata['models']['star_model'] = stars_model.metadata

        # plt.imshow(stars_model.model_img)
        # plt.show()

    def compute_body_models(self) -> None:
        """Creates navigation models for planetary bodies in the observation.

        Identifies visible bodies within the field of view, sorts them by distance,
        and creates a navigation model for each one. If body models have already been
        computed, does nothing.
        """

        if self._nav_models_to_use is not None and 'bodies' not in self._nav_models_to_use:
            self._body_models = []
            return

        if self._body_models is not None:
            # Keep cached version
            return

        obs = self._obs
        config = self._config
        logger = self._logger

        if obs.closest_planet is not None:
            body_list = [obs.closest_planet] + config.satellites(obs.closest_planet)
        else:
            body_list = []

        large_body_dict = self._obs.inventory(body_list, return_type='full')
        # Make a list sorted by range, with the closest body first, limiting to bodies
        # that are actually in the FOV
        def _body_in_fov(obs: Observation,
                         inv: dict[str, Any]) -> bool:
            """Determines if a body is within the extended field of view.

            Parameters:
                obs: The observation object.
                inv: The inventory dictionary for the body.

            Returns:
                True if the body is at least partially within the extended field of view.
            """
            return cast(bool,
                        (inv['u_max_unclipped'] >= obs.extfov_u_min and
                         inv['u_min_unclipped'] <= obs.extfov_u_max and
                         inv['v_max_unclipped'] >= obs.extfov_v_min and
                         inv['v_min_unclipped'] <= obs.extfov_v_max))

        large_bodies_by_range = [(x, large_body_dict[x])
                                 for x in large_body_dict
                                 if _body_in_fov(obs, large_body_dict[x])]
        large_bodies_by_range.sort(key=lambda x: x[1]['range'])

        logger.info('Closest planet: %s', obs.closest_planet)
        logger.info('Large body inventory by increasing range: %s',
                    ', '.join([x[0] for x in large_bodies_by_range]))

        self._body_models = []
        self._metadata['models']['body_models'] = {}

        for body, inventory in large_bodies_by_range:
            body_model = NavModelBody(obs, body,
                                      inventory=inventory,
                                      config=config)
            body_model.create_model()
            self._body_models.append(body_model)
            self._metadata['models']['body_models'][body] = body_model.metadata

    def compute_ring_models(self) -> None:
        """Creates navigation models for planetary rings in the observation.

        If ring models have already been computed, does nothing.
        """

        if self._nav_models_to_use is not None and 'rings' not in self._nav_models_to_use:
            self._ring_models = []
            return

        if self._ring_models is not None:
            # Keep cached version
            return

        self._ring_models = []
        self._metadata['models']['ring_model'] = {}
        # TODO Ring models

    def compute_titan_models(self) -> None:
        """Creates Titan-specific navigation models for the observation.

        If Titan models have already been computed, does nothing.
        """

        if self._nav_models_to_use is not None and 'titan' not in self._nav_models_to_use:
            self._titan_models = []
            return

        if self._titan_models is not None:
            # Keep cached version
            return

        self._titan_models = []
        self._metadata['models']['titan_model'] = {}
        # TODO Titan models

    def compute_all_models(self) -> None:
        """Creates all navigation models for the observation.

        This includes star models, ring models, body models, and Titan-specific models.
        """

        self.compute_star_models()
        self.compute_ring_models()
        self.compute_body_models()
        self.compute_titan_models()

    def navigate(self) -> None:
        """Performs navigation by applying different navigation techniques.

        Computes all navigation models, then applies star-based and all-model-based
        navigation techniques. Determines the final offset based on the results.
        """

        prevailing_confidence = 0.
        self._final_offset = None

        self.compute_all_models()

        self._metadata['navigation_techniques'] = {}

        # TODO If both nav_models and nav_techniques are limited to stars, then we
        # essentially navigate using stars twice.

        if self._nav_techniques_to_use is None or 'stars' in self._nav_techniques_to_use:
            nav_stars = NavTechniqueStars(self)
            nav_stars.navigate()
            self._metadata['navigation_techniques']['stars'] = nav_stars.metadata
            self._offsets['stars'] = nav_stars.offset
            if nav_stars.offset is not None:
                self._final_offset = nav_stars.offset
                if nav_stars.confidence is None:
                    raise ValueError('Star navigation technique confidence is None')
                prevailing_confidence = nav_stars.confidence
        else:
            self._offsets['stars'] = None

        if self._nav_techniques_to_use is None or 'correlate_all' in self._nav_techniques_to_use:
            nav_all = NavTechniqueCorrelateAll(self)
            nav_all.navigate()
            self._metadata['navigation_techniques']['correlate_all'] = nav_all.metadata
            self._offsets['correlate_all'] = nav_all.offset
            if nav_all.offset is not None:
                if nav_all.confidence is None:
                    raise ValueError('Correlate all navigation technique confidence is None')
                if nav_all.confidence > prevailing_confidence:
                    prevailing_confidence = nav_all.confidence
                    self._final_offset = nav_all.offset
        else:
            self._offsets['correlate_all'] = None

        self._final_confidence = prevailing_confidence

    def create_overlay(self) -> NDArrayUint8Type:
        """Creates a visual overlay combining the image and navigation annotations.

        Combines all model annotations, applies the computed offset, and generates
        an annotated image with contrast adjustment.
        """

        annotations = Annotations()

        for model in self.all_models:
            annotations.add_annotations(model.annotations)

        offset = (0., 0.)
        if self._final_offset is not None:
            offset = self._final_offset

        overlay = annotations.combine(offset=offset,
                                      #   text_use_avoid_mask=False,
                                      #   text_show_all_positions=True,
                                      #   text_avoid_other_text=False
                                      )
        img = self._obs.data.astype(np.float64)

        res = np.zeros(img.shape + (3,), dtype=np.uint8)
        bw_res = np.zeros(img.shape, dtype=np.uint8)

        # Create a min_index that is the size of the original image, properly offset.
        # This array indicates which model is at the front for each pixel.
        # TODO This is inefficient because we are creating the combined model a second
        # time. But we may need to do this in the future if NavTechniqueCorrelateAll
        # ends up being more complicated. This needs to be revisited.
        nav_all = NavTechniqueCorrelateAll(self)
        min_index = self.obs.extract_offset_image(nav_all.closest_model_index, offset)

        def _stretch_region(sub_img: NDArrayFloatType) -> NDArrayUint8Type:
            """Stretches a region of the image."""
            blackpoint = float(np.quantile(sub_img, 0.001))
            whitepoint = float(np.quantile(sub_img, 0.999))
            if blackpoint == whitepoint:
                whitepoint = blackpoint + .01
            gamma = 1  # 0.5

            img_stretched = np.floor((np.maximum(sub_img-blackpoint, 0) /
                                     (whitepoint-blackpoint))**gamma*256)
            img_stretched = np.clip(img_stretched, 0, 255)
            img_stretched = cast(NDArrayUint8Type, img_stretched.astype(np.uint8))
            return img_stretched

        already_stretched_mask = np.zeros(img.shape, dtype=bool)
        for model_index, model in enumerate(self.all_models):
            if model.stretch_regions is not None:
                for stretch_region_packed in model.stretch_regions:
                    stretch_region = np.unpackbits(stretch_region_packed, axis=0).astype(bool)
                    stretch_region = self.obs.unpad_array_to_extfov(stretch_region)
                    stretch_region = self._obs.extract_offset_image(stretch_region, offset)
                    if not np.any(stretch_region):
                        continue
                    # We stretch this region by itself if we haven't already stretched
                    # any part of it and all of it is the closest model in the view for those
                    # pixels.
                    if (not np.any(already_stretched_mask[stretch_region]) and
                        np.all(min_index[stretch_region] == model_index)):
                        bw_res[stretch_region] = _stretch_region(img[stretch_region])
                        already_stretched_mask |= stretch_region

        # Now stretch the rest of the image
        bw_res[~already_stretched_mask] = _stretch_region(img[~already_stretched_mask])

        res[:, :, 0] = bw_res
        res[:, :, 1] = bw_res
        res[:, :, 2] = bw_res

        if overlay is not None:
            overlay[overlay < 128] = 0  # TODO Hard-coded constant
            mask = np.any(overlay, axis=2)
            res[mask, :] = overlay[mask, :]

        # im = Image.fromarray(res)
        # fn = Path(obs.basename).stem
        # im.save(f'/home/rfrench/{fn}.png')

        plt.imshow(res)
        plt.show()

        # model_mask = body_model.model_mask
        # plt.imshow(model_mask)

        return res
