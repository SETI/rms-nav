import fnmatch
from typing import Any, Optional, Sequence, cast

import matplotlib.pyplot as plt  # noqa: F401
from oops import Observation
import numpy as np

from nav.annotation import Annotations
from nav.config import Config
from nav.support.file import clean_obj
from nav.nav_model import (NavModel,
                           NavModelBody,
                           NavModelBodySimulated,
                           NavModelCombined,
                           NavModelRings,
                           NavModelStars,
                           NavModelTitan)
from nav.nav_model.nav_model_body_base import NavModelBodyBase
from nav.nav_technique import NavTechniqueCorrelateAll
from nav.nav_technique import NavTechniqueManual
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayFloatType, NDArrayUint8Type


class NavMaster(NavBase):
    """Coordinates the overall navigation process using multiple models and techniques.

    This class manages the creation of different navigation models (e.g. stars, bodies,
    rings, and Titan), combines them appropriately, and applies navigation techniques to
    determine the final offset between predicted and actual positions.

    The floating point offset is of form (dv, du). If an object is predicted by the SPICE
    kernels to be at location (v, u) in the image, then the actual location of the
    object is (v+dv, u+du). This means a positive offset is equivalent to shifting a model
    up and to the right (positive v and u).
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
                Each entry must be one of 'stars', 'body:<body_name>', 'rings', 'titan'.
                Glob-style wildcards are supported.
            nav_techniques: Optional list of navigation techniques to use. If None, uses all
                techniques. Each entry must be one of 'stars', 'correlate_all'.
            config: Optional configuration object. If None, uses the default configuration.
        """

        super().__init__(config=config)

        if nav_models is None:
            nav_models = ['*']
        else:
            nav_models = [x.strip().lower() for x in nav_models]
        if nav_techniques is None:
            # Default: only run correlate_all unless user explicitly adds others
            nav_techniques = ['correlate_all']
        else:
            nav_techniques = [x.strip().lower() for x in nav_techniques]

        self._obs = obs
        self._nav_models_to_use = nav_models
        self._nav_techniques_to_use = nav_techniques
        self._final_offset: tuple[float, float] | None = None
        self._final_confidence: float | None = None
        self._offsets: dict[str, Any] = {}  # TODO Type
        self._star_models: list[NavModelStars] | None = None
        self._body_models: list[NavModelBodyBase] | None = None
        self._ring_models: list[NavModelRings] | None = None
        self._titan_models: list[NavModelTitan] | None = None

        self._combined_model: NavModelCombined | None = None

        self._metadata: dict[str, Any] = {}
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initializes the metadata dictionary."""
        obs_metadata = self.obs.get_public_metadata()
        # kernels
        # RA/DEC corners and center, un nav and nav

        self._metadata = {
            'status': 'ok',
            'observation': obs_metadata,
        }

        try:
            spice_kernels = self.obs.spice_kernels
        except Exception:
            self._metadata['spice_kernels'] = 'Not supported by instrument'  # TODO
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

    def metadata_serializable(self) -> dict[str, Any]:
        """Returns a copy of the metadata dictionary that is JSON serializable."""
        return cast(dict[str, Any], clean_obj(self._metadata))

    @property
    def star_models(self) -> list[NavModelStars]:
        """Returns the list of star navigation models, computing them if necessary."""
        self.compute_star_models()
        assert self._star_models is not None
        return self._star_models

    @property
    def body_models(self) -> list[NavModelBodyBase]:
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

    @property
    def combined_model(self) -> NavModelCombined | None:
        """Returns the final combined model."""
        return self._combined_model

    def compute_star_models(self) -> None:
        """Creates navigation models for stars in the observation.

        If star models have already been computed, does nothing.
        """

        if (self._nav_models_to_use is not None and
            not any(fnmatch.fnmatch('stars', x) for x in self._nav_models_to_use)):
            self._star_models = []
            return

        if self._star_models is not None:
            # Keep cached version
            return

        # If obs has a simulated star list, pass it to NavModelStars
        star_list = getattr(self._obs, 'sim_star_list', None)
        stars_model = NavModelStars('stars', self._obs, star_list=star_list)
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

        if obs.is_simulated:
            large_body_dict = obs.sim_inventory
        else:
            large_body_dict = obs.inventory(body_list, return_type='full')

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

        for body_name, inventory in large_bodies_by_range:
            model_name = f'body:{body_name.upper()}'
            if not any(fnmatch.fnmatch(model_name.lower(), x.lower())
                       for x in self._nav_models_to_use):
                continue
            if obs.is_simulated:
                sim_params = obs.sim_body_models[body_name]
                body_model: NavModelBodyBase = NavModelBodySimulated(model_name, obs,
                                                                     body_name, sim_params,
                                                                     config=config)
            else:
                body_model = NavModelBody(model_name, obs, body_name,
                                          inventory=inventory,
                                          config=config)
            body_model.create_model()
            self._body_models.append(body_model)
            self._metadata['models']['body_models'][body_name] = body_model.metadata

    def compute_ring_models(self) -> None:
        """Creates navigation models for planetary rings in the observation.

        If ring models have already been computed, does nothing.
        """

        if (self._nav_models_to_use is not None and
            not any(fnmatch.fnmatch('rings', x.lower()) for x in self._nav_models_to_use)):
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

        if (self._nav_models_to_use is not None and
            not any(fnmatch.fnmatch('titan', x.lower()) for x in self._nav_models_to_use)):
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

        if any(fnmatch.fnmatch('correlate_all', x.lower())
               for x in self._nav_techniques_to_use):
            nav_all = NavTechniqueCorrelateAll(self)
            nav_all.navigate()
            correlate_all_combined_model = nav_all.combined_model()
            self._combined_model = correlate_all_combined_model
            if correlate_all_combined_model is None:
                self.logger.info('Correlate all navigation technique failed')
            else:
                # plt.imshow(correlate_all_combined_model.model_img)
                # plt.show()
                # plt.imshow(correlate_all_combined_model.mask)
                # plt.show()
                # plt.imshow(correlate_all_combined_model.weighted_mask)
                # plt.show()
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

        # Manual technique
        if any(fnmatch.fnmatch('manual', x.lower())
               for x in self._nav_techniques_to_use):
            nav_manual = NavTechniqueManual(self)
            nav_manual.navigate()
            manual_combined_model = nav_manual.combined_model()
            # Keep a combined model available for overlays if needed
            if manual_combined_model is not None:
                self._combined_model = manual_combined_model
            self._metadata['navigation_techniques']['manual'] = nav_manual.metadata
            self._offsets['manual'] = nav_manual.offset
            if nav_manual.offset is not None:
                # Manual result should take precedence as an explicit override
                # TODO Eventually pay attention to confidence
                self._final_offset = nav_manual.offset
                self._final_confidence = nav_manual.confidence
                prevailing_confidence = nav_manual.confidence or prevailing_confidence
        else:
            self._offsets['manual'] = None

        self._final_confidence = prevailing_confidence

        if self._final_offset is None:
            self.logger.info('Final offset: NONE')
        else:
            self.logger.info(f'Final offset: dU {self._final_offset[1]:.3f}, '
                             f'dV {self._final_offset[0]:.3f}')
            self.logger.info(f'Final confidence: {self._final_confidence:.5f}')

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
        if self.combined_model is None:
            # If we never created a combined model earlier, we need one now for the overlay
            self._combined_model = NavModelCombined('combined',
                                                    self.obs,
                                                    list(self.all_models))
        if self.combined_model is None:
            raise ValueError('Combined model is None')
        if self.combined_model.closest_model_index is None:
            raise ValueError('Combined model closest model index is None')
        min_index = self.obs.extract_offset_array(self.combined_model.closest_model_index,
                                                  offset)

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
                    stretch_region = self._obs.extract_offset_array(stretch_region, offset)
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

        # plt.imshow(res)
        # plt.show()

        # model_mask = body_model.model_mask
        # plt.imshow(model_mask)

        return res
