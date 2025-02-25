from pathlib import Path
from typing import Any, Optional, Sequence, cast

import matplotlib.pyplot as plt
from oops import Observation
import numpy as np
from PIL import Image, ImageDraw

from nav.annotation import Annotations
from nav.config import Config
from nav.nav_model import (NavModel,
                           NavModelBody,
                           NavModelRings,
                           NavModelStars,
                           NavModelTitan)
from nav.nav_technique import (NavTechniqueAllModels,
                               NavTechniqueStars,
                               NavTechniqueTitan)
from nav.support.nav_base import NavBase
from nav.support.file import dump_yaml
from nav.support.types import NDArrayFloatType


class NavMaster(NavBase):
    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        super().__init__(config=config, logger_name=logger_name)

        self._obs = obs
        self._final_offset: tuple[float, float] | None = None
        self._offsets: dict[str, Any] = {}  # TODO Type
        self._star_models: list[NavModelStars] | None = None
        self._body_models: list[NavModelBody] | None = None
        self._ring_models: list[NavModelRings] | None = None
        self._titan_models: list[NavModelTitan] | None = None

        self._combined_model: NDArrayFloatType | None = None

    @property
    def obs(self) -> Observation:
        return self._obs

    @property
    def final_offset(self) -> tuple[float, float] | None:
        return self._final_offset

    @property
    def star_models(self) -> list[NavModelStars]:
        self.compute_star_models()
        assert self._star_models is not None
        return self._star_models

    @property
    def body_models(self) -> list[NavModelBody]:
        self.compute_body_models()
        assert self._body_models is not None
        return self._body_models

    @property
    def ring_models(self) -> list[NavModelRings]:
        self.compute_ring_models()
        assert self._ring_models is not None
        return self._ring_models

    @property
    def titan_models(self) -> list[NavModelTitan]:
        self.compute_titan_models()
        assert self._titan_models is not None
        return self._titan_models

    @property
    def all_models(self) -> Sequence[NavModel]:
        return (self.star_models + self.body_models +
                self.ring_models + self.titan_models)

    @property
    def combined_model(self) -> NDArrayFloatType | None:
        self._create_combined_model()
        return self._combined_model

    def compute_star_models(self) -> None:

        if self._star_models is not None:
            return

        stars_model = NavModelStars(self._obs)
        stars_model.create_model()
        self._star_models = [stars_model]

        # plt.imshow(stars_model.model_img)
        # plt.show()

    def compute_body_models(self) -> None:

        if self._body_models is not None:
            return

        obs = self._obs
        config = self._config
        logger = self._logger

        body_list = [obs.closest_planet] + config.satellites(obs.closest_planet)

        large_body_dict = self._obs.inventory(body_list, return_type='full')
        # Make a list sorted by range, with the closest body first, limiting to bodies
        # that are actually in the FOV
        def _body_in_fov(obs: Observation,
                         inv: dict[str, Any]) -> bool:
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

        for body, inventory in large_bodies_by_range:
            body_model = NavModelBody(obs, body,
                                      inventory=inventory,
                                      config=config)
            body_model.create_model()
            self._body_models.append(body_model)

    def compute_ring_models(self) -> None:

        if self._ring_models is not None:
            return
        self._ring_models = []
        # TODO Ring models

    def compute_titan_models(self) -> None:

        if self._titan_models is not None:
            return
        self._titan_models = []
        # TODO Titan models

    def compute_all_models(self) -> None:

        self.compute_star_models()
        self.compute_ring_models()
        self.compute_body_models()
        self.compute_titan_models()

    def _create_combined_model(self):

        if self._combined_model is not None:
            return self._combined_model

        # Create a single model which, for each pixel, has the element from the model with
        # the smallest range (to the observer), and is thus in front.
        model_imgs = []
        ranges = []
        for model in self.all_models:
            if model.model_img is None:
                continue
            model_imgs.append(model.model_img)
            # Range can just be a float if the entire model is at the same distance
            range = model.range
            if not isinstance(range, np.ndarray):
                if model.range is None:
                    range = 0
                else:
                    range = model.range
                range = cast(NDArrayFloatType, np.zeros_like(model.model_img)) + range
            ranges.append(range)

        if len(model_imgs) == 0:
            self._combined_model = None
            return

        model_imgs_arr = np.array(model_imgs)
        ranges_arr = np.array(ranges)
        min_indices = np.argmin(ranges_arr, axis=0)
        row_idx, col_idx = np.indices(min_indices.shape)
        final_model = model_imgs_arr[min_indices, row_idx, col_idx]

        self._combined_model = cast(NDArrayFloatType, final_model)

    def navigate(self) -> None:
        self.compute_all_models()

        nav_stars = NavTechniqueStars(self)
        nav_stars.navigate()
        self._offsets['stars'] = nav_stars.offset

        nav_all = NavTechniqueAllModels(self)
        nav_all.navigate()
        self._offsets['all_models'] = nav_all.offset

        if nav_stars.offset is not None:  # TODO More logic here
            self._offset = nav_stars.offset
        else:
            self._offset = nav_all.offset

    def create_overlay(self) -> None:

        obs = self._obs

        annotations = Annotations()

        for model in self.all_models:
            annotations.add_annotations(model.annotations)
            dump_yaml(model.metadata)

        offset = (0., 0.)
        if self._offset is not None:
            offset = self._offset

        overlay = annotations.combine(offset=offset,
                                      #   text_use_avoid_mask=False,
                                      #   text_show_all_positions=True,
                                      #   text_avoid_other_text=False
                                      )
        img = self._obs.data.astype(np.float64)

        res = np.zeros(img.shape + (3,), dtype=np.uint8)

        img_sorted = sorted(list(img.flatten()))
        blackpoint = img_sorted[np.clip(int(len(img_sorted)*0.005),
                                        0, len(img_sorted)-1)]
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*0.995),
                                        0, len(img_sorted)-1)]
        gamma = 0.5

        img_stretched = np.floor((np.maximum(img-blackpoint, 0) /
                                (whitepoint-blackpoint))**gamma*256)
        img_stretched = np.clip(img_stretched, 0, 255)

        img_stretched = img_stretched.astype(np.uint8)

        res[:, :, 0] = img_stretched
        res[:, :, 1] = img_stretched
        res[:, :, 2] = img_stretched

        if overlay is not None:
            overlay[overlay < 128] = 0
            mask = np.any(overlay, axis=2)
            res[mask, :] = overlay[mask, :]

        im = Image.fromarray(res)
        fn = Path(obs.basename).stem
        im.save(f'/home/rfrench/{fn}.png')

        plt.imshow(res)
        # plt.figure()
        # model_mask = body_model.model_mask
        # plt.imshow(model_mask)
        plt.show()
