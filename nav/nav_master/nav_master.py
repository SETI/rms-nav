from typing import Any, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from nav.nav_technique import (NavTechniqueCorrelation,
                               NavTechniqueStars,
                               NavTechniqueTitan)
from nav.support.types import NDArrayFloatType
from oops import Observation

from nav.annotation import Annotations
from nav.config import Config
from nav.nav_model import (NavModel,
                           NavModelBody,
                           NavModelRings,
                           NavModelStars,
                           NavModelTitan)
from nav.support.nav_base import NavBase
from nav.support.file import dump_yaml


class NavMaster(NavBase):
    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        super().__init__(config=config, logger_name=logger_name)

        self._obs = obs
        self._offset = (0, 0)
        self._star_models: list[NavModelStars] = []
        self._body_models: list[NavModelBody] = []
        self._ring_models: list[NavModelRings] = []
        self._titan_models: list[NavModelTitan] = []

        self._combined_model: NDArrayFloatType | None = None

    @property
    def obs(self) -> Observation:
        return self._obs

    @property
    def all_models(self) -> Sequence[NavModel]:
        return (self._star_models + self._body_models +
                self._ring_models + self._titan_models)

    def compute_star_models(self) -> None:
        ...

    def compute_body_models(self) -> None:

        if self._body_models:
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

        for body, inventory in large_bodies_by_range:
            body_model = NavModelBody(obs, body,
                                      inventory=inventory,
                                      config=config)
            body_model.create_model()
            self._body_models.append(body_model)

    def compute_ring_models(self) -> None:

        if self._ring_models:
            return

        # TODO Ring models

    def compute_titan_models(self) -> None:

        if self._titan_models:
            return

        # TODO Titan models

    def compute_all_models(self) -> None:

        self.compute_star_models()
        self.compute_ring_models()
        self.compute_body_models()
        self.compute_titan_models()

    def _create_combined_model(self) -> NDArrayFloatType | None:

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
            ranges.append(model.range)

        if len(model_imgs) == 0:
            ret = None

        model_imgs_arr = np.array(model_imgs)
        ranges_arr = np.array(ranges)
        min_indices = np.argmin(ranges_arr, axis=0)
        row_idx, col_idx = np.indices(min_indices.shape)
        final_model = model_imgs_arr[min_indices, row_idx, col_idx]

        self._combined_model = cast(NDArrayFloatType, final_model)
        return self._combined_model

    def navigate(self) -> None:
        self.compute_all_models()
        nav_corr = NavTechniqueCorrelation(self)
        nav_corr.navigate()

        self._offset = nav_corr.offset


    def create_overlay(self) -> None:

        obs = self._obs

        annotations = Annotations()

        for model in self.all_models:
            annotations.add_annotations(model.annotations)
            dump_yaml(model.metadata)

        overlay = annotations.combine(obs.extfov_margin_vu, offset=self._offset,
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

        # im = Image.fromarray(res)
        # fn = URL.split('/')[-1].split('.')[0]
        # im.save(f'/home/rfrench/{fn}.png')

        plt.imshow(res)
        # plt.figure()
        # model_mask = body_model.model_mask
        # plt.imshow(model_mask)
        plt.show()
