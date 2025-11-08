from typing import Optional, cast

import numpy as np

from nav.config import Config
from nav.nav_model import NavModel
from nav.obs import ObsSnapshot
from nav.support.image import gaussian_blur_cov, normalize_array
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint32Type


class NavModelCombined(NavModel):
    """A NavModel representing a combination of multiple other models."""

    def __init__(self,
                 name: str,
                 obs: ObsSnapshot,
                 models: list[NavModel],
                 *,
                 config: Optional[Config] = None) -> None:
        """A combined navigation model created by combining multiple models.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing the image data.
            models: List of navigation models to combine.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(name, obs, config=config)

        self._models = models
        self._closest_model_index: NDArrayUint32Type | None = None

        if len(models) == 0:
            return

        shape = None

        model_imgs: list[NDArrayFloatType] = []
        model_masks: list[NDArrayBoolType] = []
        weighted_model_masks: list[NDArrayFloatType] = []
        ranges: list[NDArrayFloatType] = []
        total_w = 0.
        for model in models:
            if model.model_img is None:
                continue
            if model.model_mask is None:
                raise ValueError(f'Model mask is None for model: {model.name}')
            if model.confidence is None:
                raise ValueError(f'Model confidence is None for model: {model.name}')
            if shape is None:
                shape = model.model_img.shape
            elif shape != model.model_img.shape:
                raise ValueError(f'Model image shapes differ: {shape} != '
                                 f'{model.model_img.shape}')
            if model.model_mask is None or model.model_mask.shape != shape:
                raise ValueError(f'Model image and mask shapes differ: {shape} != '
                                 f'{model.model_mask.shape}')

            model_img = normalize_array(model.model_img)
            if model.blur_amount is not None:
                model_img = gaussian_blur_cov(model_img, cast(NDArrayFloatType, model.blur_amount))
            model_img *= model.confidence
            wt_model_mask = model.model_mask.astype(np.float64) * model.confidence
            total_w += model.confidence
            model_imgs.append(model_img)
            model_masks.append(model.model_mask)
            weighted_model_masks.append(wt_model_mask)

            # Range can just be a float if the entire model is at the same distance
            rng = model.range
            if not isinstance(rng, np.ndarray):
                rng = 0 if rng is None else rng
                rng = np.zeros_like(model.model_img) + rng
            elif rng.shape != model.model_img.shape:
                raise ValueError(f'Range shape differs from model image shape: {rng.shape} != '
                                 f'{model.model_img.shape}')
            ranges.append(rng)

        if len(model_imgs) == 0:
            return

        model_imgs_arr = np.stack(model_imgs, axis=0)
        model_masks_arr = np.stack(model_masks, axis=0)
        weighted_model_masks_arr = np.stack(weighted_model_masks, axis=0)
        ranges_arr = np.stack(ranges, axis=0)

        min_indices = np.argmin(ranges_arr, axis=0)
        row_idx, col_idx = np.indices(min_indices.shape)
        final_model = model_imgs_arr[min_indices, row_idx, col_idx]
        final_mask = model_masks_arr[min_indices, row_idx, col_idx]
        final_weighted_mask = weighted_model_masks_arr[min_indices, row_idx, col_idx]

        if total_w > 0:
            final_model /= total_w
            final_weighted_mask /= total_w

        final_weighted_mask = np.clip(final_weighted_mask, 0.0, 1.0)
        self._model_img = final_model
        self._model_mask = final_mask
        self._weighted_mask = final_weighted_mask
        self._range = ranges_arr[min_indices, row_idx, col_idx]
        self._closest_model_index = min_indices

    @property
    def closest_model_index(self) -> NDArrayUint32Type | None:
        """Returns the index of the closest object for each pixel."""
        return self._closest_model_index

    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_annotations: bool = True) -> None:
        """Creates the combined model.

        Doesn't do anything here because the model is already created.
        """
        pass
