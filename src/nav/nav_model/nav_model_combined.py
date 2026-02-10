from typing import Optional, cast

import numpy as np

from nav.config import Config
from nav.obs import ObsSnapshot

from .nav_model import NavModel
from .nav_model_result import NavModelResult
from nav.support.image import gaussian_blur_cov
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint32Type

# Epsilon to avoid division by zero when scaling model images to peak 1
_COMBINE_SCALE_EPS = 1e-12


class NavModelCombined(NavModel):
    """A NavModel representing a combination of multiple other models."""

    def __init__(
        self,
        name: str,
        obs: ObsSnapshot,
        models: list[NavModel],
        *,
        config: Optional[Config] = None,
    ) -> None:
        """A combined navigation model created by combining multiple models.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing the image data.
            models: List of navigation models to combine.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(name, obs, config=config)

        self._input_models = models
        self._closest_model_index: NDArrayUint32Type | None = None

        if len(models) == 0:
            return

        # Collect all NavModelResult instances from all input models
        results: list[NavModelResult] = [r for m in models for r in m.models]

        if len(results) == 0:
            return

        shape = None
        model_imgs: list[NDArrayFloatType] = []
        model_masks: list[NDArrayBoolType] = []
        weighted_model_masks: list[NDArrayFloatType] = []
        ranges: list[NDArrayFloatType] = []
        total_w = 0.0
        valid_result_indices: list[int] = []

        for result_idx, result in enumerate(results):
            if result.model_img is None:
                continue
            if result.model_mask is None:
                raise ValueError('Result model mask is None')
            if result.confidence is None:
                raise ValueError('Result confidence is None')
            if shape is None:
                shape = result.model_img.shape
            elif shape != result.model_img.shape:
                raise ValueError(
                    f'Result image shapes differ: {shape} != {result.model_img.shape}'
                )
            if result.model_mask.shape != shape:
                raise ValueError(
                    f'Result image and mask shapes differ: {shape} != '
                    f'{result.model_mask.shape}'
                )

            # Scale each result so masked pixels have max ~1 for consistent combined range
            raw = result.model_img
            mask = result.model_mask
            masked_max = float(np.max(raw[mask])) if np.any(mask) else 1.0
            scale = max(masked_max, _COMBINE_SCALE_EPS)
            model_img = np.zeros_like(raw, dtype=np.float64)
            model_img[mask] = raw[mask] / scale

            if result.blur_amount is not None:
                model_img = np.asarray(
                    gaussian_blur_cov(
                        model_img, cast(NDArrayFloatType, result.blur_amount)
                    ),
                    dtype=np.float64,
                )
            model_img *= result.confidence
            wt_model_mask = result.model_mask.astype(np.float64) * result.confidence
            total_w += result.confidence
            model_imgs.append(model_img)
            model_masks.append(result.model_mask)
            weighted_model_masks.append(wt_model_mask)

            # Range can just be a float if the entire model is at the same distance
            rng = result.range
            if not isinstance(rng, np.ndarray):
                rng_val = np.inf if rng is None else rng
                rng_arr = self.obs.make_extfov_zeros()
                rng_arr[:, :] = np.inf
                rng_arr[result.model_mask] = rng_val
                rng = rng_arr
            elif rng.shape != result.model_img.shape:
                raise ValueError(
                    f'Range shape differs from result image shape: {rng.shape} != '
                    f'{result.model_img.shape}'
                )
            ranges.append(rng)
            valid_result_indices.append(result_idx)

        if len(model_imgs) == 0:
            return

        model_imgs_arr = np.stack(model_imgs, axis=0)
        model_masks_arr = np.stack(model_masks, axis=0)
        weighted_model_masks_arr = np.stack(weighted_model_masks, axis=0)
        ranges_arr = np.stack(ranges, axis=0)

        min_indices = np.argmin(ranges_arr, axis=0)
        index_lookup = np.asarray(valid_result_indices, dtype=np.uint32)
        row_idx, col_idx = np.indices(min_indices.shape)
        final_model = model_imgs_arr[min_indices, row_idx, col_idx]
        final_mask = model_masks_arr[min_indices, row_idx, col_idx]
        final_weighted_mask = weighted_model_masks_arr[min_indices, row_idx, col_idx]

        if total_w > 0:
            final_model /= total_w
            final_weighted_mask /= total_w

        final_weighted_mask = np.clip(final_weighted_mask, 0.0, 1.0)
        combined_result = NavModelResult(
            model_img=final_model,
            model_mask=final_mask,
            weighted_mask=final_weighted_mask,
            range=ranges_arr[min_indices, row_idx, col_idx],
            blur_amount=None,
            uncertainty=None,
            confidence=None,
            stretch_regions=None,
            annotations=None,
        )
        self._models.append(combined_result)
        self._closest_model_index = index_lookup[min_indices]

    @property
    def closest_model_index(self) -> NDArrayUint32Type | None:
        """Returns the index of the closest object for each pixel."""
        return self._closest_model_index

    def create_model(
        self,
        *,
        always_create_model: bool = False,
        never_create_model: bool = False,
        create_annotations: bool = True,
    ) -> None:
        """Creates the combined model.

        Doesn't do anything here because the model is already created.
        """
        pass
