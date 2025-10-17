from typing import Optional, cast, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .nav_technique import NavTechnique
from nav.config import Config
# from nav.support.correlate import find_correlation_and_offset
from nav.support.image import gaussian_blur_cov, normalize_array
from nav.support.types import NDArrayFloatType, NDArrayUint32Type, NDArrayBoolType

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechniqueCorrelateAll(NavTechnique):
    """Implements navigation technique using correlation across all available models.

    Parameters:
        *args: Variable length argument list passed to parent class
        **kwargs: Arbitrary keyword arguments passed to parent class
    """
    def __init__(self,
                 nav_master: 'NavMaster',
                 *,
                 config: Optional[Config] = None) -> None:
        super().__init__(nav_master, config=config)

        self._combined_model: NDArrayFloatType | None = None
        self._combined_mask: NDArrayBoolType | None = None
        self._combined_weighted_mask: NDArrayFloatType | None = None
        self._closest_model_index: NDArrayUint32Type | None = None

    @property
    def combined_model(self) -> NDArrayFloatType | None:
        """Returns the combined navigation model, creating it if necessary."""
        self._create_combined_model()
        return self._combined_model

    @property
    def combined_mask(self) -> NDArrayBoolType | None:
        """Returns the combined mask, creating it if necessary."""
        self._create_combined_model()
        return self._combined_mask

    @property
    def combined_weighted_mask(self) -> NDArrayFloatType | None:
        """Returns the combined weighted mask, creating it if necessary."""
        self._create_combined_model()
        return self._combined_weighted_mask

    @property
    def combined_range(self) -> NDArrayFloatType | None:
        """Returns the combined range, creating it if necessary."""
        self._create_combined_model()
        return self._combined_range

    @property
    def closest_model_index(self) -> NDArrayUint32Type | None:
        """Returns the index of the closest model for each pixel in the combined model."""
        self._create_combined_model()
        return self._closest_model_index

    def _create_combined_model(self, override_cache: bool = False) -> None:
        """Creates a combined model from all individual navigation models.

        For each pixel, selects the model element from the object with the smallest range
        (closest to the observer), which would appear in front of other objects. Also
        does the same for the stretch regions.

        Each model is individually processed to implement requested blurring, normalization,
        masking, and confidence adjustments.

        The result is stored in the combined_model and closest_model_index properties.

        Parameters:
            override_cache: If True, forces a re-creation of the combined model even if
                a cached version is available.
        """

        if self._combined_model is not None and not override_cache:
            # Keep cached version
            return

    # # Bodies: limb annulus extraction + gradient model
    # # Set annulus half-width in pixels from crater_height/resolution
    # base = 1.5  # pixels
    # k_bumpy = 2.0  # scale factor
    # for b in body_components or []:
    #     B = normalize(b.array)
    #     # Limb emphasis: gradient magnitude of a slightly blurred silhouette
    #     Bg = gaussian_blur_cov(B, b.U)  # optional extra uncertainty blur
    #     G = gradient_magnitude(Bg)
    #     # limb mask = thresholded gradient band, dilated by width ~ base + k*H/R
    #     thr = 0.5 * np.max(G) if np.max(G) > 0 else 0.0
    #     limb = (G >= thr).astype(np.float64)
    #     width = base + k_bumpy * (b.crater_height / max(b.resolution, 1e-12))
    #     # approximate dilation by Gaussian blur of the binary mask then clip
    #     limb_band = gaussian_blur_cov(limb, np.diag([width**2, width**2]))
    #     limb_band = (limb_band > 0.1).astype(np.float64)

    #     # model contribution = gradient magnitude within limb band
    #     M_eff += b.confidence * (G * limb_band)
    #     W_eff += b.confidence * limb_band
    #     total_w += b.confidence

    # # Rings
    # for r in ring_components or []:
    #     R = normalize(r.array)
    #     Rb = gaussian_blur_cov(R, r.U)  # anisotropic radial blur encoded in U
    #     M_eff += r.confidence * Rb
    #     W_eff += r.confidence * r.W
    #     total_w += r.confidence

        # Create a single model which, for each pixel, has the element from the model with
        # the smallest range (to the observer), and is thus in front.
        model_imgs: list[NDArrayFloatType] = []
        model_masks: list[NDArrayBoolType] = []
        weighted_model_masks: list[NDArrayFloatType] = []
        ranges: list[NDArrayFloatType] = []
        total_w = 0.
        for model in self.nav_master.all_models:
            if model.model_img is None:
                continue
            model_img = normalize_array(model.model_img)
            if model.blur_amount is not None:
                model_img = gaussian_blur_cov(model_img, model.blur_amount)
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
            ranges.append(rng)

        if len(model_imgs) == 0:
            self._combined_model = None
            return

        # Ensure shapes align
        shapes = {img.shape for img in model_imgs}
        if len(shapes) != 1:
            raise ValueError(f'Model image shapes differ: {shapes}')
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
        self._combined_model = cast(NDArrayFloatType, final_model)
        self._combined_mask = cast(NDArrayBoolType, final_mask)
        self._combined_weighted_mask = cast(NDArrayFloatType, final_weighted_mask)
        self._closest_model_index = min_indices

        if False:
            plt.imshow(self._combined_model)
            plt.title('Combined model')
            plt.figure()
            plt.imshow(self._combined_mask)
            plt.title('Combined mask')
            plt.figure()
            plt.imshow(self._combined_weighted_mask)
            plt.title('Combined weighted mask')
            plt.show()

    def navigate(self) -> None:
        """Performs navigation using correlation across all available models.

        Attempts to find correlation between observed image and combined model,
        computing the offset if successful.
        """

        with self.logger.open('NAVIGATION PASS: ALL MODELS CORRELATION'):
            obs = self.nav_master.obs
            final_model = self.combined_model
            final_mask = self.combined_mask
            final_weighted_mask = self.combined_weighted_mask
            assert final_model is not None

            model_offset_list = find_correlation_and_offset(
                obs.extdata, final_model, extfov_margin_vu=obs.extfov_margin_vu,
                logger=self.logger)

            if len(model_offset_list) > 0:
                offset = (-float(model_offset_list[0][0][0]), -float(model_offset_list[0][0][1]))
                self._offset = offset
                self._confidence = float(model_offset_list[0][1])
                self.logger.info('All models navigation technique final offset: '
                                 f'{offset[0]:.2f}, {offset[1]:.2f}')
            else:
                self._offset = None
                self._confidence = None
                self.logger.info('All models navigation technique failed')

        self._metadata['offset'] = self._offset
        self._metadata['confidence'] = self._confidence
