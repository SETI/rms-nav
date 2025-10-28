import json

from filecache import FCPath
from PIL import Image

from nav.config import DEFAULT_LOGGER
from nav.obs import Obs
from nav.dataset.dataset import ImageFiles
from nav.nav_master import NavMaster


#   *,
#   allow_stars: bool = True,
#   allow_rings: bool = True,
#   allow_moons: bool = True,
#   allow_central_planet: bool = True,
#   force_offset_amount: Optional[float] = None,
#   cartographic_data: Optional[CartographicData] = None,
#   bootstrapped=False, sqs_handle=None,
#   loaded_kernel_type="reconstructed",
#   sqs_use_gapfill_kernels=False,
#   max_allowed_time=None

def navigate_image_files(obs_class: type[Obs],
                         image_files: ImageFiles,
                         results_root: FCPath) -> bool:

    logger = DEFAULT_LOGGER

    if len(image_files.image_files) != 1:
        logger.error("Expected exactly one image per batch; got %d", len(image_files.image_files))
        return False

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path

    with logger.open(str(image_path)):
        try:
            snapshot = obs_class.from_file(image_path)
        except OSError as e:
            if 'SPICE(CKINSUFFDATA)' in str(e) or 'SPICE(SPKINSUFFDATA)' in str(e):
                logger.error('No SPICE kernel available for "%s"', image_path)
                return False
            logger.exception('Error reading image "%s"', image_path)
            return False

        nm = NavMaster(snapshot)
        nm.compute_all_models()

        nm.navigate()

        overlay = nm.create_overlay()

        public_metadata_file = results_root / (image_file.results_path_stub + '_metadata.json')
        summary_png_file = results_root / (image_file.results_path_stub + '_summary.png')

        try:
            public_metadata_file.write_text(json.dumps(nm.metadata, indent=2))
        except TypeError:
            logger.error('Metadata is not JSON serializable: %s', nm.metadata)

        png_local = summary_png_file.get_local_path()
        im = Image.fromarray(overlay)
        im.save(png_local)
        summary_png_file.upload()

        return True
