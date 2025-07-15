from pathlib import Path

from nav.inst import Inst
from nav.nav_master import NavMaster
from nav.config import DEFAULT_LOGGER


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

def process_one_image(inst_class: Inst,
                      label_path: Path,
                      image_path: Path) -> bool:
    logger = DEFAULT_LOGGER
    with logger.open(str(image_path)):
        try:
            s = inst_class.from_file(image_path)
        except OSError as e:
            if 'SPICE(CKINSUFFDATA)' in str(e) or 'SPICE(SPKINSUFFDATA)' in str(e):
                logger.info(f'No SPICE kernel available for "{image_path}"')
                return False
            logger.info(f'Error reading image "{image_path}": {e}')
            return False

        nm = NavMaster(s)
        nm.compute_all_models()

        nm.navigate()

        nm.create_overlay()

        return True

    # if bootstrapped:
    #     type = "bootstrap"
    # else:
    #     type = "plain"

    # # if max_allowed_time is None:
    # #     max_allowed_time = arguments.max_allowed_time

    # if not arguments.results_in_s3:
    #     offset_exists = os.path.exists(nav.file.img_to_offset_path(
    #                                             image_path, arguments.instrument_host,
    #                                             type=type))
    #     if offset_exists:
    #         if not arguments.force_offset:
    #             main_logger.debug("Skipping %s - offset file exists", image_path)
    #             return False
    #         else:
    #             main_logger.debug(
    #                 "Processing %s - offset file exists but redoing offsets", image_path)
    #     else:
    #         main_logger.debug("Processing %s - no previous offset file", image_path)
    # else:
    #     main_logger.debug("Processing %s - Using S3 storage", image_path)

    # if arguments.max_subprocesses:
    #     # If we allow subprocesses, then just spawn a subprocess and let it do all
    #     # the work
    #     run_and_maybe_wait([nav.config.PYTHON_EXE, nav.config.CBMAIN_OFFSET_PY] +
    #                        collect_cmd_line(image_path, sqs_use_gapfill_kernels),
    #                        image_path,
    #                        bootstrapped, sqs_handle, max_allowed_time)
    #     return True

    # metadata = None

    # image_path_local = image_path
    # image_path_local_cleanup = None
    # image_name = nav.file.clean_name(image_path)

    # image_log_path = None
    # image_log_path_local = None
    # image_log_path_local_cleanup = None

    # offset_path = None
    # offset_path_local = None
    # offset_path_local_cleanup = None

    # overlay_path = None
    # overlay_path_local = None
    # overlay_path_local_cleanup = None

    # png_path = None
    # png_path_local = None
    # png_path_local_cleanup = None

    # ### Set up logging

    # if image_loglevel != nav.logging_setup.LOGGING_SUPERCRITICAL:
    #     if arguments.image_logfile is not None:
    #         image_log_path = arguments.image_logfile
    #     else:
    #         image_log_path = nav.file.img_to_log_path(
    #                                   image_path, arguments.instrument_host, "OFFSET",
    #                                   type=type,
    #                                   root=RESULTS_DIR,
    #                                   make_dirs=not arguments.results_in_s3)
    #     image_log_path_local = image_log_path
    #     if arguments.results_in_s3:
    #         image_log_path_local = nav.file.clean_join(nav.config.TMP_DIR,
    #                                                        image_name+"_imglog.txt")
    #         image_log_path_local_cleanup = image_log_path_local
    #     else:
    #         if os.path.exists(image_log_path_local):
    #             os.remove(image_log_path_local) # XXX Need option to not do this

    #     image_log_filehandler = nav.logging_setup.add_file_handler(
    #                                 image_log_path_local, image_loglevel)
    # else:
    #     image_log_filehandler = None
    # # >>> image_log_path_local is live

    # image_logger = logging.getLogger("cb")

    # image_logger.info("Command line: %s", " ".join(command_list))
    # image_logger.info("GIT Status:   %s", nav.misc.current_git_version())

    # ### Download the image file if necessary

    # if arguments.retrieve_from_pds:
    #     err, image_path_local = nav.web.retrieve_image_from_pds(
    #         arguments.instrument_host, arguments.planet, image_path,
    #         image_logger=image_logger)
    #     image_path_local_cleanup = image_path_local
    #     if err is not None:
    #         main_logger.error(err)
    #         image_logger.error(err)
    #         err = "Failed to retrieve file from " + image_path
    #         metadata = {"status":         "error",
    #                     "status_detail1": err,
    #                     "status_detail2": None}
    # # >>> image_path_local is live
    # # >>> image_log_path_local is live

    # ### Set up the offset path

    # if arguments.write_offset_file:
    #     offset_path = nav.file.img_to_offset_path(
    #                               image_path, arguments.instrument_host,
    #                               type=type,
    #                               root=RESULTS_DIR,
    #                               make_dirs=not arguments.results_in_s3)
    #     offset_path_local = offset_path
    #     if arguments.results_in_s3:
    #         offset_path_local = nav.file.clean_join(nav.config.TMP_DIR, image_name+".off")
    #         offset_path_local_cleanup = offset_path_local
    # # >>> image_path_local is live
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live

    # ### Read the image file

    # obs = None
    # if metadata is None:
    #     try:
    #         obs = nav.file_oops.read_img_file(
    #             image_path_local, arguments.instrument_host, arguments.planet,
    #             orig_path=image_path)
    #     except KeyboardInterrupt:
    #         raise
    #     except:
    #         main_logger.exception("File reading failed - %s", image_path)
    #         image_logger.exception("File reading failed - %s", image_path)
    #         err = "File reading failed:\n" + traceback.format_exc()
    #         metadata = {"status":         "error",
    #                     "status_detail1": str(sys.exc_info()[1]),
    #                     "status_detail2": err}

    # ### Then immediately delete the image file if necessary

    # nav.file.safe_remove(image_path_local_cleanup)
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live

    # if obs is not None:
    #     if arguments.instrument_host == "cassini":
    #         if obs.dict["SHUTTER_STATE_ID"] == "DISABLED":
    #             main_logger.info("Skipping because shutter disabled - %s",
    #                              image_path)
    #             image_logger.info("Skipping because shutter disabled - %s",
    #                               image_path)
    #             metadata = {"status":         "skipped",
    #                         "status_detail1": "Shutter disabled",
    #                         "status_detail2": None}
    #         elif obs.texp < 1e-4: # 5 ms is smallest commandable exposure
    #             main_logger.info("Skipping because zero exposure time - %s",
    #                              image_path)
    #             image_logger.info("Skipping because zero exposure time - %s",
    #                               image_path)
    #             metadata = {"status":         "skipped",
    #                         "status_detail1": "Zero exposure time",
    #                         "status_detail2": None}

    # ### Set up profiling and find the offset

    # image_pr = None
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live
    # # >>> image_pr is live

    # if metadata is None and obs is not None:
    #     if arguments.profile and arguments.is_subprocess:
    #         # Per-image profiling
    #         image_pr = cProfile.Profile()
    #         image_pr.enable()

    #     obs = Navigation(obs, arguments.instrument_host)
    #     try:
    #         metadata = nav.offset.master_find_offset(
    #                               obs, create_overlay=True,
    #                               allow_stars=allow_stars,
    #                               allow_rings=allow_rings,
    #                               allow_moons=allow_moons,
    #                               allow_central_planet=allow_central_planet,
    #                               force_offset_amount=force_offset_amount,
    #                               bodies_cartographic_data=cartographic_data,
    #                               bootstrapped=bootstrapped,
    #                               stars_show_streaks=
    #                                  arguments.show_star_streaks,
    #                               label_rings_backplane=
    #                                  arguments.label_rings_backplane,
    #                               offset_config=offset_config,
    #                               stars_config=stars_config,
    #                               rings_config=rings_config,
    #                               bodies_config=bodies_config,
    #                               loaded_kernel_type=loaded_kernel_type)
    #     except KeyboardInterrupt:
    #         raise
    #     except:
    #         main_logger.exception("Offset finding failed - %s", image_path)
    #         image_logger.exception("Offset finding failed - %s", image_path)
    #         err = "Offset finding failed:\n" + traceback.format_exc()
    #         metadata = {"bootstrapped":   bootstrapped,
    #                     "status":         "error",
    #                     "status_detail1": str(sys.exc_info()[1]),
    #                     "status_detail2": err}

    # # At this point we"re guaranteed to have a metadata dict

    # metadata["AWS_HOST_AMI_ID"] = nav.aws.AWS_HOST_AMI_ID
    # metadata["AWS_HOST_PUBLIC_NAME"] = nav.aws.AWS_HOST_PUBLIC_NAME
    # metadata["AWS_HOST_PUBLIC_IPV4"] = nav.aws.AWS_HOST_PUBLIC_IPV4
    # metadata["AWS_HOST_INSTANCE_ID"] = nav.aws.AWS_HOST_INSTANCE_ID
    # metadata["AWS_HOST_INSTANCE_TYPE"] = nav.aws.AWS_HOST_INSTANCE_TYPE
    # metadata["AWS_HOST_ZONE"] = nav.aws.AWS_HOST_ZONE
    # if "spice_kernels" not in metadata and obs is not None:
    #     try:
    #         metadata["spice_kernels"] = obs.spice_kernels
    #     except AttributeError:
    #         metadata["spice_kernels"] = None

    # ### Create the overlay path

    # if (metadata["status"] == "ok" and
    #     (arguments.write_overlay_file or arguments.write_offset_file)):
    #     overlay_path = nav.file.img_to_overlay_path(
    #                                 image_path,
    #                                 type=type,
    #                                 root=RESULTS_DIR,
    #                                 make_dirs=not arguments.results_in_s3,
    #                                 instrument_host=arguments.instrument_host)
    #     overlay_path_local = overlay_path
    #     if arguments.results_in_s3:
    #         overlay_path_local = nav.file.clean_join(nav.config.TMP_DIR,
    #                                                  image_name+".ovr")
    #         overlay_path_local_cleanup = overlay_path_local
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live
    # # >>> overlay_path_local is live
    # # >>> image_pr is live

    # ### Write the offset, possibly with an overlay

    # if arguments.write_offset_file:
    #     try:
    #         nav.file.write_offset_metadata_path(
    #                         offset_path_local, metadata,
    #                         overlay=(arguments.write_overlay_file and
    #                                  overlay_path_local is not None),
    #                         overlay_path=overlay_path_local)
    #     except KeyboardInterrupt:
    #         raise
    #     except:
    #         main_logger.exception("Offset file writing failed - %s",
    #                               image_path)
    #         image_logger.exception("Offset file writing failed - %s",
    #                                image_path)
    #         metadata = {}
    #         err = "Offset file writing failed:\n" + traceback.format_exc()
    #         metadata = {"bootstrapped":   bootstrapped,
    #                     "status":         "error",
    #                     "status_detail1": str(sys.exc_info()[1]),
    #                     "status_detail2": err}

    #         try:
    #             # It seems weird to try again, but it might work with less metadata
    #             nav.file.write_offset_metadata_path(offset_path_local, metadata,
    #                                                 overlay=False)
    #         except KeyboardInterrupt:
    #             raise
    #         except:
    #             main_logger.exception(
    #                   "Offset file writing failed - %s", image_path)
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live
    # # >>> overlay_path_local is live
    # # >>> image_pr is live

    # ### Create the PNG path

    # if metadata["status"] == "ok" and arguments.write_png_file:
    #     if arguments.png_also_bw:
    #         bw_bools = [False,True]
    #     else:
    #         bw_bools = [False]
    #     for bw_bool in bw_bools:
    #         png_path = nav.file.img_to_png_path(
    #                                 image_path,
    #                                 type=type,
    #                                 root=RESULTS_DIR,
    #                                 make_dirs=not arguments.results_in_s3,
    #                                 no_color=bw_bool,
    #                                 instrument_host=arguments.instrument_host)
    #         png_path_local = png_path
    #         if arguments.results_in_s3:
    #             png_path_local = nav.file.clean_join(nav.config.TMP_DIR,
    #                                                  image_name+".png")
    #             png_path_local_cleanup = png_path_local
    #     # >>> image_log_path_local is live
    #     # >>> offset_path_local is live
    #     # >>> overlay_path_local is live
    #     # >>> image_pr is live

    #     ### Write the PNG file

    #         png_image = nav.offset.offset_create_overlay_image(
    #                             obs, metadata,
    #                             offset_config=offset_config,
    #                             blackpoint=arguments.png_blackpoint,
    #                             whitepoint=arguments.png_whitepoint,
    #                             gamma=arguments.png_gamma,
    #                             interpolate_missing_stripes=True,
    #                             no_color=bw_bool)
    #         try:
    #             nav.file.write_png_path(png_path_local, png_image)
    #         except KeyboardInterrupt:
    #             raise
    #         except:
    #             main_logger.exception(
    #                   "PNG file writing failed - %s", image_path)

    # if metadata["status"] == "ok":
    #     NUM_FILES_COMPLETED += 1
    #     filename = nav.file.clean_name(image_path)
    #     results = filename + " - " + nav.offset.offset_result_str(metadata)
    #     results += " (%.2f sec/image)" % ((time.time()-START_TIME)/
    #                                       float(NUM_FILES_COMPLETED))
    #     main_logger.info(results)

    # if image_pr is not None:
    #     image_pr.disable()
    #     s = io.StringIO()
    #     sortby = "cumulative"
    #     ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
    #     ps.print_stats()
    #     ps.print_callers()
    #     image_logger.info("Profile results:\n%s", s.getvalue())
    # # >>> image_log_path_local is live
    # # >>> offset_path_local is live
    # # >>> overlay_path_local is live

    # nav.logging_setup.remove_file_handler(image_log_filehandler)

    # ### Copy results to S3

    # if arguments.results_in_s3:
    #     if offset_path_local is not None:
    #         nav.aws.aws_copy_to_s3(offset_path_local,
    #                                arguments.aws_results_bucket, offset_path,
    #                                main_logger)
    #     if image_log_path_local is not None:
    #         nav.aws.aws_copy_to_s3(image_log_path_local,
    #                                arguments.aws_results_bucket, image_log_path,
    #                                main_logger)
    #     if metadata["status"] == "ok":
    #         if png_path is not None:
    #             nav.aws.aws_copy_to_s3(png_path_local,
    #                                    arguments.aws_results_bucket, png_path,
    #                                    main_logger)
    #         if overlay_path is not None:
    #             nav.aws.aws_copy_to_s3(overlay_path_local,
    #                                    arguments.aws_results_bucket, overlay_path,
    #                                    main_logger)

    # if not arguments.is_subprocess:
    #     # Leave this here so it can be seen by the parent process
    #     nav.file.safe_remove(offset_path_local_cleanup)
    # nav.file.safe_remove(overlay_path_local_cleanup)
    # nav.file.safe_remove(png_path_local_cleanup)
    # nav.file.safe_remove(image_log_path_local_cleanup)

    # if metadata["status"] == "ok" and arguments.display_offset_results:
    #     nav.gui_offset_data.display_offset_data(obs, metadata, canvas_size=None)

    # return metadata["status"] == "ok"
