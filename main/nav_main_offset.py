################################################################################
# cb_main_offset.py
#
# This is the main top-level driver for offset finding. It enumerates one or
# more images either from scanning an index file, a holdings directory,
# or from an AWS SQS queue, and for each computes the offset and saves the
# resulting offset and preview image files.
################################################################################

import argparse
# import cProfile
# import datetime
# import io
# import logging
# import os
# import pstats
# import subprocess
from pathlib import Path
import sys
import time

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.dataset import (DataSetCassiniISS,
                         DataSetGalileoSSI,
                         DataSetNewHorizonsLORRI,
                         DataSetVoyagerISS)
from nav.config.logger import DEFAULT_IMAGE_LOGGER

# import tkinter
# import traceback

# import spicedb

# import nav.aws
# import nav.config
# import nav.file
# import nav.file_oops
# import nav.gui_offset_data
# import nav.logging_setup
# import nav.misc
# from   nav.nav import Navigation
# import nav.offset
# import nav.web


MAIN_LOG_NAME = "nav_main_offset"

DATASET = None


################################################################################
#
# ARGUMENT PARSING
#
################################################################################

def parse_args(command_list):
    dataset_id = command_list[0]

    global DATASET

    match dataset_id.upper():
        case 'COISS':
            DATASET = DataSetCassiniISS()
        case 'GOSSI':
            DATASET = DataSetGalileoSSI()
        case 'NHLORRI':
            DATASET = DataSetNewHorizonsLORRI()
        case 'VGISS':
            DATASET = DataSetVoyagerISS()
        case _:
            raise ValueError(f'Unknown dataset {dataset_id}')

    cmdparser = argparse.ArgumentParser(
        description="Navigation & Backplane Main Interface for Offsets",
        epilog="""Default behavior is to perform an offset pass on all Cassini images
                that don't have associated offset files""")

    # Arguments about the general navigation process
    nav_group = cmdparser.add_argument_group("Navigation")
    nav_group.add_argument(
        "--force-offset", action="store_true", default=False,
        help="Force offset computation even if the offset file exists")
    nav_group.add_argument(
        "--display-offset-results", action="store_true", default=False,
        help="Graphically display the results of the offset process")
    nav_group.add_argument(
        "--force-offset-amount", type=str, metavar="U,V",
        help="Force the offset to be u,v")
    nav_group.add_argument(
        "--stars-only", action="store_true", default=False,
        help="""Navigate only using stars;
                implies --no-allow-rings --no-allow-moons --no-allow-central-body""")
    nav_group.add_argument(
        "--allow-stars", action=argparse.BooleanOptionalAction, default=True,
        help="Include stars in navigation")
    nav_group.add_argument(
        "--rings-only", action="store_true", default=False,
        help="""Navigate only using rings;
                implies --no-allow-stars --no-allow-moons --no-allow-central-body""")
    nav_group.add_argument(
        "--allow-rings", action=argparse.BooleanOptionalAction, default=True,
        help="Include rings in navigation")
    nav_group.add_argument(
        "--moons-only", action="store_true", default=False,
        help="""Navigate only using moons;
                implies --no-allow-stars --no-allow-rings --no-allow-central-body""")
    nav_group.add_argument(
        "--allow-moons", action=argparse.BooleanOptionalAction, default=True,
        help="Include moons in navigation")
    nav_group.add_argument(
        "--central-planet-only", action="store_true", default=False,
        help="""Navigate only using the central planet (e.g. Saturn, Jupiter);
                implies --no-allow-stars --no-allow-rings --no-allow-moons""")
    nav_group.add_argument(
        "--allow-central-planet", action=argparse.BooleanOptionalAction, default=True,
        help="Include the central planet in navigation")
    nav_group.add_argument(
        "--body-cartographic-data", dest="body_cartographic_data", action="append",
        metavar="BODY=MOSAIC",
        help="""The mosaic providing cartographic data for the given BODY
                (not currently supported)""")
    nav_group.add_argument(
        "--use-predicted-kernels", action=argparse.BooleanOptionalAction, default=False,
        help="Use predicted CK kernels")
    nav_group.add_argument(
        "--use-gapfill-kernels", action=argparse.BooleanOptionalAction, default=False,
        help="Use gapfill kernels")
    nav_group.add_argument(
        "--use-kernel", action="append",
        help="Use specified CK kernel(s)")
    nav_group.add_argument(
        "--use-cassini-nac-wac-offset", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the computed offset between NAC and WAC frames")

    # Arguments about offset, overlay, and PNG file generation
    output_group = cmdparser.add_argument_group("Output")
    output_group.add_argument(
        "--write-offset-file", action=argparse.BooleanOptionalAction, default=True,
        help="Generate an offset file; no implies --no-overlay-file")
    output_group.add_argument(
        "--write-overlay-file", action=argparse.BooleanOptionalAction, default=True,
        help="Generate an overlay file")
    output_group.add_argument(
        "--write-png-file", action=argparse.BooleanOptionalAction, default=True,
        help="Generate a PNG file")
    output_group.add_argument(
        "--no-write-results", action="store_true", default=False,
        help="""Don't write offset, overlay, or PNG files;
                implies --no-write-offset-file --no-write-overlay-file
                --no-write-png-file""")
    output_group.add_argument(
        "--png-also-bw", action=argparse.BooleanOptionalAction, default=False,
        help="Produce a black and white PNG file along with the color one")
    output_group.add_argument(
        "--png-blackpoint", type=float, default=None,
        help="Set the blackpoint for the PNG file")
    output_group.add_argument(
        "--png-whitepoint", type=float, default=None,
        help="Set the whitepoint for the PNG file")
    output_group.add_argument(
        "--png-gamma", type=float, default=None,
        help="Set the gamma for the PNG file")
    output_group.add_argument(
        "--metadata-label-font", type=str, default=None, metavar="FONTFILE,SIZE",
        help="Set the font for the PNG metadata info")
    output_group.add_argument(
        "--stars-label-font", type=str, default=None, metavar="FONTFILE,SIZE",
        help="Set the font for star labels")
    output_group.add_argument(
        "--rings-label-font", type=str, default=None, metavar="FONTFILE,SIZE",
        help="Set the font for ring labels")
    output_group.add_argument(
        "--bodies-label-font", type=str, default=None, metavar="FONTFILE,SIZE",
        help="Set the font for body labels (moons and central planet)")
    output_group.add_argument(
        "--label-rings-backplane", action=argparse.BooleanOptionalAction, default=False,
        help="Label backplane longitude and radius on ring images")
    output_group.add_argument(
        "--show-star-streaks", action=argparse.BooleanOptionalAction, default=False,
        help="Show star streaks in the overlay and PNG files")

    # Add all the arguments related to selecting files
    DATASET.add_selection_arguments(cmdparser)

    # Add all the arguments related to logging
    # nav.logging_setup.add_arguments(cmdparser, MAIN_LOG_NAME, "OFFSET")

    # Arguments about subprocesses
    # subprocess_group = cmdparser.add_argument_group("Subprocess handling")
    # subprocess_group.add_argument(
    #     "--max-subprocesses", type=int, default=0, metavar="NUM",
    #     help="The maximum number jobs to perform in parallel")
    # subprocess_group.add_argument(
    #     "--max-allowed-time", type=int, default=10*60, metavar="SECS",
    #     help="The maximum time allowed for a subprocess to run")
    # subprocess_group.add_argument(
    #     "--is-subprocess", action="store_true",
    #     help="Internal flag used to indicate this process was spawned by a parent")

    # Add all the arguments related to AWS
    # aws_group = cmdparser.add_argument_group("AWS")
    # aws_group.add_argument(
    #     "--aws", action="store_true", default=False,
    #     help="""Set for running on AWS EC2; implies
    #             --retrieve-from-pds
    #             --results-in-s3 --use-sqs --no-overlay-file
    #             --deduce-aws-processors""")
    # nav.aws.aws_add_arguments(cmdparser, nav.aws.SQS_OFFSET_QUEUE_NAME,
    #                         group=aws_group)

    # Misc arguments
    misc_group = cmdparser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--profile", action=argparse.BooleanOptionalAction, default=False,
        help="Do performance profiling")

    # Arguments for retrieving image and index data
    # misc_group.add_argument(
    #     "--retrieve-from-pds", action=argparse.BooleanOptionalAction, default=False,
    #     help="""Retrieve the image files and indexes from pds-rings.seti.org instead of
    #             from the local disk""")
    # misc_group.add_argument(
    #     "--update-indexes", action=argparse.BooleanOptionalAction, default=False,
    #     help="""Update the index files from the PDS instead of using the existing
    #             locally-cached copies""")

    # Add all the arguments about the local environment
    # nav.config.config_add_override_arguments(cmdparser, group=misc_group)

    arguments = cmdparser.parse_args(command_list[1:])
    # nav.file.validate_selection_arguments(arguments)

    return arguments


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

# def collect_cmd_line(image_path, use_gapfill_kernels=False):
#     def _yes_no_arg(val, param):
#         if val:
#             return [f"--{param}"]
#         return [f"--no-{param}"]

#     ret = []

#     # -- Arguments from the main argument list --

#     if arguments.force_offset:
#         ret += ["--force-offset"]

#     # There"s no point in a set of subprocesses displaying a result UI or
#     # all forcing the same offset amount
#     assert not arguments.display_offset_results
#     assert not arguments.force_offset_amount

#     ret += _yes_no_arg(arguments.allow_stars, "allow-stars")
#     ret += _yes_no_arg(arguments.allow_rings, "allow-rings")
#     ret += _yes_no_arg(arguments.allow_moons, "allow-moons")
#     ret += _yes_no_arg(arguments.allow_central_planet, "allow-central-planet")
#     if arguments.body_cartographic_data:
#         for entry in arguments.body_cartographic_data:
#             ret += ["--body-cartographic-data", entry]
#     ret += _yes_no_arg(arguments.use_predicted_kernels, "use-predicted-kernels")
#     ret += _yes_no_arg(arguments.use_gapfill_kernels or use_gapfill_kernels,
#                        "use-gapfill-kernels")
#     if arguments.use_kernel:
#         ret += ["--use-kernel"]
#         ret.extend(arguments.use_kernel)
#     ret += _yes_no_arg(arguments.use_cassini_nac_wac_offset, "use-cassini-nac-wac-offset")

#     ret += _yes_no_arg(arguments.write_offset_file, "write-offset-file")
#     ret += _yes_no_arg(arguments.write_overlay_file, "write-overlay-file")
#     ret += _yes_no_arg(arguments.write_png_file, "write-png-file")
#     ret += _yes_no_arg(arguments.png_also_bw, "png-also-bw")
#     if arguments.png_blackpoint:
#         ret += ["--png-blackpoint", str(arguments.png_blackpoint)]
#     if arguments.png_whitepoint:
#         ret += ["--png-whitepoint", str(arguments.png_whitepoint)]
#     if arguments.png_gamma:
#         ret += ["--png-gamma", str(arguments.png_gamma)]
#     if arguments.metadata_label_font:
#         ret += ["--metadata-label-font", arguments.metadata_label_font]
#     if arguments.stars_label_font:
#         ret += ["--stars-label-font", arguments.stars_label_font]
#     if arguments.rings_label_font:
#         ret += ["--rings-label-font", arguments.rings_label_font]
#     if arguments.bodies_label_font:
#         ret += ["--bodies-label-font", arguments.bodies_label_font]
#     ret += _yes_no_arg(arguments.label_rings_backplane, "label-rings-backplane")
#     ret += _yes_no_arg(arguments.show_star_streaks, "show-star-streaks")
#     ret += _yes_no_arg(arguments.retrieve_from_pds, "retrieve-from-pds")
#     # Subprocesses don't need index files because they"re only processing one image
#     ret += ["--no-update-indexes"]

#     # -- Arguments from AWS --
#     if arguments.results_in_s3:
#         ret += ["--results-in-s3", "--aws-results-bucket", arguments.aws_results_bucket]
#     # We don't need SQS arguments because only the top-level driver is retrieving from
#     # the queue

#     # -- Arguments from file selection --
#     ret += ["--instrument-host", arguments.instrument_host]
#     # Subprocesses don't need file selection because they"re only processing one image

#     # -- Arguments from logging --
#     ret += ["--main-logfile-level", "none"]
#     ret += ["--main-console-level", "none"]
#     ret += ["--image-logfile-level", arguments.image_logfile_level]
#     ret += ["--image-console-level", arguments.image_console_level]

#     # -- Misc arguments --
#     ret += ["--is-subprocess"]
#     if arguments.override_results_root:
#         ret += ["--override-results-root", arguments.override_results_root]
#     ret += _yes_no_arg(arguments.profile, "profile")

#     ret += ["--image-full-path", image_path]

#     return ret


# SUBPROCESS_LIST = []

# def wait_for_subprocess(all=False):
#     """Wait for one (or all) subprocess slots to open up."""
#     global NUM_FILES_COMPLETED
#     ec2_termination_count = 0
#     subprocess_count = arguments.max_subprocesses-1
#     if all:
#         subprocess_count = 0
#     while len(SUBPROCESS_LIST) > 0:
#         if ec2_termination_count == 5: # Check every 5 seconds
#             ec2_termination_count = 0
#             term = nav.aws.aws_check_for_ec2_termination()
#             if term:
#                 # Termination notice! We have two minutes
#                 main_logger.error("Termination notice received - shutdown at %s",
#                                   term)
#                 exit_processing()
#         else:
#             ec2_termination_count += 1
#         cur_time = time.time()
#         for i in range(len(SUBPROCESS_LIST)):
#             (pid, old_image_path, bootstrapped, old_sqs_handle,
#              proc_start_time, proc_max_time) = SUBPROCESS_LIST[i]
#             filename = nav.file.clean_name(old_image_path)
#             if pid.poll() is not None:
#                 # The subprocess completed!
#                 if bootstrapped:
#                     type_pref = "force_bootstrap"
#                 else:
#                     type_pref = "force_plain"
#                 if arguments.results_in_s3:
#                     # If the results are stored in S3, they were already copied there
#                     # by the subprocess. Just read the metadata and delete the local
#                     # file.
#                     offset_path = nav.file.clean_join(nav.config.TMP_DIR,
#                                                       filename+".off")
#                     metadata = nav.file.read_offset_metadata_path(offset_path)
#                     nav.file.safe_remove(offset_path)
#                 else:
#                     # Otherwise the local file is the real offset file, so read it and
#                     # leave as-is.
#                     metadata = nav.file.read_offset_metadata(
#                                             old_image_path, arguments.instrument_host,
#                                             arguments.planet, type_pref=type_pref)
#                 NUM_FILES_COMPLETED += 1
#                 results = filename + " - " + nav.offset.offset_result_str(metadata)
#                 results += " (%.2f sec/image)" % ((time.time()-START_TIME)/
#                                                   float(NUM_FILES_COMPLETED))
#                 main_logger.info(results)
#                 del SUBPROCESS_LIST[i]
#                 if old_sqs_handle is not None:
#                     # It really completed, so remove it from the SQS queue
#                     nav.aws.AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
#                                                           ReceiptHandle=old_sqs_handle)
#                 break
#             if cur_time > proc_max_time:
#                 # If a subprocess has been running for too long, kill it
#                 # Note no offset file will be written in this case
#                 pid.kill()
#                 NUM_FILES_COMPLETED += 1
#                 results = filename + " - KILLED DUE TO TIMEOUT"
#                 main_logger.info(results)
#                 del SUBPROCESS_LIST[i]
#                 if old_sqs_handle is not None:
#                     # We don't want to keep trying it if it's timing out
#                     nav.aws.AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
#                                                           ReceiptHandle=old_sqs_handle)
#                 break

#         if len(SUBPROCESS_LIST) <= subprocess_count:
#             # A slot opened up! Or all processes finished. Depending on what we're
#             # waiting for.
#             break
#         time.sleep(1)

# def run_and_maybe_wait(args, image_path, bootstrapped, sqs_handle,
#                        max_allowed_time):
#     """Run one subprocess, waiting as necessary for a slot to open up."""
#     wait_for_subprocess()

#     main_logger.debug("Spawning subprocess %s", str(args))

#     pid = subprocess.Popen(args)
#     SUBPROCESS_LIST.append((pid, image_path, bootstrapped, sqs_handle,
#                             time.time(), time.time()+max_allowed_time))


###############################################################################
#
# PERFORM INDIVIDUAL OFFSETS ON NAC/WAC IMAGES
#
###############################################################################

def process_offset_one_image(image_path,
                             allow_stars=True, allow_rings=True,
                             allow_moons=True, allow_central_planet=True,
                             force_offset_amount=None, cartographic_data=None,
                             bootstrapped=False, sqs_handle=None,
                             loaded_kernel_type="reconstructed",
                             sqs_use_gapfill_kernels=False,
                             max_allowed_time=None):
    if arguments.show_image_list_only:
        main_logger.info("Would process: %s", image_path)
        return True

    global NUM_FILES_COMPLETED

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

def exit_processing():
    end_time = time.time()

    main_logger.info("Total files processed %d", NUM_FILES_PROCESSED)
    main_logger.info("Total files skipped %d", NUM_FILES_SKIPPED)
    main_logger.info("Total elapsed time %.2f sec", end_time-START_TIME)

    if arguments.profile and arguments.max_subprocesses == 0:
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        main_logger.info("Profile results:\n%s", s.getvalue())

    nav.logging_setup.close_main_logging()

    if (arguments.results_in_s3 and
        arguments.main_loglevel.upper() != "NONE"):
        nav.aws.aws_copy_to_s3(main_log_path_local,
                               arguments.aws_results_bucket, main_log_path,
                               main_logger)
        nav.file.safe_remove(main_log_path_local)

    sys.exit(0)


###############################################################################
#
# MAIN
#
###############################################################################

if __name__ == "__main__":

    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    # nav.config.config_perform_overrides(arguments)

    # RESULTS_DIR = nav.config.CB_RESULTS_ROOT

    # # We don't support B&W PNG files in S3
    # assert not arguments.results_in_s3 or not arguments.png_also_bw, \
    #     "B&W PNG not supported with S3"

    # if arguments.update_indexes:
    #     assert arguments.retrieve_from_pds, \
    #         "--update-indexes meaningless without --retrieve-from-pds"

    # if arguments.aws:
    #     arguments.retrieve_from_pds = True
    #     arguments.results_in_s3 = True
    #     arguments.use_sqs = True
    #     arguments.write_overlay_file = False
    #     arguments.deduce_aws_processors = True
    # if arguments.results_in_s3:
    #     RESULTS_DIR = ""

    # if nav.aws.AWS_ON_EC2_INSTANCE:
    #     if arguments.deduce_aws_processors:
    #         arguments.max_subprocesses = nav.aws.AWS_PROCESSORS[
    #             nav.aws.AWS_HOST_INSTANCE_TYPE]

    if arguments.stars_only:
        arguments.allow_rings = False
        arguments.allow_moons = False
        arguments.allow_central_planet = False
    if arguments.rings_only:
        arguments.allow_stars = False
        arguments.allow_moons = False
        arguments.allow_central_planet = False
    if arguments.moons_only:
        arguments.allow_stars = False
        arguments.allow_rings = False
        arguments.allow_central_planet = False
    if arguments.central_planet_only:
        arguments.allow_stars = False
        arguments.allow_rings = False
        arguments.allow_moons = False
    if arguments.no_write_results:
        arguments.write_offset_file = False
        arguments.write_overlay_file = False
        arguments.write_png_file = False

    if arguments.profile and arguments.max_subprocesses == 0:
        # Only do image offset profiling if we"re going to do the actual work in
        # this process
        pr = cProfile.Profile()
        pr.enable()

    if arguments.display_offset_results:
        root = tkinter.Tk()
        root.withdraw()

    kernel_type = "reconstructed"
    # if arguments.use_predicted_kernels:
    #     kernel_type = "predicted"
    #     assert arguments.use_kernel is None or len(arguments.use_kernel) == 0
    # if arguments.use_kernel is not None and len(arguments.use_kernel) > 0:
    #     kernel_type = "none"

    # Ignore Mercury and Pluto
    # if arguments.instrument_host == "cassini":
    #     import oops.hosts.cassini.iss as coiss
    #     coiss.initialize(ck=kernel_type, planets=(2,3,4,5,6,7,8),
    #                      offset_wac=arguments.use_cassini_nac_wac_offset,
    #                      gapfill=arguments.use_gapfill_kernels)

    if kernel_type == "none":
        for kernel in arguments.use_kernel:
            spicedb.furnish_by_filepath(kernel)
    loaded_kernel_type = kernel_type
    if loaded_kernel_type == "none":
        loaded_kernel_type = "custom" # Required by master_find_offset

    # main_log_path = arguments.main_logfile
    # main_log_path_local = main_log_path
    # if (arguments.results_in_s3 and
    #     arguments.main_loglevel.upper() != "NONE" and
    #     main_log_path is None):
    #     main_log_path_local = "/tmp/mainlog.txt" # For CloudWatch logs
    #     main_log_datetime = datetime.datetime.now().isoformat()[:-7]
    #     main_log_datetime = main_log_datetime.replace(":","-")
    #     main_log_path = "logs/"+MAIN_LOG_NAME+"/"+main_log_datetime+"-"
    #     if nav.aws.AWS_HOST_INSTANCE_ID is not None:
    #         main_log_path += nav.aws.AWS_HOST_INSTANCE_ID
    #     else:
    #         main_log_path += "-"+str(os.getpid())
    #     main_log_path += ".log"

    main_logger = DEFAULT_IMAGE_LOGGER
    # main_logger, image_logger = nav.logging_setup.setup_main_logging(
    #             MAIN_LOG_NAME, arguments.main_logfile_level,
    #             arguments.main_console_level, main_log_path_local,
    #             arguments.image_logfile_level, arguments.image_console_level)

    # image_loglevel = nav.logging_setup.decode_level(arguments.image_logfile_level)

    force_offset_amount = None

    # if arguments.force_offset_amount:
    #     if arguments.force_offset_amount[0] == ",":
    #         arguments.force_offset_amount = arguments.force_offset_amount[1:]
    #     x, y = arguments.force_offset_amount.split(",")
    #     force_offset_amount = (float(x.replace(""","")), float(y.replace(""","")))

    # offset_config = nav.config.OFFSET_DEFAULT_CONFIG.copy()
    # if arguments.metadata_label_font is not None:
    #     x, y = arguments.metadata_label_font.split(",")
    #     offset_config["font"] = (x.replace("\"",""), int(y.replace("\"","")))
    # stars_config = nav.config.STARS_DEFAULT_CONFIG.copy()
    # if arguments.stars_label_font is not None:
    #     x, y = arguments.stars_label_font.split(",")
    #     stars_config["font"] = (x.replace("\"",""), int(y.replace("\"","")))
    # rings_config = nav.config.RINGS_DEFAULT_CONFIG.copy()
    # if arguments.rings_label_font is not None:
    #     x, y = arguments.rings_label_font.split(",")
    #     rings_config["font"] = (x.replace("\"",""), int(y.replace("\"","")))
    # bodies_config = nav.config.BODIES_DEFAULT_CONFIG.copy()
    # if arguments.bodies_label_font is not None:
    #     x, y = arguments.bodies_label_font.split(",")
    #     bodies_config["font"] = (x.replace("\"",""), int(y.replace("\"","")))

    START_TIME = time.time()
    NUM_FILES_PROCESSED = 0
    NUM_FILES_SKIPPED = 0
    NUM_FILES_COMPLETED = 0

    # main_logger.info("**********************************")
    # main_logger.info("*** BEGINNING MAIN OFFSET PASS ***")
    # main_logger.info("**********************************")
    # main_logger.info("")
    # main_logger.info("Host Local Name: %s", nav.aws.AWS_HOST_FQDN)
    # if nav.aws.AWS_ON_EC2_INSTANCE:
    #     main_logger.info("Host Public Name: %s (%s) in %s", nav.aws.AWS_HOST_PUBLIC_NAME,
    #                      nav.aws.AWS_HOST_PUBLIC_IPV4, nav.aws.AWS_HOST_ZONE)
    #     main_logger.info("Host AMI ID: %s", nav.aws.AWS_HOST_AMI_ID)
    #     main_logger.info("Host Instance Type: %s", nav.aws.AWS_HOST_INSTANCE_TYPE)
    #     main_logger.info("Host Instance ID: %s", nav.aws.AWS_HOST_INSTANCE_ID)
    # main_logger.info("GIT Status:   %s", nav.misc.current_git_version())
    # main_logger.info("")
    # main_logger.info("Command line: %s", " ".join(command_list))
    # main_logger.info("")
    # main_logger.info("Subprocesses:  %d", arguments.max_subprocesses)
    # main_logger.info("")
    # main_logger.info("Allow stars:   %s", str(arguments.allow_stars))
    # main_logger.info("Allow rings:   %s", str(arguments.allow_rings))
    # main_logger.info("Allow moons:   %s", str(arguments.allow_moons))
    # main_logger.info("Allow planet:  %s", str(arguments.allow_central_planet))
    # main_logger.info("BOTSIM offset: %s", str(force_offset_amount))
    # main_logger.info("Pred kernels:  %s", str(arguments.use_predicted_kernels))
    # if kernel_type == "none":
    #     main_logger.info("Specific kernels:")
    #     for kernel in arguments.use_kernel:
    #         main_logger.info("  %s", kernel)

    cartographic_data = {}

    # if arguments.body_cartographic_data:
    #     main_logger.info("")
    #     main_logger.info("Cartographic data provided:")
    #     for cart in arguments.body_cartographic_data:
    #         idx = cart.index("=")
    #         if idx == -1:
    #             main_logger.error("Bad format for --body-cartographic-data: %s",
    #                               cart)
    #             sys.exit(-1)
    #         body_name = cart[:idx].upper()
    #         mosaic_path = cart[idx+1:]
    #         cartographic_data[body_name] = mosaic_path
    #         main_logger.info("    %-12s %s", body_name, mosaic_path)

    #     for body_name in cartographic_data:
    #         mosaic_path = cartographic_data[body_name]
    #         mosaic_metadata = nav.file.read_mosaic_metadata_path(mosaic_path)
    #         if mosaic_metadata is None:
    #             main_logger.error("No mosaic file %s", mosaic_path)
    #             sys.exit(-1)
    #         cartographic_data[body_name] = mosaic_metadata

    bootstrapped = len(cartographic_data) > 0

    # if arguments.retrieve_from_pds and arguments.update_indexes:
    #     nav.web.update_index_files_from_pds(arguments.instrument_host,
    #                                         arguments.planet, main_logger)

    # if arguments.use_sqs:
        # main_logger.info("")
        # main_logger.info("*** Using SQS to retrieve filenames")
        # main_logger.info("")
        # SQS_QUEUE = None
        # try:
        #     SQS_QUEUE = nav.aws.AWS_SQS_RESOURCE.get_queue_by_name(
        #         QueueName=arguments.sqs_queue_name)
        # except:
        #     main_logger.error("Failed to retrieve SQS queue %s", arguments.sqs_queue_name)
        #     exit_processing()
        # SQS_QUEUE_URL = SQS_QUEUE.url
        # while True:
        #     term = nav.aws.aws_check_for_ec2_termination()
        #     if term:
        #         # Termination notice! We have two minutes
        #         main_logger.error("Termination notice received - shutdown at %s",
        #                           term)
        #         exit_processing()
        #     if arguments.max_subprocesses > 0:
        #         wait_for_subprocess()
        #     messages = SQS_QUEUE.receive_messages(MaxNumberOfMessages=1,
        #                                           WaitTimeSeconds=10)
        #     for message in messages:
        #         parameters = message.body.split(";")
        #         image_path = parameters[0]
        #         sqs_use_gapfill_kernels = "--use-gapfill-kernels" in parameters[1:]
        #         max_allowed_time = None
        #         for param in parameters:
        #             if param.startswith("--max-allowed-time="):
        #                 max_allowed_time = int(param.split("=")[1])
        #                 break
        #         receipt_handle = message.receipt_handle
        #         if image_path == "DONE":
        #             # Delete it and send it again to the next instance
        #             nav.aws.AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
        #                                                   ReceiptHandle=receipt_handle)
        #             SQS_QUEUE.send_message(MessageBody="DONE")
        #             main_logger.info("DONE message received - exiting")
        #             wait_for_subprocess(all=True)
        #             exit_processing()
        #         if process_offset_one_image(
        #                         image_path,
        #                         allow_stars=arguments.allow_stars,
        #                         allow_rings=arguments.allow_rings,
        #                         allow_moons=arguments.allow_moons,
        #                         allow_central_planet=arguments.allow_central_planet,
        #                         force_offset_amount=force_offset_amount,
        #                         cartographic_data=cartographic_data,
        #                         bootstrapped=bootstrapped,
        #                         sqs_handle=receipt_handle,
        #                         loaded_kernel_type=loaded_kernel_type,
        #                         sqs_use_gapfill_kernels=sqs_use_gapfill_kernels,
        #                         max_allowed_time=max_allowed_time):
        #             NUM_FILES_PROCESSED += 1
        #             if arguments.max_subprocesses == 0:
        #                 nav.aws.AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
        #                                                       ReceiptHandle=
        #                                                         receipt_handle)
        #         else:
        #             NUM_FILES_SKIPPED += 1
        #             nav.aws.AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
        #                                                   ReceiptHandle=receipt_handle)
    # else:
        # main_logger.info("")
        # nav.file.log_arguments(arguments, main_logger.info)
        # main_logger.info("")
    if True:
        for image_path in DATASET.yield_image_filenames_from_arguments(arguments):
            if process_offset_one_image(
                            image_path,
                            allow_stars=arguments.allow_stars,
                            allow_rings=arguments.allow_rings,
                            allow_moons=arguments.allow_moons,
                            allow_central_planet=arguments.allow_central_planet,
                            force_offset_amount=force_offset_amount,
                            cartographic_data=cartographic_data,
                            bootstrapped=bootstrapped,
                            loaded_kernel_type=loaded_kernel_type):
                NUM_FILES_PROCESSED += 1
            else:
                NUM_FILES_SKIPPED += 1

    # wait_for_subprocess(all=True)

    exit_processing()
