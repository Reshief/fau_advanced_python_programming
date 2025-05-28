#!/usr/bin/env python3

import argparse
import sys
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0], description="Script to display the files in a directory in alphabetic order \n\n"\
    "" \
    "The `input_folder` option needs to point to a valid directory. An error will be logged if the folder does not exist")

    parser.add_argument(
        "-log",
        choices=["info", "debug", "warn", "error",],
        default="warn",
        help="The log level to be set on the logger"
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        default=".",
        help="The input folder for future processing. Default is the current working directory ('.')"
    )

    args = parser.parse_args(sys.argv[1:])

    loglevel = args.log
    input_folder = args.input_folder

    logmap = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,}
    
    logger.setLevel(logmap[loglevel])

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logmap[loglevel])

    logger.addHandler(stream_handler)

    logger.info("The loglevel is '{}'".format(loglevel))
    logger.info("The directory '{}' will be analyzed".format(input_folder))

    # catch possible issues with the input folder
    if not os.path.exists(input_folder):
        logger.error("The input path '{}' does not exist".format(input_folder))
        sys.exit(1)
    if not os.path.isdir(input_folder):
        logger.error("The input path '{}' is not a directory".format(input_folder))
        sys.exit(1)
    
    logfile_handler = logging.FileHandler(input_folder+"/.basic.log", encoding="utf-8")
    logfile_handler.setLevel(logmap[loglevel])
    logger.addHandler(logfile_handler)

    



