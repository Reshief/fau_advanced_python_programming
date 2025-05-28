#!/usr/bin/env python3

import argparse
import sys
import logging
import pathlib
from typing import Tuple, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_directory_listing(directory_path: pathlib.Path, local_logger: logging.Logger) -> Union[Tuple[List[str], List[str]],None]:
    """
    Function to list the visible files and directories within the provided folder.

    If the `directory_path` does not exist or isn't a directory, the function will output an error to the provided `local_logger`.
    If the `directory_path` is appropriate, the `local_logger` will be extended with a FileHandler logging all future outputs to a file '.basic.log' in that directory.

    The function skips over all hidden entries as well as all non-file and non-directory entries.
    Encounters of such entries will be logged as info-messages.

    Parameters
    ----------
    directory_path : pathlib.Path
        The path object pointing to a directory to be analyzed
    local_logger : logging.Logger
        The logger to use for outputting log messages within this function
    
    Returns
    ----------
    Tuple[List[str], List[str]] or None
        In case of an error, None is returned. Otherwise, a pair of lists, the first containing the names of visible files the second containing the names of visible directories within `directory_path` will be returned.

    Warns
    ----------
    No visible entries in the chosen directory:
        Will be logged to the logger in case no entries were found in the target directory

    
    """

    # pay close attention that you only work on the local variables defined via the parameters of this function. 
    # Naming conflicts and accidental references to global variables are very common when shifting code into a function

    # catch possible issues with the input folder
    if not directory_path.exists():
        local_logger.error("The input path '{}' does not exist".format(directory_path))
        return None
    if not directory_path.is_dir():
        local_logger.error("The input path '{}' is not a directory".format(directory_path))
        return None
    
    # create log file in the 
    logfile_handler = logging.FileHandler(directory_path / ".basic.log", encoding="utf-8")
    # Instead of using the argument from the command line, we use the logging level already set for the logger provided to us
    logfile_handler.setLevel(local_logger.getEffectiveLevel())
    local_logger.addHandler(logfile_handler)

    files_list: List[str] = []
    directories_list: List[str] = []

    # iterate over the directory contents:
    for entry in directory_path.iterdir():
        # Ensure entry is a file or a directory
        if not entry.is_file() and not entry.is_dir():
            local_logger.info("Encountered non-file and non-directory '{}'".format(entry.name))
            continue
        # Check whether entry is hidden
        if entry.name.startswith("."):
            local_logger.info("Encountered hidden entry '{}'".format(entry.name))
            continue
        
        # Add entries to respective lists
        if entry.is_file():
            files_list.append(entry.name)
        if entry.is_dir():
            directories_list.append(entry.name)

    # Sort alphabetically
    files_list.sort()
    directories_list.sort()

    # Output empty warning
    if len(files_list) == 0 and len(directories_list) == 0:
        local_logger.warning("No visible entries in the chosen directory")

    # we should remove the file handler once we are done to clean up. It's good practice
    local_logger.removeHandler(logfile_handler)

    return (files_list, directories_list)


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


    directory_result =  create_directory_listing(pathlib.Path(input_folder), logger)

    if directory_result is None:
        # deal with error return value
        print("Failed to retrieve directory contents")
        sys.exit(1)
    else:
        # only unpack the results once you know it is not the error return value
        files, directories = directory_result

        # output the results
        print("FILES:")
        for file in files:
            print(file)

        print("FOLDERS:")
        for folder in directories:
            print(folder)