#!/usr/bin/env python3

import argparse
import sys

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

    print("The loglevel is '{}'".format(loglevel))
    print("The directory '{}' will be analyzed".format(input_folder))

