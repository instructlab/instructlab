# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import pathlib

# Local
from .defaults import LOG_FORMAT


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.filename == os.path.basename(__file__):
            # Replace our current file name with 'serving_backend' since the message is coming from stdout or
            # stderr. Printing this file name is not useful in this case, we need to print where the line is coming from.
            record.filename = "serving_backend"
            record.lineno = 0  # Remove line number, no point of showing it
        return super().format(record)


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message)

    def flush(self):
        pass


def add_file_handler_to_logger(
    logger: logging.Logger, log_file: pathlib.Path | None
) -> None:
    """
    Adds a file handler to the specified logger to write log messages to a file.

    This function checks if a file handler for the specified log file already exists in the logger.
    If it does, the function does nothing. Otherwise, it creates a new file handler that appends
    log messages to the specified file and adds it to the logger.

    Args:
        logger (logging.Logger): The logger to which the file handler should be added.
        log_file (pathlib.Path): The path to the log file where log messages should be written.
    """

    if log_file is None:
        return

    log_file_path = os.path.abspath(log_file)
    logger.debug(f"Checking for existing file handler for log file: {log_file}")

    # Check if a file handler for the specified log file already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == log_file_path:
                logger.debug(
                    f"Log file already exists in handler: {log_file}, doing nothing!"
                )
                return
    logger.debug(f"No file handler found for log file: {log_file}, adding one!")

    # Use the "append" mode
    mode = "a"

    # Create file handler and add it to the logger
    file_handler = logging.FileHandler(log_file, mode, encoding="utf-8")
    logger.debug(f"Adding file handler for log file: {log_file}")
    logger.addHandler(file_handler)

    # Create formatter and set it
    formatter = CustomFormatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.debug(f"Added file handler for log file: {log_file}")


def configure_logging(
    *,
    log_level: str,
    debug_level: int = 0,
    fmt: str,
) -> None:
    """Configure logging framework"""
    root = logging.getLogger()

    # reset handlers, removes existing stream logger
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    stream = logging.StreamHandler()
    stream.setFormatter(CustomFormatter(fmt=fmt))
    root.addHandler(stream)

    instructlab_lgr = logging.getLogger("instructlab")
    openai_lgr = logging.getLogger("openai")
    httpx_lgr = logging.getLogger("httpx")

    # reset loggers to inherit from root logger
    for lgr in [instructlab_lgr, openai_lgr, httpx_lgr]:
        lgr.setLevel(logging.NOTSET)

    if log_level == "DEBUG":
        if debug_level >= 2:
            # debug level 2 enables debug logging for everything
            root.setLevel(logging.DEBUG)
        else:
            # debug level 1 enables debugging for instructlab
            # other loggers are set to INFO (except for openai and httpx)
            root.setLevel(logging.INFO)
            instructlab_lgr.setLevel(logging.DEBUG)
    else:
        # INFO, WARNING, ERROR, FATAL and higher
        root.setLevel(log_level)

    if log_level != "DEBUG" or debug_level < 2:
        # Set logging level of OpenAI client and httpx library to ERROR to
        # suppress INFO messages unless debug_level is >= 2.
        openai_lgr.setLevel(logging.ERROR)
        httpx_lgr.setLevel(logging.ERROR)
