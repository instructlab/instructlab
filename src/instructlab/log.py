# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import sys

FORMAT = "%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.filename == os.path.basename(__file__):
            # Replace our current file name with 'serving_backend' since the message is coming from stdout or
            # stderr. Printing this file name is not useful in this case, we need to print where the line is coming from.
            record.filename = "serving_backend"
            record.lineno = 0  # Remove line number, no point of showing it
        return super().format(record)


class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        if message == "\n":  # Flush the buffer when an empty line is encountered
            self.flush()

    def flush(self):
        if self.buffer:
            self.level(self.buffer.strip())
            self.buffer = ""

    def isatty(self):
        return False


def stdout_stderr_to_logger(logger, log_file):
    if log_file:
        # Use the existing log file if it exists and append to it
        mode = "a"
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode, encoding="utf-8")
        logger.addHandler(file_handler)
        # Create formatter and set it for both handlers
        formatter = CustomFormatter(FORMAT)
        file_handler.setFormatter(formatter)

    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)


def configure_logging(
    *,
    log_level: str,
    debug_level: int = 0,
    fmt: str = FORMAT,
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
        # suppress INFO messages unless debug_leve is >= 2.
        openai_lgr.setLevel(logging.ERROR)
        httpx_lgr.setLevel(logging.ERROR)
