# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import sys

FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d: %(funcName)s %(message)s"


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
    root.setLevel(log_level)

    if log_level == "DEBUG":
        # silence noisy modules
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("markdown_it").setLevel(logging.INFO)
    # Set logging level of OpenAI client and httpx library to ERROR to suppress INFO messages
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
