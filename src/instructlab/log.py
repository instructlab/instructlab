# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import sys

FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.filename == os.path.basename(__file__):
            # Replace our current file name with 'llama_cpp' since the message is coming from stdout or
            # stderr. Printing this file name is not useful in this case, we need to print where the line is coming from.
            record.filename = "llama_cpp"
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


def stdout_stderr_to_logger(logger):
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)
