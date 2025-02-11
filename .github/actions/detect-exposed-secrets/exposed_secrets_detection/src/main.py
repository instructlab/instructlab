# SPDX-License-Identifier: Apache-2.0

# Local
from .cli import create_parser, run_validation


def run():
    """
    Run CLI and parse user args
    """
    parser = create_parser()
    args = vars(parser.parse_args())
    run_validation(args)


if __name__ == "__main__":
    run()
