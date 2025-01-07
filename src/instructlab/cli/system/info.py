# SPDX-License-Identifier: Apache-2.0
# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.system.info import get_sysinfo_by_category

logger = logging.getLogger(__name__)


@click.command()
@clickext.display_params
def info():
    """Print system information"""
    categories = get_sysinfo_by_category()
    print_system_info(categories)


def print_system_info(categories):
    """Print formatted system information"""
    for idx, (category, items) in enumerate(categories.items()):
        if idx > 0:
            print()
        print(f"{category}:")
        for key, value in items:
            print(f"  {key}: {value}")
