# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab import lab

# pylint does not understand click's decorators
lab.cli()  # pylint: disable=no-value-for-parameter
