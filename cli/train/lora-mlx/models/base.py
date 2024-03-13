# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import inspect


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
