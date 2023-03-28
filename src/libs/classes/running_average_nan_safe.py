# Copyright (C) 2023  NASK PIB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math
import torch

from ignite.metrics import RunningAverage
from typing import Union


__all__ = ["RunningAverage"]


class RunningAverageNanSafe(RunningAverage):

    def __init__(self, *args, **kwargs):
        super(RunningAverageNanSafe, self).__init__(*args, **kwargs)

    def compute(self) -> Union[torch.Tensor, float]:
        if self._value is None:
            self._value = self._get_src_value()
        else:
            if math.isnan(self._value):
                self._value = self._get_src_value()
            elif not math.isnan(self._get_src_value()):
                self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()

        return self._value
