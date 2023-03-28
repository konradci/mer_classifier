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


import json
import os
import numpy as np

from torch.utils.data import Dataset


class SpectStnDataset(Dataset):
    def __init__(self, dtype, cut_off, normalize, cap, cache_file, path_log):
        if not (os.path.exists(f'{cache_file}.npz') and os.path.exists(f'{cache_file}.json')):
            raise RuntimeError('No data files for Dataset')

        with open(f'{cache_file}.json', 'r', encoding='utf-8') as jfh:
            meta = json.load(jfh)
            assert meta[0] == dtype
            self.dtype = dtype
            self.dim_0f = meta[1]
            self.dim_1t = meta[2]
            self.v_min = float(meta[3])
            self.v_max = float(meta[4])
            self.cut_off = (meta[5] == 1)
            self.normalize = (meta[6] == 1)

        with np.load(f'{cache_file}.npz') as npfh:
            self.cache_x = npfh['cache_x']
            self.cache_y = npfh['cache_y']
            self.recid = npfh['recid']
        return

    def __len__(self):
        return self.cache_y.shape[0]

    def __getitem__(self, item_id):
        return self.cache_x[item_id, :], self.cache_y[item_id]


def b2i(b):
    return 1 if b else 0
