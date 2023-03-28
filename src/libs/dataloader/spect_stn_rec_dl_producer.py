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


import configparser
import numpy as np
import os

from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple
from sklearn.model_selection import StratifiedKFold

from libs.dataset.spect_stn_dataset import SpectStnDataset
from libs.utils import getConfigSection, get_logger


class SpectStnRecLevelDataLoaderProducer:

    def __init__(self, cfg_path, dtype, cut_off, normalize, cap=0, batch_size=64):
        self.dtype = dtype
        self.batch_size = batch_size

        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(cfg_path)
        config = cfg_parser[getConfigSection()]
        os.makedirs(config.get('path_cache'), exist_ok=True)
        cache_file = os.path.join(config.get('path_cache'), f'spect_{dtype}')
        cache_file += '_CT' if cut_off else '_CF'
        cache_file += '_NT' if normalize else '_NF'
        if cap > 0:
            cache_file += f'_CAP{cap}'
        split_file = f'{cache_file}_split.npz'
        logger = get_logger(f'SDP_{dtype}')
        self.dataset = SpectStnDataset(dtype, cut_off, normalize, cap, cache_file, config.get('path_log'))

        if os.path.exists(split_file):
            with np.load(split_file) as npfh:
                id_rec_tr = npfh['id_rec_tr']       # ids of train records from data_ex
                id_rec_vl = npfh['id_rec_vl']
                id_rec_ts = npfh['id_rec_ts']
                y_rec_tr = npfh['y_rec_tr']         # classes of above train records
                y_rec_vl = npfh['y_rec_vl']
                y_rec_ts = npfh['y_rec_ts']
        else:
            tmp = np.concatenate([np.expand_dims(self.dataset.recid, 1), np.expand_dims(self.dataset.cache_y, 1)], axis=1)
            tmp = np.unique(tmp, axis=0)
            id_rec = tmp[:, 0]       # id of recording in data_ex
            y_rec = tmp[:, 1]         # class of this recording

            skf = StratifiedKFold(n_splits=5, shuffle=True)
            train_index, vt_index = skf.split(id_rec, y_rec).__next__()
            id_rec_tr, id_rec_vt = id_rec[train_index], id_rec[vt_index]
            y_rec_tr, y_rec_vt = y_rec[train_index], y_rec[vt_index]

            skf = StratifiedKFold(n_splits=2, shuffle=True)
            validation_index, test_index = skf.split(id_rec_vt, y_rec_vt).__next__()
            id_rec_vl, id_rec_ts = id_rec_vt[validation_index], id_rec_vt[test_index]
            y_rec_vl, y_rec_ts = y_rec_vt[validation_index], y_rec_vt[test_index]

            logger.info(f'Saving {split_file}')
            np.savez_compressed(split_file, id_rec_tr=id_rec_tr, id_rec_vl=id_rec_vl, id_rec_ts=id_rec_ts,
                                y_rec_tr=y_rec_tr, y_rec_vl=y_rec_vl, y_rec_ts=y_rec_ts)

        s_tr = set(id_rec_tr)
        s_vl = set(id_rec_vl)
        s_ts = set(id_rec_ts)
        assert len(s_tr.intersection(s_vl)) == 0
        assert len(s_tr.intersection(s_ts)) == 0
        assert len(s_vl.intersection(s_ts)) == 0

        # recordings are stratified
        self.train, self.train_cnt = self.calculate_weights(id_rec_tr, y_rec_tr)
        self.validation, self.validation_cnt = self.calculate_weights(id_rec_vl, y_rec_vl)
        self.test, self.test_cnt = self.calculate_weights(id_rec_ts, y_rec_ts)


    # calculate wieghts for items in dataset
    def calculate_weights(self, recids, reccls):
        # r2i = self.dataset.getRecordToItemsMap()
        cnt = {0: 0, 1: 0}
        for j, recid in enumerate(recids):
            indices = np.where(self.dataset.recid == recid)[0]
            cnt[reccls[j]] += indices.shape[0]
        w = {0: 1, 1: cnt[0] / cnt[1]}
        weights = np.zeros((self.dataset.__len__()), dtype=np.float32)

        done = set()
        for j, recid in enumerate(recids):
            cls = reccls[j]
            indices = np.where(self.dataset.recid == recid)[0]

            for j in range(indices.shape[0]):
                indice = indices[j]
                weights[indice] = w[cls]
                assert indice not in done
                done.add(indice)

        return weights, cnt[0] + cnt[1]

    def produce(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dl_tr = DataLoader(self.dataset, self.batch_size, pin_memory=False, num_workers=1,
                           sampler=WeightedRandomSampler(self.train, self.train_cnt))
        dl_vl = DataLoader(self.dataset, self.batch_size, pin_memory=False, num_workers=1,
                           sampler=WeightedRandomSampler(self.validation, self.validation_cnt))
        dl_ts = DataLoader(self.dataset, self.batch_size, pin_memory=False, num_workers=1,
                           sampler=WeightedRandomSampler(self.test, self.test_cnt))

        return dl_tr, dl_vl, dl_ts
