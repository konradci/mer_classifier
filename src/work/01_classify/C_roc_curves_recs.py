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
import locale
import time

import numpy as np
import os
import pandas as pd
import re
import torch
import torch.cuda.amp as amp

from enum import Enum
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from libs.grapher import gen_fig_roc
from libs.model.classifier_ResNet_Attention import Classifier_ResNet_Time_Attention_Drop
from libs.model.classifier_ResNet_NoAttention import Classifier_ResNet_NoAttention
from libs.utils import get_logger, getConfigSection, count_splits, make_splits, spectrographer, timer


logger = get_logger('ROC')

"""
Generate ROC curves for whole recordings classification.
"""


def main():
    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    cfg_parser = configparser.ConfigParser()
    pwd_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.abspath(os.path.join(pwd_dir, '../../config'))
    cfg_path = os.path.join(cfg_dir, 'config_coi.ini')
    cfg_parser.read(cfg_path)
    config = cfg_parser[getConfigSection()]

    path_checkpoint = config.get('path_checkpoint')
    path_roc = config.get('path_roc')
    os.makedirs(path_roc, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(path_checkpoint):
        for filename in [fn for fn in filenames if fn.lower().endswith('.pt')]:
            worker(config, dirpath, filename, path_roc)


def worker(config, i_dir, i_file, o_dir):
    pattern = re.compile(r'^([a-z]+)_(\w{4})_(\d{4})_(\d)(\d)_\[(\w+?)_(\d+)\]')
    match = pattern.search(i_file)
    if match:
        kind = match.group(1)
        network = match.group(2)
        layers = match.group(3)
        lr = match.group(4)
        wd = match.group(5)
        prfx = match.group(6)
        npg = int(match.group(7))

        if network == 'RAT6':
            cls_class = Classifier_ResNet_Time_Attention_Drop
        elif network == 'RNA6':
            cls_class = Classifier_ResNet_NoAttention
        else:
            raise ValueError('bad network type')
    else:
        raise ValueError('bad file naming')

    fb, _ = os.path.splitext(i_file)

    pt_file = os.path.join(i_dir, i_file)

    for m in Mode:
        roc_npz_file = os.path.join(o_dir, f'{m.name}_rec_{fb}.npz')
        if os.path.exists(roc_npz_file):
            with np.load(roc_npz_file) as npfh:
                results_rec = npfh['results_rec']
        else:
            results_rec = calculate_roc_data(config, cls_class, layers, prfx, npg, pt_file, m, roc_npz_file)

        roc(results_rec, os.path.join(o_dir, f'{m.name}_rec_{fb}'))


@timer(logger=logger)
def calculate_roc_data(config, cls_class, layers, prfx, npg, pt_file, mode, roc_npz_file):
    dtype = f'{prfx}_{npg:04d}'
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    blocks = [int(c) for c in layers]

    logger.info('Creating network instance')
    clsb = cls_class(129, 53, blocks).to(device)
    logger.info(f'Loading checkpoint file {pt_file}')
    checkpoint = torch.load(pt_file)
    logger.info('Setting weights')
    clsb.load_state_dict(checkpoint['classifier'])
    logger.info('Setting eval mode')
    clsb.eval()

    if mode == Mode.TS:
        raw_file = os.path.join(config.get('path_raw'), f'raw_test.npz')
        meta_file = os.path.join(config.get('path_raw'), f'raw_test.feather')
    elif mode == Mode.VL:
        raw_file = os.path.join(config.get('path_raw'), f'raw_validation.npz')
        meta_file = os.path.join(config.get('path_raw'), f'raw_validation.feather')
    else:
        raise ValueError('bad mode')

    if raw_file in raw_cache:
        logger.info('Loading test data from cache')
        data, lens, clss = raw_cache[raw_file]
    else:
        logger.info(f'Loading test data from {raw_file}')
        with np.load(raw_file) as npfh:
            data = npfh['data']
        meta = pd.read_feather(meta_file)
        lens = meta['length'].to_numpy()
        clss = meta['class'].to_numpy()
        raw_cache[raw_file] = (data, lens, clss)

    chunks_total = 0
    for i in range(data.shape[0]):
        data_len = lens[i]
        chunks_total += count_splits(data_len, 12_000, 2_400)

    results_rec = np.zeros((data.shape[0], 2), dtype=np.float32)

    time.sleep(0.1)
    for i in tqdm(range(data.shape[0])):
        data_len = lens[i]
        raw_data = data[i, :data_len]
        cls = clss[i]
        if cls > 1:  # SNr ---> MISS
            cls = 0

        splits1 = make_splits(raw_data, 12_000, 2_400)
        splits2 = spectrographer(splits1, 24000, npg).astype(np.float16)
        splits2 /= 503.2        # normalize
        splits3 = torch.tensor(splits2).to(device)
        with amp.autocast():
            with torch.no_grad():
                ret = clsb.forward(splits3)

        rets = torch.sigmoid(torch.nan_to_num(ret))
        r_m = torch.mean(rets).detach().cpu().item()
        results_rec[i, ::] = [r_m, cls]

    np.savez_compressed(roc_npz_file, results_rec=results_rec)
    return results_rec


def roc(data: np.ndarray, fb):
    pred = data[:, 0]
    clss = data[:, 1]
    clss = np.greater_equal(clss, 1)
    _clss = np.logical_not(clss)

    b_vv = 2
    b_ss = (0, 0)
    b_th = 0

    v_min = 0.0
    v_max = 1.0
    nn = 1000
    points = []
    time.sleep(0.1)
    for threshold in np.linspace(v_min, v_max, nn):
        decisions = np.greater_equal(pred, threshold)
        _decisions = np.logical_not(decisions)
        tp = np.logical_and(decisions, clss)
        fn = np.logical_and(_decisions, clss)
        tn = np.logical_and(_decisions, _clss)
        fp = np.logical_and(decisions, _clss)
        tpc = np.sum(tp.astype(np.float32))
        fnc = np.sum(fn.astype(np.float32))
        tnc = np.sum(tn.astype(np.float32))
        fpc = np.sum(fp.astype(np.float32))
        sns = tpc / (tpc + fnc)
        spc = tnc / (tnc + fpc)
        # logger.info(f'{i:3d}\t\t{threshold:5.3f}\t\t\t{sns:5.3f}\t\t\t{spc:5.3f}\t\t\t{sns + spc:5.3f}')
        points.append((1 - spc, sns))

        vv = (1 - sns)**2 + (1 - spc)**2
        if vv <= b_vv:
            b_vv = vv
            b_ss = (sns, spc)
            b_th = threshold

    points.reverse()
    auroc = 0
    for i in range(len(points) - 1):
        auroc += (points[i][1] + points[i + 1][1]) * (points[i + 1][0] - points[i][0]) / 2
    logger.info(f'AUC ROC {auroc:7.4f}')
    logger.info(f'AUC ROC {roc_auc_score(clss, pred):7.4f}')

    data_x = np.array([x for (x, y) in points], dtype=np.float32)
    data_y = np.array([y for (x, y) in points], dtype=np.float32)

    fig = gen_fig_roc(data_x, data_y, (-0.01, 1.01), (-0.01, 1.01), '1-specificity', 'sensitivity', linewidth=5,
                      redDot=(1 - b_ss[1], b_ss[0]), aur_value=f'{auroc:6.3f}',
                      redDotLbl=f'threshold:  {b_th:4.3f}\nsensitivity: {b_ss[0]:4.3f}\nspecificity: {b_ss[1]:4.3f}',
                      figsize=(12, 12), fontsize=36)
    fig.savefig(f'{fb}.eps', dpi='figure')


class Mode(Enum):
    VL = 1
    TS = 2


if __name__ == '__main__':
    raw_cache = {}
    main()
