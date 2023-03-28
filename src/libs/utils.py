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


import logging
import numpy as np
import os
import psutil
import scipy.signal as signal
import subprocess
import sys
import time

from datetime import datetime
from functools import wraps


libs_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.abspath(os.path.join(libs_dir, '../config'))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_file_logger(log_dir: str, file: str, name: str, overwrite=False) -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    log_file = os.path.join(log_dir, f'{file}_debug.log')
    if overwrite and os.path.exists(log_file):
        os.unlink(log_file)
    fhd = logging.FileHandler(log_file, encoding='utf-8')
    fhd.setLevel(logging.DEBUG)
    fhd.setFormatter(formatter)

    log_file = os.path.join(log_dir, f'{file}_info.log')
    if overwrite and os.path.exists(log_file):
        os.unlink(log_file)
    fhi = logging.FileHandler(log_file, encoding='utf-8')
    fhi.setLevel(logging.INFO)
    fhi.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(fhd)
    logger.addHandler(fhi)
    return logger


logger = get_logger(__name__)


def isPosix():
    return os.name == 'posix'


def getConfigSection():
    if isPosix():
        return 'DGX'
    else:
        return 'WIN'


def poweroff():
    if isPosix():
        return

    preventer = os.path.join(config_dir, 'no_shutdown')
    if os.path.exists(preventer):
        return

    logger.info('--- power off in 120 s ---')
    off = subprocess.Popen(['c:\\Windows\\System32\\shutdown', '/s', '/t', '120'])
    off.communicate()


def get_timestamp_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def count_splits(data_len, slice_len, step):
    return np.arange(slice_len, data_len + 1, step).shape[0]


def make_splits(data, slice_len, step):
    ib = np.arange(0, len(data), step)
    ie = np.arange(slice_len, len(data) + 1, step)
    ranges = []
    for i in range(len(ie)):
        ranges.append(np.arange(ib[i], ie[i]))
    return data[np.r_[ranges]]


def spectrographer(tbl: np.ndarray, sample_rate, nperseg):
    assert tbl.dtype == np.float32
    _, _, spectrogram = signal.spectrogram(tbl[0, :], sample_rate, nperseg=nperseg)
    ret = np.zeros((tbl.shape[0], spectrogram.shape[0], spectrogram.shape[1]), dtype=np.float32)
    for i in range(tbl.shape[0]):
        _, _, spectrogram = signal.spectrogram(tbl[i, :], sample_rate, nperseg=nperseg,
                                               window=('tukey', .25), noverlap=None, nfft=None, detrend='constant',
                                               return_onesided=True, scaling='density', axis=-1, mode='psd'
                                               )
        ret[i, :, :] = spectrogram
    return ret


def timer(logger=None):
    def dec_outer(func):
        @wraps(func)
        def dec_inner(*args, **kwargs):
            t0 = time.time()
            ret = func(*args, **kwargs)
            t1 = time.time()
            if logger is None:
                print(f'Finished in {t1 - t0:.2f} s.')
            else:
                logger.info(f'Finished in {t1 - t0:.2f} s.')
            return ret
        return dec_inner
    return dec_outer
