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


import numpy as np
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt


def gen_fig_roc(data_x: np.ndarray, data_y: np.ndarray, xrange, yrange, lbl_x, lbl_y,
            figsize=(8, 8), fontsize=18, color='black', linewidth=1.0, show=False,
            redDot=None, redDotLbl=None, aur_value=None) -> matplotlib.figure.Figure:
    gen_fig(data_x, data_y, xrange, yrange, lbl_x, lbl_y, figsize, fontsize, color, linewidth, show, closeFig=False)
    fig = plt.figure(num=0)

    if redDot is None:
        trail = [int(round(x)) for x in np.linspace(0, data_x.shape[0] - 1, 11)]
        for i in trail:
            plt.plot([data_x[i]], [data_y[i]], marker=f'x', markersize=7, color="magenta")
        plt.text(data_x[trail[5]] + 0.05, data_y[trail[5]] - 0.05,
                 f'{0.5:4.2f}\nsensitivity: {data_y[trail[5]]:4.3f}\nsensitivity: {1 - data_x[trail[5]]:4.3f}',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
    else:
        plt.plot([redDot[0]], [redDot[1]], marker='o', markersize=15, color="red")
        if redDotLbl is not None:
            plt.text(redDot[0] + 0.05, redDot[1] - 0.1, redDotLbl, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

    if aur_value is not None:
        plt.text(0.5, 0.5, aur_value, horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

    if show:
        plt.show()
    plt.close()
    return fig


def gen_fig_simple(data_i: np.ndarray, data_o: np.ndarray):
    assert data_i.shape[0] == data_o.shape[0]
    assert data_i.shape[1] == data_o.shape[1]

    plots = data_i.shape[0]
    matplotlib.use('Agg')
    fig, axs = plt.subplots(plots, figsize=(12, 4 * plots))
    xx = range(data_i.shape[1])

    for fid in range(plots):
        mxv = max([np.max(np.abs(data_i[fid, :])), np.max(np.abs(data_o[fid, :]))]) * 1.1
        axs[fid].axhline(0, color='black')
        axs[fid].axvline(0, color='black')
        axs[fid].plot(xx, data_i[fid, :], color='black', linewidth=3.0)
        axs[fid].plot(xx, data_o[fid, :], color='red', linewidth=3.0, alpha=0.5)
        axs[fid].set_xlim((min(xx), max(xx)))
        axs[fid].set_ylim((-mxv, mxv))
    plt.close()

    return fig


def gen_fig(data_x: np.ndarray, data_y: np.ndarray, xrange, yrange, lbl_x, lbl_y,
            figsize=(8, 8), fontsize=18, color='black', linewidth=1.0, show=False, closeFig=True) -> matplotlib.figure.Figure:

    assert data_x.ndim == 1
    assert data_y.ndim == 1
    assert data_x.shape[0] == data_y.shape[0]

    if not show:
        matplotlib.use('Agg')
    fig: matplotlib.figure.Figure = plt.figure(num=0, figsize=figsize)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.plot(data_x, data_y, color=color, linewidth=linewidth)
    plt.xlabel(lbl_x, fontsize=fontsize)
    plt.ylabel(lbl_y, fontsize=fontsize)
    plt.xlim(xrange)
    plt.ylim(yrange)
    if show:
        plt.show()
    if closeFig:
        plt.close()
    return fig


def gen_figs(data_x: np.ndarray, data_y: np.ndarray, xrange, yrange, lbl_x, lbl_y, title, threshold,
             figsize=(8, 8), fontsize=18, colors=None, linewidths=None, labels=None, show=False) -> matplotlib.figure.Figure:

    assert data_x.ndim == 1
    assert data_y.ndim == 2
    assert data_x.shape[0] == data_y.shape[1]

    if colors is None:
        colors = []
        for j in range(data_y.shape[0]):
            colors.append('black')

    if linewidths is None:
        linewidths = []
        for j in range(data_y.shape[0]):
            linewidths.append(1.0)

    if labels is None:
        labels = []
        for j in range(data_y.shape[0]):
            labels.append(f'lbl_{j}')

    thr_line = np.ones(data_x.shape, dtype=data_x.dtype)
    thr_line = thr_line * threshold

    # if not show:
    #     matplotlib.use('Agg')
    fig = plt.figure(figsize=figsize)
    plt.axhline(0, color='black')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    for j in range(data_y.shape[0]):
        plt.plot(data_x, data_y[j, :], color=colors[j], linewidth=linewidths[j], label=labels[j])
    plt.plot(data_x, thr_line, color='magenta', linewidth=1, linestyle='--')
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(lbl_x, fontsize=fontsize)
    plt.ylabel(lbl_y, fontsize=fontsize)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(range(xrange[0], xrange[1] + 1, 1000))
    yticks = [x / 10 for x in range(0, 11, 1)]
    plt.yticks(yticks)
    plt.xticks(rotation=75)
    plt.grid(color='gray', linestyle=':', linewidth=1)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()
    return fig

