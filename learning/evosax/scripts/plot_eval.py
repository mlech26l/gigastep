import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns
sns.set()
sns.set_style("ticks")

import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["legend.handlelength"] = 1.0
matplotlib.rcParams["legend.columnspacing"] = 1.5
# plt.rcParams["axes.axisbelow"] = False

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


### Eval vs simple pool
# winrate_28ado = [23.8, 23.8, 27.5, 27.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5]
generation = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
des_23_ego_win = [13.8, 26.2, 26.2, 31.2, 38.8, 38.8, 61.7, 66.2, 66.2, 66.2, 66.2]
des_23_ego_tie = [0.0, 17.5, 17.5, 47.5, 37.5, 37.5, 6.3, 6.3, 6.3, 6.3, 6.3]
des_23_ego_wpt = [win + tie for (win, tie) in zip(des_23_ego_win, des_23_ego_tie)]
snes_0_ado_win = [13.8, 25.0, 23.8, 23.8, 23.8, 23.8, 23.8, 51.2, 52.5, 70.0, 70.0]
snes_0_ado_tie = [0.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 37.5, 25.0, 5.0, 5.0]
snes_0_ado_wpt = [win + tie for (win, tie) in zip(snes_0_ado_win, snes_0_ado_tie)]
openes_4_ego_win = [13.8, 37.5, 33.8, 52.5, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
openes_4_ego_tie = [0.0, 3.8, 55.0, 35.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
openes_4_ego_wpt = [win + tie for (win, tie) in zip(openes_4_ego_win, openes_4_ego_tie)]

# openes_4_ego_win = [

colors = sns.color_palette("husl", 3)
text_size = 16
label_size = 14

with sns.axes_style("darkgrid"):
    # figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 5))
    # figure = plt.figure(figsize=(15, 4))
    # gridspec = figure.add_gridspec(1, 6)
    figure = plt.figure(figsize=(8, 4), constrained_layout = True)
    gridspec = figure.add_gridspec(1, 1)

    alpha = 0.1
    linewidth = 3

    axis = figure.add_subplot(gridspec[0, 0])
    # axis.fill_between(steps, exp1_winrate_mean - exp1_winrate_stdv, exp1_winrate_mean + exp1_winrate_stdv, color=list(colors[0] + (alpha,)))
    axis.plot(generation, des_23_ego_win, linestyle='-', color='b', linewidth=linewidth, label=r'DES')
    axis.plot(generation, des_23_ego_wpt, linestyle=':', color='b', linewidth=linewidth)
    axis.fill_between(generation, des_23_ego_win, des_23_ego_wpt, color='b', alpha=0.1)
    axis.plot(generation, snes_0_ado_win, linestyle='-', color='g', linewidth=linewidth, label=r'SNES')
    axis.plot(generation, snes_0_ado_wpt, linestyle=':', color='g', linewidth=linewidth)
    axis.fill_between(generation, snes_0_ado_win, snes_0_ado_wpt, color='g', alpha=0.1)
    axis.plot(generation, openes_4_ego_win, linestyle='-', color='r', linewidth=linewidth, label=r'OpenES')
    axis.plot(generation, openes_4_ego_wpt, linestyle=':', color='r', linewidth=linewidth)
    axis.fill_between(generation, openes_4_ego_win, openes_4_ego_wpt, color='r', alpha=0.1)

    axis.set_xlim([0, 2000])
    axis.set_ylim([0.0, 100.0])
    axis.set_xlabel(r"Generation", fontsize=text_size)
    axis.set_ylabel(r"Win rate [%]", fontsize=text_size)
    axis.set_title(r"Win + Tie Rate over Policy Pool" + '\n', fontsize=text_size)

    axis.tick_params(axis='both', which='major', labelsize=label_size)

    legend_size = 16
    # offset = (0.5, -0.4)
    offset = (0.5, 0.0)
    axis.legend(*map(reversed, axis.get_legend_handles_labels()), loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)

# PNG
fig_name = './learning/evosax/scripts/figures/sim_eval.png'
figure.savefig(fig_name, bbox_inches='tight', format='png')