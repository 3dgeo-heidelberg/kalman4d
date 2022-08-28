import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import cmcrameri.cm as cmc
import seaborn as sns
import tqdm

from pathlib import Path

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale = 1.3)

phi = -120 * np.pi/180.
tfM = np.array([
    [np.cos(phi), 0, np.sin(phi)],
    [0, 1, 0],
    [-np.sin(phi), 0, np.cos(phi)],
])

ref_epoch = 1629226820.0
interval = 1 * 60 * 60
end = 1850

gs = grid_spec.GridSpec(3, 1)
fig = plt.figure(figsize=(10, 9), dpi=400)
axs = []

for cpcnt, cpidx in enumerate([30914, 16223, 1501]):
    infiles = [ Path(s) for s in [
            r"synth_kalman\interpolated.npy",
            r"synth_kalman\kromer_w12.npy",
            r"synth_kalman\kromer_w24.npy",
            r"synth_kalman\vals2021_kalman_X_q0.002.npy",
            r"synth_kalman\vals2021_kalman_XV_q0.0005.npy",
            r"synth_kalman\vals2021_kalman_XVA_q5e-05.npy",
        ]
    ]
    names = [
        'Linear interpolation',
        'Temporal median (w=12)',
        'Temporal median (w=24)',
        'Kalman order 0 ($\sigma$=0.002 m)',
        'Kalman order 1 ($\sigma$=0.0005 m/day)',
        'Kalman order 2 ($\sigma$=0.00005 m/day$^2$)',
    ]
    DATA = [np.load(str(infile))
            for infile in tqdm.tqdm(infiles)]


    colors = cmc.batlow(np.linspace(0, 1, len(DATA)))
    dt = np.linspace(0, 40, len(DATA[0][0][3:]))

    ax_a = fig.add_subplot(gs[cpcnt, 0])
    for dati, (dat, color) in enumerate(zip(DATA, colors)):
        ax_a.plot(dt, dat[cpidx][3:end], label=names[dati], c=color)

    curr_pt = DATA[0][cpidx][:3]
    curr_pt = tfM.T @ curr_pt
    displ = 0.05  * (curr_pt[1] - 50) / (50) * (np.sin(np.linspace(-np.pi / 2, np.pi / 2, 40)[1:]) + 1) / 2
    ax_a.plot(dt, displ, 'k--',  label='True change')

    ax_a.set_ylabel(f'Change value [m]')
    axs.append(ax_a)
    ax_a.annotate(str(chr(97+cpcnt)) + ")", (0.0,1-(0.1+0.3*cpcnt)),
                  xycoords='figure fraction',
                  fontsize='x-large')

axs[-1].set_xlabel(f'Time [days]')
legend = axs[0].legend(loc='upper right')
sns.despine()
plt.suptitle("Comparison of different filter models on synthetic data ")
fig.align_ylabels(axs)
plt.tight_layout()
plt.savefig(r'kalman_model_comparison_synth.pdf')
fig.show()