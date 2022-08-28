import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from pathlib import Path
import seaborn as sns
import tqdm
import datetime
import cmcrameri.cm as cmc

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale = 1.3)

infiles = [
    Path(r"kalman_results\q0.0005_X.las"),
    Path(r"kalman_results\q0.02_XV.las"),
    Path(r"kalman_results\q0.002_XVA.las"),
    ]
infiles = infiles[::-1]
DATA = [np.load(str(infile).replace(".las", ".npy"))
        for infile in tqdm.tqdm(infiles)]
DATA_unc = [np.load(str(infile).replace(".las", "_unc.npy"))
        for infile in tqdm.tqdm(infiles)]

labels = [
    'Kalman$_{0}^{0.0005 m}$',
    'Kalman$_{1}^{0.02 m/day}$',
    'Kalman$_{2}^{0.002 m/day^2}$',
][::-1]

kwargs = [
    {},
    {},
    {},
]

cps = {0: "Erosion rill",
       1: "Avalanche area",
       3: "Rock face",
       4: "Deep erosion rill",
       5: "Boulder",
       }

ref_epoch = 1629226820.0
interval = 1 * 60 * 60

colors = cmc.roma(np.linspace(0, 1, len(DATA)+1))[[1,0,3]]
end = 1850
dt = [datetime.datetime.fromtimestamp(ref_epoch + interval * i) for i in range(len(DATA[0][0][3:end]))]

gs = grid_spec.GridSpec(len(cps),1)
fig = plt.figure(figsize=(10,2*len(cps)), dpi=400)
axs = []
for cpidx, (cp, label) in enumerate(cps.items()):
    ax_a = fig.add_subplot(gs[cpidx, 0])
    for dix, (data, dat_unc, lab, kwar) in enumerate(zip(DATA, DATA_unc, labels, kwargs)):
        ax_a.plot(dt, data[cp, 3:end], label=lab, c=colors[dix], alpha=0.6, **kwar)
        ax_a.plot(dt, dat_unc[cp][:len(dt)], linewidth=0.3, c=colors[dix], alpha=0.6, linestyle='dotted')
        ax_a.plot(dt, -1 * dat_unc[cp][:len(dt)], linewidth=0.3, c=colors[dix], alpha=0.6, linestyle='dotted')
    ax_a.set_ylabel(f'Change value [m]')
    ax_a.set_yticks(np.arange(round(ax_a.get_ylim()[0], 2), ax_a.get_ylim()[1], 0.01), minor=True)
    if cpidx < (len(cps)-1):
        ax_a.set_xticklabels([])
    ax_a.grid(which="minor",alpha=0.1)
    axs.append(ax_a)
    ax_a.annotate(str(chr(97+cpidx)) + f") {label}", (0.0,1-(0.088+0.181*cpidx)),
                  xycoords='figure fraction',
                  fontsize='x-large')

handles, labels = axs[0].get_legend_handles_labels()
order = [2,1,0]
legend = axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left', bbox_to_anchor=(0.0, -0.01), fontsize='x-small')

sns.despine()
plt.suptitle("Comparison of Kalman models of different order")
fig.align_ylabels(axs)
plt.tight_layout(h_pad=2.5)
plt.savefig(r'kalman_order.pdf')

fig.show()