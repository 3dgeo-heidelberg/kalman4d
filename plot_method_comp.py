import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

import seaborn as sns
import tqdm
import datetime

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale = 1.3)

infiles = [
    r"kalman-results\interpolated.npy",
    r"kalman-results\kromer_w48.npy",
    r"kalman-results\kromer_w96.npy",
    r"kalman-results\vals2021_kalman_q0.02.npy",
]

labels = [
    'Linear interp.',
    'Temp. median$^{48}$',
    'Temp. median$^{96}$',
    'Kalman$_1^{0.02 m/day}$',
]

kwargs = [
    {"linewidth":0.3, "color":'grey'},
    {},
    {},
    {},
]

cps = {159690: "Erosion rill",
       377272: "Avalanche area",
       181882: "Rock face",
       138500: "Deep erosion rill",
       158929: "Boulder",
       }

DATA = [np.load(infile)[np.array([int(k) for k in cps.keys()]), :]
        for infile in tqdm.tqdm(infiles)]

ref_epoch = 1629226820.0
interval = 1 * 60 * 60
end = 1850
dt = [datetime.datetime.fromtimestamp(ref_epoch + interval * i) for i in range(len(DATA[0][0][3:end]))]

gs = grid_spec.GridSpec(len(cps),1)
fig = plt.figure(figsize=(10,2*len(cps)), dpi=400)
axs = []
for cpidx, (cp, label) in enumerate(cps.items()):
    ax_a = fig.add_subplot(gs[cpidx, 0])
    for data, lab, kwar in zip(DATA, labels, kwargs):
        ax_a.plot(dt, data[cpidx, 3:end], label=lab, **kwar)
    ax_a.set_ylabel(f'Change value [m]')
    ax_a.set_yticks(np.arange(round(ax_a.get_ylim()[0], 2), ax_a.get_ylim()[1], 0.01), minor=True)
    if cpidx < (len(cps)-1):
        ax_a.set_xticklabels([])
    ax_a.grid(which="minor",alpha=0.1)
    axs.append(ax_a)

    ax_a.annotate(str(chr(97+cpidx)) + f") {label}", (0.0,1-(0.089+0.18*cpidx)),
                  xycoords='figure fraction',
                  fontsize='x-large')
axs[0].legend(fontsize='x-small', loc='lower left')
sns.despine()
plt.suptitle("Comparison of multitemporal change quantification")
fig.align_ylabels(axs)
plt.tight_layout(h_pad=2.5)
plt.savefig(r'C:\Users\Lukas\Documents\Papers\Kalman_ESurfD\abb\timeline_method_comp.pdf')
fig.show()