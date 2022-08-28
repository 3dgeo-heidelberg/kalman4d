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

ref_epoch = 1629226820.0
interval = 1 * 60 * 60
end = 1850
gs = grid_spec.GridSpec(3, 1)
fig = plt.figure(figsize=(10, 9), dpi=400)
axs = []
units = ['m', 'm/day', 'm/day$^2$']
select = [3, 1, 2]

for order, model in enumerate(['X', 'XV', 'XVA']):
    infiles = sorted(list(Path(r"kalman_results").glob(f"*_{model}.las")))[::-1]
    DATA = [np.load(str(infile).replace(".las", ".npy"))
            for infile in tqdm.tqdm(infiles)]
    DATA_unc = [np.load(str(infile).replace(".las", "_unc.npy"))
            for infile in tqdm.tqdm(infiles)]


    colors = cmc.batlow(np.linspace(0, 1, len(DATA)))
    dt = [datetime.datetime.fromtimestamp(ref_epoch + interval * i) for i in range(len(DATA[0][0][3:end]))]

    for cpidx in [0]:
        ax_a = fig.add_subplot(gs[order, 0])
        for dati, (dat, dat_unc, color) in enumerate(zip(DATA, DATA_unc, colors)):
            ax_a.plot(dt, dat[cpidx][3:end], label=infiles[dati].name.split("_")[0].replace("q", "$\sigma$=") + f" {units[order]}", c=color)
            ax_a.plot(dt, dat_unc[cpidx][:len(dt)], linewidth=0.3, c=color, alpha=0.4, linestyle='dotted')
            ax_a.plot(dt, -1 * dat_unc[cpidx][:len(dt)], linewidth=0.3, c=color, alpha=0.4, linestyle='dotted')
        ax_a.set_ylabel(f'Change value [m]\nLevel of Detection [m]\n')
        ax_a.set_ylim(-0.15, 0.05)
        axs.append(ax_a)
        legend = ax_a.legend(loc='lower left')
        for tix, text in enumerate(legend.get_texts()):
            if tix == select[order]:
                text.set_weight('bold')
    ax_a.annotate(str(chr(97+order)) + f") Order {order} model ({','.join(model.lower())})", (0.0,1-(0.09+0.306*order)),
                  xycoords='figure fraction',
                  fontsize='x-large')
sns.despine()
plt.suptitle("Comparison of Kalman filter models")
plt.tight_layout(h_pad=2.5)
plt.savefig(r'kalman_model_comparison.pdf')
fig.show()