from pathlib import Path
import tqdm
import numpy as np

phi = -120 * np.pi/180.
tfM = np.array([
    [np.cos(phi), 0, np.sin(phi)],
    [0, 1, 0],
    [-np.sin(phi), 0, np.cos(phi)],
])

infiles = sorted(list(Path(r"synth_kalman").glob(f"*.npy")))[::-1]
infiles = [inf for inf in infiles if  'unc' not in str(inf)]
DATA = [np.load(str(infile))
        for infile in tqdm.tqdm(infiles)]

for modelid, model in enumerate(DATA):
    sum_sq_res = 0
    for cpidx in range(len(model)):
        curr_pt = model[cpidx][:3]
        curr_pt = tfM.T @ curr_pt
        curr_ts = model[cpidx][3:]
        displ = 0.05 * (curr_pt[1] - 50) / (50) * (np.sin(np.linspace(-np.pi / 2, np.pi / 2, 40)[1:]) + 1) / 2
        res = displ - curr_ts
        res = np.sum(np.power(res, 2))
        if not np.isnan(res):
            sum_sq_res += res

    print(infiles[modelid], sum_sq_res)