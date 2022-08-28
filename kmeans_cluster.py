import time

import pandas as pd
from sklearn.cluster import KMeans
from scipy import spatial
import tqdm
import numpy as np
from datetime import datetime, timedelta

import laspy

def limited_cumsum(array, lb=0):
    result = np.zeros(array.size)
    result[0] = array[0]
    for k in range(1, array.size):
        result[k] = max(lb, result[k-1]+array[k])
    return result

# from https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
def datetime2year(dt):
    year_part = dt - datetime(year=dt.year, month=1, day=1)
    year_length = (
        datetime(year=dt.year + 1, month=1, day=1)
        - datetime(year=dt.year, month=1, day=1)
    )
    return dt.year + year_part / year_length

def read_from_las(path, useevery=1):
    inFile = laspy.read(path)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()[::useevery, :]
    try:
        n0 = inFile.points.array["normalx"][::useevery]
        n1 = inFile.points.array["normaly"][::useevery]
        n2 = inFile.points.array["normalz"][::useevery]
        normals = np.stack((n0,n1,n2)).T
    except:
        normals = None

    las_fields = sorted(list(inFile.points.point_format.extra_dimension_names))

    date_count = len([f for f in las_fields if f.startswith("val_")])
    vals = np.full((coords.shape[0], date_count), np.nan)
    uncs = np.full((coords.shape[0], date_count), np.nan)
    dates = []
    dateidx = 0
    for las_field in las_fields:
        if las_field.startswith("val_"):
            field_date = las_field[4:]
            date = datetime.fromtimestamp(int(float(field_date)))
            dates.append(date)
            vals[:, dateidx] = inFile.points.array["val_%s" % field_date][::useevery]
            uncs[:, dateidx] = inFile.points.array["unc_%s" % field_date][::useevery]
            dateidx += 1

    return coords, normals, vals, uncs, dates


def write_to_las(path, points, attrdict):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")

    for attrname in attrdict:
        try:
            dt = attrdict[attrname].dtype
            if dt == bool:
                attrdict[attrname] = attrdict[attrname].astype(int)
                dt = attrdict[attrname].dtype
            header.add_extra_dim(laspy.ExtraBytesParams(name=attrname.lower(), type=dt, description=attrname.lower()))
        except Exception as e:
            print("Failed adding dimension %s: %s" % (attrname.lower(), e))
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.00025, 0.00025, 0.00025])

    # 2. Create a Las
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    for attrname in attrdict:
        setattr(las, attrname.lower(), attrdict[attrname])
    las.write(path)


def main(infile, npyfile, ks, outFile, useevery=1):
    coords, normals, vals, uncs, dates = read_from_las(infile, useevery)

    ## override feature space from numpy file
    f = np.load(npyfile)
    # filter out points removed from the las
    val_idx = []

    c_tree = spatial.cKDTree(f[:, :3])
    d, i = c_tree.query(coords, 1, distance_upper_bound=1e-5)

    vals = f[i, 3:]

    max_displ = np.nanmax(np.abs(vals), axis=1)
    within_limits = max_displ < 3
    vals = vals[within_limits, :]
    coords = coords[within_limits, :]

    labels = np.full((coords.shape[0], len(ks)), np.nan)
    for kidx, k in enumerate(tqdm.tqdm(ks)):
        X = vals
        nan_indicator = np.logical_not(np.isnan(np.sum(X, axis=1)))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X[nan_indicator, :])
        this_labels =  kmeans.labels_
        ranks = np.argsort(np.argsort(np.bincount(this_labels)))
        this_labels = ranks[this_labels]
        labels[nan_indicator, kidx] = this_labels
        centers = kmeans.cluster_centers_[ranks, ...]

        import matplotlib.pyplot as plt
        import cmcrameri.cm as cmc
        plt.figure(figsize=(10, 5))
        colors = cmc.bukavu(np.linspace(0, 1, k))
        for line, color in zip(range(centers.shape[0]), colors):
            plt.plot(centers[line, :], color=color, label=f'Cluster {line}')
        plt.legend()
        plt.savefig(outFile.replace('.las', f'_k{k}.svg'))

        for idx in np.unique(this_labels):
            this_points = X[nan_indicator, :][this_labels == idx]
            within_var = np.mean(np.var(this_points, axis=1))
            print(f"k={k}, Cluster {idx}, mean var={within_var:.5f}")

    out_attrs = {}
    for kidx, k in enumerate(ks):
        out_attrs["label_%d" % k] = labels[:, kidx]

    write_to_las(outFile, coords, out_attrs)

if __name__ == '__main__':
    infile = [
        r"corepoints.las",
              ]
    npyfile = [
        r"kalman_results\q0.02.npy",
              ]

    outfile = [
        r"cluster_results\q0.02.las",
    ]
    ks = [4, 8, 10, 12]
    ks = ks [::-1]
    for inf, npyf, outf in zip(infile, npyfile, outfile):
        main(inf, npyfile=npyf, ks=ks, outFile=outf, useevery=1)