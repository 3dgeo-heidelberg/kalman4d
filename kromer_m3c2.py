import glob
import time

import pandas as pd
import scipy.ndimage
import tqdm
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as sstats
from scipy import spatial
import matplotlib.pyplot as plt
import laspy

import multiprocessing

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

def proc_cps(Lp_changes, Lp_unc, p_dates, Lcorepoint_id, filter_size):
    chunklist = []
    for p_changes, p_unc, corepoint_id in zip(Lp_changes, Lp_unc, Lcorepoint_id):
        chunklist.append(
            proc_cp(p_changes, p_unc, p_dates, corepoint_id, filter_size)
        )

def proc_cp(p_changes, p_unc, p_dates, corepoint_id, filter_size):

    if all(np.isnan(p_changes)):
        #print("Skipping point #%d" % corepoint_id)
        return (
                [[np.nan],[np.nan],[np.nan]])
    p_changes = np.concatenate(([0.0], p_changes))
    valid_idx = np.logical_not(np.isnan(p_changes))
    p_changes = p_changes[valid_idx]
    ts_raw = np.array([datetime2year(date) * 365 for date in p_dates])[valid_idx]  # time series in days
    ts_raw -= ts_raw[0]
    interval = 1./24  # (evaluation interval in days, e.g. 1/12 = 2 hours)
    f = scipy.interpolate.interp1d(ts_raw, p_changes)
    ts = np.arange(ts_raw[0], ts_raw[-1], interval)
    changes = f(ts)

    moving_median_smooth = scipy.ndimage.median_filter(changes, filter_size, mode='nearest')

    return (
        # [ts, moving_median_smooth, [np.nan] * len(ts)] # -- for temporal median smoothing
        [ts, changes, [np.nan] * len(ts)]  # -- for simple linear interpolation
            )

def main(infile, ref_epoch, outFile, filter_size, useevery=1, mkplot=None, exportFilter=False, exportFreshD=False):
    if mkplot is None:
        mkplot = list()
    # read first file:
    print("Loading file %s" % infile[0])
    coords, normals, changes, unc, p_dates = read_from_las(infile[0], useevery=useevery)
    p_kdtree = spatial.KDTree(coords, leafsize=16, copy_data=True)

    for file in infile[1:]:
        print("Loading file %s" % file)
        coords_i, _, changes_i, unc_i, p_dates_i = read_from_las(file)  # do not pass "useevery" here due to different order
        dists, ids = p_kdtree.query(coords_i, k=1, distance_upper_bound=0.01)
        valid_dists = np.isfinite(dists)
        ids = ids[valid_dists]
        coords_i = coords_i[valid_dists]
        changes_i = changes_i[valid_dists]
        unc_i = unc_i[valid_dists]
        inverse_ids = sorted(np.arange(0, len(coords_i)), key=lambda x: ids[x])

        changes = np.concatenate((changes, changes_i[inverse_ids, :]), axis=1)
        unc = np.concatenate((unc, unc_i[inverse_ids, :]), axis=1)
        assert np.allclose(coords, coords_i[inverse_ids, :])
        p_dates = p_dates + p_dates_i

    valid_dates = np.array(p_dates) < datetime.fromtimestamp(9999999999)
    p_dates = [p_date for idx, p_date in enumerate(p_dates) if valid_dates[idx]]
    changes = changes[:, valid_dates]
    unc = unc[:, valid_dates]


    p_dates = [datetime.fromtimestamp(int(float(ref_epoch)))] + p_dates
    print(p_dates)

    pool = multiprocessing.Pool(15)
    ts = time.time()
    process_corepoints = range(coords.shape[0])
    filters = [None] * len(process_corepoints)

    results = pool.starmap(proc_cp, tqdm.tqdm([[changes[corepoint_id, :], unc[corepoint_id, :],
                                                p_dates, corepoint_id, filter_size] for corepoint_id in process_corepoints]),
                           chunksize=1)
    print("Processing took %s s" % ((time.time()-ts)))

    changes = None
    unc = None

    for ptid, ptvals in enumerate(results):
        filters[ptid] = ptvals

    if exportFilter:
        all_dates = []
        for items in filters:
            if len(items) < 3:
                continue
            dates, vals, uncs = items
            if len(dates) == 1:
                continue
            all_dates += list(dates)
        all_dates = sorted(list(set(all_dates)))
        exportData = np.full((coords.shape[0], len(all_dates)+3), np.nan)
        exportData[:, :3] = coords
        anySigChange = np.full((coords.shape[0], ), np.nan)
        for pix, (dates, vals, uncs) in enumerate(filters):
            if len(dates) == 1:
                continue
            for date, val, unc in zip(dates, vals, uncs):
                dix = all_dates.index(date)
                exportData[pix, 3+dix] = val
                if abs(val) > unc and (np.isnan(anySigChange[pix]) or abs(anySigChange[pix]) < abs(val)):
                    anySigChange[pix] = val
        np.save(outFile.replace(".las", ".npy"), exportData)


if __name__ == '__main__':
    infile = glob.glob(r"m3c2-ep-results\*.las")
    ref_epoch = 1629226820.0   # unix timestamp of the zero epoch (VALS 2021)
    filter_size = 48
    print("Using the following window size for median: (hours)", filter_size)
    outfile = r"kalman-results\linear-interpolated.las"

    main(infile, ref_epoch,  outfile, filter_size=filter_size, useevery=1,
         mkplot=[1],
         exportFilter=True, exportFreshD=False)
