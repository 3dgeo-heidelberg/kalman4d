import glob
import time

from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters, EfficientFCParameters
import pandas as pd
import tqdm
from filterpy.kalman import KalmanFilter as kf
from filterpy.common import Q_discrete_white_noise
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as sstats
from scipy import spatial
import matplotlib.pyplot as plt

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

def read_from_las(path, useevery=1, start=0, end=-1):
    from laspy.file import File as LasFile
    inFile = LasFile(path, mode='r')
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[start:end:useevery, :]
    try:
        normals = getattr(inFile, 'normals', None)
    except:
        normals = None
    if normals is None:
        try:
            n0 = inFile.points["point"]["NormalX"]
            n1 = inFile.points["point"]["NormalY"]
            n2 = inFile.points["point"]["NormalZ"]
            normals = np.stack((n0,n1,n2)).T
        except:
            normals = None
    normals = normals[start:end:useevery, :]
    las_fields = sorted(list(inFile.points["point"].dtype.fields))
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
            vals[:, dateidx] = inFile.points["point"]["val_%s" % field_date][start:end:useevery]
            uncs[:, dateidx] = inFile.points["point"]["unc_%s" % field_date][start:end:useevery]
            dateidx += 1

    return coords, normals, vals, uncs, dates

las_data_types = {
    int: 4,
    float: 10,  # or 10 for double prec
    np.dtype(np.float64): 10,
    np.dtype(np.int32): 4
}

def write_to_las(path, points, attrdict):
    LasFile = None  # full reset
    LasHeader = None
    from laspy.file import File as LasFile
    from laspy.header import Header as LasHeader
    hdr = LasHeader(version_major=1, version_minor=4)
    outFile = LasFile(path, mode="w", header=hdr)
    for attrname in attrdict:
        if attrname == "normals":
            outFile.define_new_dimension(name=attrname, data_type=30, description=attrname)
            continue
        try:
            dt = 9 # default data type
            if attrdict[attrname].dtype in las_data_types:
                dt = las_data_types[attrdict[attrname].dtype]
            else:
                print("Unknown data type: '%s', attemping saving as float." % attrdict[attrname].dtype)
            outFile.define_new_dimension(name=attrname.lower(), data_type=dt, description=attrname.lower())
        except Exception as e:
            print("Failed adding dimension %s: %s" % (attrname.lower(), e))
    xmin, ymin, zmin = np.min(points, axis=0)
    outFile.header.offset = [xmin, ymin, zmin]
    outFile.header.scale = [0.001, 0.001, 0.001]
    outFile.x = points[:, 0]
    outFile.y = points[:, 1]
    outFile.z = points[:, 2]
    for attrname in attrdict:
        setattr(outFile, attrname.lower(), attrdict[attrname])
    outFile.close()
    outFile = None

def proc_cp(p_changes, p_unc, p_dates, corepoint_id, p_val, mkplot, Q_vals, exportFilter, exportFreshD):

    SqSumRes = np.full((len(Q_vals),), np.nan)
    if all(np.isnan(p_changes)):
        print("Skipping point #%d" % corepoint_id)
        return (SqSumRes, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                [[np.nan],[np.nan], [np.nan]], {})

    p_changes = [item if not np.isnan(item) else None for item in
                 p_changes]  # "nan" breaks processing, None means no observation here
    p_unc = [item if not np.isnan(item) else None for item in p_unc]

    ts_raw = np.array([datetime2year(date) * 365 for date in p_dates])  # time series in days
    ts_raw -= ts_raw[0]

    ts_full = np.arange(ts_raw[0], ts_raw[-1], 1. / 240)  # define here range and interval

    ts_nones = np.array([None] * len(ts_full))
    ts = np.concatenate((ts_raw, ts_full))
    order = np.argsort(ts)
    interp_idx = order>len(ts_raw)-1  # indices with even time spacing

    meas_order = [item - 1 for item in order[1:]]
    ts = ts[order]
    p_unc_raw = p_unc
    p_unc = np.concatenate((p_unc_raw, ts_nones))
    p_unc = p_unc[meas_order]
    dts = np.gradient(ts)

    zs = np.concatenate((p_changes, ts_nones))[meas_order]
    Fs = [np.array([[1., dt, 1/2*dt**2],
                    [0., 1., dt],
                    [0., 0., 1.]]) for dt in dts]
    Rs = [np.array(u) for u in p_unc]

    for q_id, q in enumerate(Q_vals):

        my_filter = kf(dim_x=3, dim_z=1)
        my_filter.x = np.array([[0.],
                                [0.],
                                [0.]])  # initial state (location and velocity)

        my_filter.F = np.array([[1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]])  # state transition matrix

        my_filter.H = np.array([[1., 0., 0.]])  # Measurement function
        my_filter.P = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # covariance matrix (initital state)

        Qs = [Q_discrete_white_noise(dim=3, dt=dt, var=(q ** 2), block_size=1) for dt in dts]

        # Actual Kalman filter step
        (mu, cov, _, _) = my_filter.batch_filter(zs, Fs=Fs, Rs=Rs, Qs=Qs)
        # RTS smoother step
        (xs, Ps, Ks, Pp) = my_filter.rts_smoother(mu, cov, Fs=Fs, Qs=Qs)

        res = np.array([zi - xi for (zi, xi) in zip(zs, xs) if (zi is not None and xi is not None)])
        res = res[:, 0]  # only position residuals
        norm_res = res ** 2
        SqSumRes[q_id] = np.nansum(norm_res)

        if q_id + 1 == len(Q_vals) or True:  # only for the last q
            smoother_unc = np.array([(p_val * elem[0, 0]) ** (0.5) for elem in Ps])
            vel_unc = np.array([(p_val * elem[1, 1]) ** (0.5) for elem in Ps])
            smoother_val = np.array([elem[0][0] for elem in xs])

            smoother_vel = [elem[1][0] for elem in xs]
            smoother_acc = [elem[2][0] for elem in xs]

            sig_change = np.abs(smoother_unc) <= np.abs(smoother_val)
            sig_change[0] = False  # because loD == 0 for the first epoch
            sign_date = ts[1:][sig_change]
            sign_date = sign_date[0] if len(sign_date) > 0 else np.nan

            time_of_max_v = ts[1:][np.argmax(smoother_vel)]
            time_of_max_a = ts[1:][np.argmax(smoother_acc)]
            time_of_min_v = ts[1:][np.argmin(smoother_vel)]
            time_of_min_a = ts[1:][np.argmin(smoother_acc)]

            if exportFreshD:
                settings = EfficientFCParameters()
                dfs = pd.DataFrame([np.ones(smoother_val.shape), smoother_val, smoother_vel, smoother_acc],
                                   ["id", "pos", "vel", "acc"]).T
                try:
                    fresh = extract_features(dfs, column_id="id", n_jobs=0, disable_progressbar=True, default_fc_parameters=settings)
                except Exception as e:
                    print("ERROR IN tsfresh:", e)
                freshD = fresh.to_dict('records')[0]
            else:
                freshD = {}

            filterVals = [ts[interp_idx], smoother_val[interp_idx[1:]], smoother_unc[interp_idx[1:]]]

            if corepoint_id in mkplot:
                print("Plotting point #%d" % corepoint_id)
                plt.figure(figsize=(12, 5))
                plt.plot(ts[1:], zs, "rx", label="Measurement")
                plt.plot(ts[1:], mu[:, 0], "b--", label="Kalman filter state")
                plt.plot(ts[1:], xs[:, 0], "g--", label="Kalman smoother state")
                plt.plot(ts[1:], xs[:, 1], "r--", label="Kalman smoother velocity")
                plt.plot(ts[1:], xs[:, 2], "y--", label="Kalman smoother acceleration")
                plt.fill_between(ts[1:],
                                 smoother_unc, -1 * smoother_unc, color="g", alpha=0.4, label="Uncertainty smoother")
                plt.fill_between(ts[1:],
                                 vel_unc, -1 * vel_unc, color="grey", alpha=0.4, label="Uncertainty smoother vel")
                plt.plot(ts_raw[1:], [(p_val * elem) ** (0.5) if elem is not None else None for elem in p_unc_raw],
                         "r.", label="Uncertainty single")
                plt.plot(ts_raw[1:], [-1 * (p_val * elem) ** (0.5) if elem is not None else None for elem in p_unc_raw],
                         "r.")
                plt.xlabel("Epoch")
                plt.ylabel("Change [m]")
                plt.ylim([-0.1, 0.125])
                plt.title("Change detection ($\sigma$ = %f)" % q)
                plt.tight_layout()
                plt.legend()
                plt.show()
                plt.savefig('plot_%s_q%s.svg' % (corepoint_id, q))

    return (SqSumRes, sign_date,
            smoother_val[-1], smoother_unc[-1],
            np.sum(np.abs(smoother_acc)), np.mean(np.abs(smoother_vel)),
            np.max(smoother_acc), np.max(smoother_vel),
            np.mean(smoother_acc), np.mean(smoother_vel),
            smoother_acc[-1], smoother_vel[-1], time_of_max_v, time_of_max_a,
            np.min(smoother_acc), np.min(smoother_vel),time_of_min_v, time_of_min_a,
            np.max(smoother_val), np.min(smoother_val), smoother_acc[np.argmax(smoother_val)], smoother_acc[np.argmin(smoother_val)],
            filterVals, freshD
            )

def main(infile, ref_epoch, Q_vals, outFile, useevery=1, mkplot=None, exportFilter=False, exportFreshD=False):
    if mkplot is None:
        mkplot = list()
    # read first file:
    print("Loading file %s" % infile[0])
    coords, normals, changes, unc, p_dates = read_from_las(infile[0], useevery=useevery)
    p_kdtree = spatial.KDTree(coords, leafsize=16)

    for file in infile[1:]:
        print("Loading file %s" % file)
        coords_i, _, changes_i, unc_i, p_dates_i = read_from_las(file, useevery=useevery)
        dists, ids = p_kdtree.query(coords_i, k=1, distance_upper_bound=1, workers=-1)
        inverse_ids = sorted(np.arange(0, len(coords_i)), key=lambda x: ids[x])
        changes = np.concatenate((changes, changes_i[inverse_ids, :]), axis=1)
        unc = np.concatenate((unc, unc_i[inverse_ids, :]), axis=1)
        p_dates = p_dates + p_dates_i

    p_dates = [datetime.fromtimestamp(int(float(ref_epoch)))] + p_dates

    SqSumResS = np.full((coords.shape[0], len(Q_vals)), np.nan)
    SignDates = np.full((coords.shape[0],), np.nan)
    FinalChange = np.full((coords.shape[0],), np.nan)
    FinalLoD = np.full((coords.shape[0],), np.nan)
    totalCurv = np.full((coords.shape[0],), np.nan)
    meanSlope = np.full((coords.shape[0],), np.nan)
    maxAcc = np.full((coords.shape[0],), np.nan)
    maxVel = np.full((coords.shape[0],), np.nan)
    meanAcc = np.full((coords.shape[0],), np.nan)
    meanVel = np.full((coords.shape[0],), np.nan)
    lastAcc = np.full((coords.shape[0],), np.nan)
    lastVel = np.full((coords.shape[0],), np.nan)
    tMaxVel = np.full((coords.shape[0],), np.nan)
    tMaxAcc = np.full((coords.shape[0],), np.nan)
    minVel = np.full((coords.shape[0],), np.nan)
    minAcc = np.full((coords.shape[0],), np.nan)
    tMinVel = np.full((coords.shape[0],), np.nan)
    tMinAcc = np.full((coords.shape[0],), np.nan)
    smoothMax = np.full((coords.shape[0],), np.nan)
    smoothMin = np.full((coords.shape[0],), np.nan)
    accAtMax = np.full((coords.shape[0],), np.nan)
    accAtMin = np.full((coords.shape[0],), np.nan)
    filters = [None] * coords.shape[0]
    freshD = [None] * coords.shape[0]

    p = 1  # three dimensional, projected to 1D
    p_val = sstats.chi2.ppf(.95, p)

    pool = multiprocessing.Pool(5)
    ts = time.time()
    results = pool.starmap(proc_cp, tqdm.tqdm([[changes[corepoint_id, :], unc[corepoint_id, :],
                                                p_dates, corepoint_id, p_val, mkplot, Q_vals, exportFilter, exportFreshD] for corepoint_id in range(coords.shape[0])]),
                           chunksize=1)
    print("Processing took %s s" % ((time.time()-ts)))
    for ptid, ptvals in enumerate(results):
        SqSumResS[ptid, :] = ptvals[0]
        SignDates[ptid] = ptvals[1]
        FinalChange[ptid] = ptvals[2]
        FinalLoD[ptid] = ptvals[3]
        totalCurv[ptid] = ptvals[4]
        meanSlope[ptid] = ptvals[5]
        maxAcc[ptid] = ptvals[6]
        maxVel[ptid] = ptvals[7]
        meanAcc[ptid] = ptvals[8]
        meanVel[ptid] = ptvals[9]
        lastAcc[ptid] = ptvals[10]
        lastVel[ptid] = ptvals[11]
        tMaxVel[ptid] = ptvals[12]
        tMaxAcc[ptid] = ptvals[13]

        minVel[ptid] = ptvals[14]
        minAcc[ptid] = ptvals[15]
        tMinVel[ptid] = ptvals[16]
        tMinAcc[ptid] = ptvals[17]

        smoothMax[ptid] = ptvals[18]
        smoothMin[ptid] = ptvals[19]
        accAtMax[ptid] = ptvals[20]
        accAtMin[ptid] = ptvals[21]

        filters[ptid] = ptvals[22]
        freshD[ptid] = ptvals[23]



    out_attrs = {
        'normals': normals
    }
    for qid, q in enumerate(Q_vals):
        out_attrs["%.3f" % q] = SqSumResS[:, qid]

    for cpidx in range(coords.shape[0]):
        plt.plot(Q_vals, SqSumResS[cpidx, :])

    if exportFilter:
        all_dates = []
        for dates, vals, uncs in filters:
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

        out_attrs['anySigChange'] = np.logical_not(np.isnan(anySigChange))
        out_attrs['anySigChangeMagn'] = anySigChange

    if exportFreshD:
        fresh_elements = np.full((coords.shape[0], len(freshD[0])+3), np.nan)
        fresh_elements[:, :3] = coords
        for pix, freshDict in enumerate(freshD):
            for kix, (key, val) in enumerate(freshDict.items()):
                fresh_elements[pix][kix+3] = val
        np.save(outFile.replace(".las", "_fresh.npy"), fresh_elements)


    min_q_idx = np.argmin(SqSumResS, axis=1)
    out_attrs['min_res_q_magn'] = SqSumResS[np.arange(len(min_q_idx)), min_q_idx]
    out_attrs['date'] = SignDates
    out_attrs['change'] = FinalChange
    out_attrs['last_lod'] = FinalLoD
    out_attrs['totalCurv'] = totalCurv
    out_attrs['meanAbsSlope'] = meanSlope
    out_attrs['maxVel'] = maxVel
    out_attrs['minVel'] = minVel
    out_attrs['maxAcc'] = maxAcc
    out_attrs['minAcc'] = minAcc
    out_attrs['meanVel'] = meanVel
    out_attrs['meanAcc'] = meanAcc
    out_attrs['lastVel'] = lastVel
    out_attrs['lastAcc'] = lastAcc
    out_attrs['tMaxVel'] = tMaxVel
    out_attrs['tMinVel'] = tMinVel
    out_attrs['tMaxAcc'] = tMaxAcc
    out_attrs['tMinAcc'] = tMinAcc

    out_attrs['maxDispl'] = smoothMax
    out_attrs['minDispl'] = smoothMin
    out_attrs['accAtMax'] = accAtMax
    out_attrs['accAtMin'] = accAtMin

    write_to_las(outFile, coords, out_attrs)

if __name__ == '__main__':
    infile = glob.glob(r"change_full_info_*.las")
    ref_epoch = 1597874411.0   # unix timestamp of the zero epoch (VALS 2020)
    Q_vals = [0.05]
    print("Using the following [time] uncertainties: (m/day)", Q_vals)
    outfile = r"kalman_25cm_Proc_q0_05.las"
    main(infile, ref_epoch, Q_vals, outfile, useevery=1, mkplot=[], exportFilter=True, exportFreshD=False)