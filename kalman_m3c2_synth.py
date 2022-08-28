import glob
import time

import tqdm
from filterpy.kalman import KalmanFilter as kf
from filterpy.common import Q_discrete_white_noise
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
    leguncs = np.full((coords.shape[0], date_count), np.nan)
    dates = []
    dateidx = 0
    for las_field in sorted(las_fields):
        if las_field.startswith("val_"):
            field_date = las_field[4:]
            date = float(field_date)
            dates.append(date)
            vals[:, dateidx] = inFile.points.array["val_%s" % field_date][::useevery]
            uncs[:, dateidx] = inFile.points.array["unc_%s" % field_date][::useevery]
            leguncs[:, dateidx] = inFile.points.array["legunc_%s" % field_date][::useevery]
            dateidx += 1

    return coords, normals, vals, uncs, dates, leguncs


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

def proc_cps(Lp_changes, Lp_unc, p_dates, Lcorepoint_id, p_val, mkplot, Q_vals, exportFilter, exportFreshD):
    chunklist = []
    for p_changes, p_unc, corepoint_id in zip(Lp_changes, Lp_unc, Lcorepoint_id):
        chunklist.append(
            proc_cp(p_changes, p_unc, p_dates, corepoint_id, p_val, mkplot, Q_vals, exportFilter, exportFreshD)
        )

def proc_cp(p_changes, p_unc, p_dates, corepoint_id, p_val, mkplot, Q_vals, exportFilter, exportFreshD, dim, outfile_q):

    SqSumRes = np.full((len(Q_vals),), np.nan)
    if all(np.isnan(p_changes)):
        #print("Skipping point #%d" % corepoint_id)
        return (SqSumRes, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                [[np.nan],[np.nan], [np.nan]], {}, np.nan)

    p_changes = [item if not np.isnan(item) else None for item in
                 p_changes]  # "nan" breaks processing, None means no observation here
    p_unc = [item if not np.isnan(item) else None for item in p_unc]

    ts_raw = np.array(p_dates) / 3600. # time series in days
    ts_raw -= ts_raw[0]

    ts_full = np.arange(ts_raw[0], ts_raw[-1], 1)

    ts_nones = np.array([None] * len(ts_full))
    ts = np.concatenate((ts_raw, ts_full))
    order = np.argsort(ts)
    interp_idx = order>len(ts_raw)-1  # indices with even time spacing

    meas_order = [item - 1 for item in order[1:]]
    ts = ts[order]
    p_unc_raw = p_unc
    p_unc = np.concatenate((p_unc_raw, ts_nones))
    p_unc = p_unc[meas_order]
    dts = np.diff(ts)

    zs = np.concatenate((p_changes, ts_nones))[meas_order]
    if dim == 3:
        Fs = [np.array([[1., dt, 1/2*dt**2],
                        [0., 1., dt],
                        [0., 0., 1.]]) for dt in dts]
    elif dim == 2:
        Fs = [np.array([[1., dt],
                        [0., 1.]]) for dt in dts]
    elif dim == 1:
        Fs = [np.array([[1.]]) for dt in dts]
    Rs = [np.array(u) for u in p_unc]

    for q_id, q in enumerate(Q_vals):

        my_filter = kf(dim_x=dim, dim_z=1)
        my_filter.F = np.eye(dim)
        my_filter.x = np.zeros((dim, 1))
        H = np.zeros((1, dim))
        H[0,0] = 1
        my_filter.H = H
        P = np.eye(dim)
        P[0,0] = 0
        my_filter.P = P
        if dim >= 2:
            Qs = [Q_discrete_white_noise(dim=dim, dt=dt, var=(q ** 2), block_size=1) for dt in dts]
        else:
            Qs = [q ** 2 for dt in dts]

        (mu, cov, _, _) = my_filter.batch_filter(zs, Fs=Fs, Rs=Rs, Qs=Qs)
        (xs, Ps, Ks, Pp) = my_filter.rts_smoother(mu, cov, Fs=Fs, Qs=Qs)

        res = np.array([zi - xi for (zi, xi) in zip(zs, xs) if (zi is not None and xi is not None)])
        res = res[:, 0]  # only position residuals
        norm_res = res ** 2  # / res_unc
        SqSumRes[q_id] = np.nansum(norm_res)

        if q_id + 1 == len(Q_vals) or True:  # only for the last q
            smoother_unc = np.array([(p_val * elem[0, 0]) ** (0.5) for elem in Ps])
            if dim >= 2:
                vel_unc = np.array([(p_val * elem[1, 1]) ** (0.5) for elem in Ps])
            else:
                vel_unc = np.array([(0.0) ** (0.5) for elem in Ps])
            smoother_val = np.array([elem[0][0] for elem in xs])

            if dim >= 2:
                smoother_vel = [elem[1][0] for elem in xs]
            else:
                smoother_vel = [0.0 for elem in xs]
            if dim == 3:
                smoother_acc = [elem[2][0] for elem in xs]
            else:
                smoother_acc = [0.0 for elem in xs]

            sig_change = np.abs(smoother_unc) <= np.abs(smoother_val)
            perc_sig = np.count_nonzero(sig_change) / len(smoother_unc)
            sig_change[0] = False  # because loD == 0 for the first epoch
            sign_date = ts[1:][sig_change]
            sign_date = sign_date[0] if len(sign_date) > 0 else np.nan

            time_of_max_v = ts[1:][np.argmax(smoother_vel)]
            time_of_max_a = ts[1:][np.argmax(smoother_acc)]
            time_of_min_v = ts[1:][np.argmin(smoother_vel)]
            time_of_min_a = ts[1:][np.argmin(smoother_acc)]

            freshD = {}

            filterVals = [ts[interp_idx], smoother_val[interp_idx[1:]], smoother_unc[interp_idx[1:]]]

            if corepoint_id % 1000 == 0 and False:
                plt.figure(figsize=(24, 10))
                plt.plot(ts[1:], zs, "rx", label="Measurement")
                plt.plot(ts[1:], mu[:, 0], "b--", label="Kalman filter state")
                plt.plot(ts[1:], xs[:, 0], "g--", label="Kalman smoother state")
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
                plt.ylim([-0.1, 0.1])
                plt.title("Change detection ($\sigma$ = %f)" % q)
                plt.tight_layout()
                plt.legend()

                plt.savefig('kalman_plot_d%s_%s_q%s.svg' % (dim, corepoint_id, q))
                plt.close()

    return (SqSumRes, sign_date,
            smoother_val[-1], smoother_unc[-1],
            np.sum(np.abs(smoother_acc)), np.mean(np.abs(smoother_vel)),
            np.max(smoother_acc), np.max(smoother_vel),
            np.mean(smoother_acc), np.mean(smoother_vel),
            smoother_acc[-1], smoother_vel[-1], time_of_max_v, time_of_max_a,
            np.min(smoother_acc), np.min(smoother_vel),time_of_min_v, time_of_min_a,
            np.max(smoother_val), np.min(smoother_val), smoother_acc[np.argmax(smoother_val)], smoother_acc[np.argmin(smoother_val)],
            filterVals, freshD, perc_sig
            )

def main(infile, ref_epoch, Q_vals, outFile, useevery=1, mkplot=None, exportFilter=False, exportFreshD=False, dim=3):
    if mkplot is None:
        mkplot = list()
    # read first file:
    print("Loading file %s" % infile[0])
    coords, normals, changes, unc, p_dates, leguncs = read_from_las(infile[0], useevery=useevery)
    p_kdtree = spatial.KDTree(coords, leafsize=16, copy_data=True)

    for file in infile[1:]:
        print("Loading file %s" % file)
        coords_i, _, changes_i, unc_i, p_dates_i, leguncs_i = read_from_las(file)  # do not pass "useevery" here due to different order
        dists, ids = p_kdtree.query(coords_i, k=1, distance_upper_bound=0.01)
        valid_dists = np.isfinite(dists)
        ids = ids[valid_dists]
        coords_i = coords_i[valid_dists]
        changes_i = changes_i[valid_dists]
        unc_i = unc_i[valid_dists]
        leguncs_i = leguncs_i[valid_dists]
        inverse_ids = sorted(np.arange(0, len(coords_i)), key=lambda x: ids[x])

        changes = np.concatenate((changes, changes_i[inverse_ids, :]), axis=1)
        unc = np.concatenate((unc, unc_i[inverse_ids, :]), axis=1)
        leguncs = np.concatenate((leguncs, leguncs_i[inverse_ids, :]), axis=1)
        assert np.allclose(coords, coords_i[inverse_ids, :])
        p_dates = p_dates + p_dates_i

    valid_dates = np.ones_like(p_dates).astype(bool)

    p_dates = np.array([p_date for idx, p_date in enumerate(p_dates) if valid_dates[idx]])
    changes = changes[:, valid_dates]
    unc = unc[:, valid_dates]

    order = np.argsort(p_dates)
    p_dates = p_dates[order]
    changes = changes[:, order]
    unc = unc[:, order]

    p_dates = np.concatenate([[float(ref_epoch)], p_dates])
    print(p_dates)


    p = 3  # three-dimensional
    p_val = sstats.chi2.ppf(.95, p)

    process_corepoints = range(coords.shape[0])
    for q in Q_vals:
        pool = multiprocessing.Pool(10)
        ts = time.time()
        outfile_q = outfile
        results = pool.starmap(proc_cp, tqdm.tqdm([[changes[corepoint_id, :], unc[corepoint_id, :],
                                                    p_dates, corepoint_id, p_val, mkplot, [q], exportFilter, exportFreshD, dim, outfile_q] for corepoint_id in process_corepoints]),
                               chunksize=1)
        print("Processing took %s s" % ((time.time()-ts)))

        num_cp = len(process_corepoints)
        SqSumResS = np.full((num_cp, len(Q_vals)), np.nan)
        SignDates = np.full((num_cp,), np.nan)
        FinalChange = np.full((num_cp,), np.nan)
        FinalLoD = np.full((num_cp,), np.nan)
        totalCurv = np.full((num_cp,), np.nan)
        meanSlope = np.full((num_cp,), np.nan)
        maxAcc = np.full((num_cp,), np.nan)
        maxVel = np.full((num_cp,), np.nan)
        meanAcc = np.full((num_cp,), np.nan)
        meanVel = np.full((num_cp,), np.nan)
        lastAcc = np.full((num_cp,), np.nan)
        lastVel = np.full((num_cp,), np.nan)
        tMaxVel = np.full((num_cp,), np.nan)
        tMaxAcc = np.full((num_cp,), np.nan)
        minVel = np.full((num_cp,), np.nan)
        minAcc = np.full((num_cp,), np.nan)
        tMinVel = np.full((num_cp,), np.nan)
        tMinAcc = np.full((num_cp,), np.nan)
        smoothMax = np.full((num_cp,), np.nan)
        smoothMin = np.full((num_cp,), np.nan)
        accAtMax = np.full((num_cp,), np.nan)
        accAtMin = np.full((num_cp,), np.nan)
        dateLeg = np.full((num_cp,), np.nan)
        percLeg = np.full((num_cp,), np.nan)
        percSig = np.full((num_cp,), np.nan)
        filters = [None] * num_cp
        freshD = [None] * num_cp

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
            percSig[ptid] = ptvals[24]

            sig_dates = np.array(p_dates[1:])[np.abs(changes[ptid, :]) >= (1.96 * np.sqrt((leguncs[ptid, :])))]
            percLeg[ptid] = len(sig_dates) / (len(p_dates)-1)
            dateLeg[ptid] =  np.min(sig_dates) if len(sig_dates) > 0 else np.nan
        dateLeg -= p_dates[0]

        out_attrs = {}
        out_attrs['NormalX'] = normals[process_corepoints, 0]
        out_attrs['NormalY'] = normals[process_corepoints, 1]
        out_attrs['NormalZ'] = normals[process_corepoints, 2]

        for qid, q in enumerate(Q_vals):
            out_attrs["%.3f" % q] = SqSumResS[:, qid]

        if exportFilter:
            all_dates = []
            for dates, vals, uncs in filters:
                if len(dates) == 1:
                    continue
                all_dates += list(dates)
            all_dates = sorted(list(set(all_dates)))
            exportData = np.full((num_cp, len(all_dates)+3), np.nan)
            exportUncData = np.full((num_cp, len(all_dates)), np.nan)
            exportData[:, :3] = coords[process_corepoints, :]
            anySigChange = np.full((num_cp, ), np.nan)
            for pix, (dates, vals, uncs) in enumerate(filters):
                if len(dates) == 1:
                    continue
                for date, val, unce in zip(dates, vals, uncs):
                    dix = all_dates.index(date)
                    exportData[pix, 3+dix] = val
                    exportUncData[pix, dix] = unce
                    if abs(val) > unce and (np.isnan(anySigChange[pix]) or abs(anySigChange[pix]) < abs(val)):
                        anySigChange[pix] = val
            np.save(outfile_q.replace(".las", ".npy"), exportData)
            np.save(outfile_q.replace(".las", "_unc.npy"), exportUncData)

            out_attrs['anySigChange'] = np.logical_not(np.isnan(anySigChange))
            out_attrs['anySigChangeMagn'] = anySigChange

        if exportFreshD:
            fresh_elements = np.full((coords.shape[0], len(freshD[0])+3), np.nan)
            fresh_elements[:, :3] = coords
            for pix, freshDict in enumerate(freshD):
                for kix, (key, val) in enumerate(freshDict.items()):
                    fresh_elements[pix][kix+3] = val
            np.save(outfile_q.replace(".las", "_fresh.npy"), fresh_elements)

        min_q_idx = np.argmin(SqSumResS, axis=1)
        out_attrs['min_res_q_magn'] = SqSumResS[np.arange(len(min_q_idx)), min_q_idx]
        out_attrs['date'] = SignDates
        out_attrs['dateLeg'] = dateLeg
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
        out_attrs['percLeg'] = percLeg
        out_attrs['percSig'] = percSig

        write_to_las(outfile_q, coords[process_corepoints, :], out_attrs)

if __name__ == '__main__':
    infile = glob.glob(r"synth_m3c2\*.las")
    ref_epoch = 0

    for dim, name, Q_vals in zip(
        [1,2,3],
        ['X', 'XV', 'XVA'],
        [
            [0.0005, 0.002, 0.005],
            [0.0002],
            [0.00002, 0.00001, 0.000005]
        ]
    ):
        for q in Q_vals:
            outfile = rf"synth_kalman\kalman_{name}_q{q}.las"
            main(infile, ref_epoch, [q], outfile, useevery=1,
                 mkplot=[1],
                 exportFilter=True, exportFreshD=False, dim=dim)
