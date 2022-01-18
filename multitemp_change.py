
from scipy import spatial
import numpy as np
import math
import multiprocessing as mp
import pickle
import scipy.io as sio
import os
from collections import namedtuple
import re
from tqdm import tqdm
import time
import sys
from datetime import datetime

import tf_helper


# from https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
def datetime2year(dt):
    year_part = dt - datetime(year=dt.year, month=1, day=1)
    year_length = (
        datetime(year=dt.year + 1, month=1, day=1)
        - datetime(year=dt.year, month=1, day=1)
    )
    return dt.year + year_part / year_length

np.seterr(divide='ignore', invalid='ignore')

M3C2MetaInfo = namedtuple('M3C2MetaInfo', ('spInfos', 'tfM', 'Cxx', 'redPoint', 'searchrad', 'maxdist'))
SPMetaInfo = namedtuple('SPMetaInfo', ('origin', 'sigma_range', 'sigma_yaw', 'sigma_scan', 'ppm'))

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

dij = np.zeros((3, 3))
dij[0, 0] = dij[1, 1] = dij[2, 2] = 1

def picklebig(obj, file):
    max_bytes = 2 ** 31 - 1
    ## write
    bytes_out = pickle.dumps(obj)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def unpicklebig(file):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

n = np.zeros((3,))
poa_pts = np.zeros((3,100))
path_opt = np.einsum_path('mi, ijk, j, kn -> mn', dij, eijk, n, poa_pts, optimize='optimal')


def getAlongAcrossSqBatch(pts, poa, n):
    pts_poa = pts - poa[:, np.newaxis]
    alongs = n.dot(pts_poa)
    #pts_poa_n = pts_poa + n[:, np.newaxis]
    poa_pts = poa[:, np.newaxis] - pts
    crosses = np.einsum('mi, ijk, j, kn -> mn', dij, eijk, n, poa_pts, optimize=path_opt[0])
    across2 = np.einsum('ij, ij -> j', crosses, crosses)
    return (alongs, across2)

def read_from_las(path):
    from laspy.file import File as LasFile
    inFile = LasFile(path, mode='r')
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
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
    print(path, "Normals: ", normals)
    scanpos = getattr(inFile, 'pt_src_id', None)

    if "amplitude" in inFile.points.dtype.fields["point"][0].fields:
        amp = inFile.points["point"]["amplitude"]
    elif "Amplitude" in inFile.points.dtype.fields["point"][0].fields:
        amp = inFile.points["point"]["Amplitude"]
    else:
        amp = None
    if "deviation" in inFile.points.dtype.fields["point"][0].fields:
        dev = inFile.points["point"]["deviation"]
    elif "Deviation" in inFile.points.dtype.fields["point"][0].fields:
        dev = inFile.points["point"]["Deviation"]
    else:
        dev = None
    return coords, normals, scanpos, amp, dev

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


def process_corepoint_list(corepoints, corepoint_normals,
                           p_idx, p_shm_names, p_sizes, p_positions, p_dates,
                           M3C2Meta, idx, return_dict, pbarQueue):
    pj_shm = [mp.shared_memory.SharedMemory(name=pi_shm_name) for pi_shm_name in p_shm_names]
    p1_coords, *pi_coords = [np.ndarray(pi_size, dtype=np.float, buffer=pi_shm.buf) for (pi_size, pi_shm) in zip(p_sizes, pj_shm)]
    p1_positions, *pi_positions = p_positions
    p1_idx, *pi_idx = p_idx
    pbarQueue.put((0, 1))  # point processing

    max_dist = M3C2Meta['maxdist']
    search_radius = M3C2Meta['searchrad']

    M3C2_vals = np.full((corepoints.shape[0], len(pi_coords)), np.nan, dtype=np.float64)
    M3C2_uncs = np.full((corepoints.shape[0], len(pi_coords)), np.nan, dtype=np.float64)

    for cp_idx, p1_neighbours in enumerate(p1_idx):
        if cp_idx % 10 == 9:
            pbarQueue.put((10, 0))  # point processing
        elif len(p1_idx) - cp_idx < 10:
            pbarQueue.put((1, 0))
        n = corepoint_normals[cp_idx]
        p1_curr_pts = p1_coords[p1_neighbours, :]
        along1, acrossSq1 = getAlongAcrossSqBatch(p1_curr_pts.T, corepoints[cp_idx], n)
        p1_curr_pts = p1_curr_pts[np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius ** 2), :]
        p1_scanPos = p1_positions[p1_neighbours]
        p1_scanPos = p1_scanPos[np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius ** 2)]
        if p1_curr_pts.shape[0] < M3C2Meta["minneigh"]:
            continue
        elif p1_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
            p1_curr_pts = p1_curr_pts[np.argsort(acrossSq1[:M3C2Meta['maxneigh']])]
            p1_scanPos = p1_scanPos[np.argsort(acrossSq1[:M3C2Meta['maxneigh']])]
        p1_CoG, p1_local_Cxx = get_local_mean_and_Cxx_nocorr(M3C2Meta, p1_curr_pts, p1_scanPos, epoch=0, tf=False) # only one dataset has been transformed


        for epoch_i, (p2_idx, p2_coords, p2_positions) in enumerate(zip(pi_idx, pi_coords, pi_positions)):  # for every epoch do
            p2_neighbours = p2_idx[cp_idx]
            p2_curr_pts = p2_coords[p2_neighbours, :]
            along2, acrossSq2 = getAlongAcrossSqBatch(p2_curr_pts.T, corepoints[cp_idx], n)
            p2_curr_pts = p2_curr_pts[np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius ** 2), :]
            p2_scanPos = p2_positions[p2_neighbours]
            p2_scanPos = p2_scanPos[np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius ** 2)]
            if p2_curr_pts.shape[0] < M3C2Meta["minneigh"]:
                continue
            elif p2_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
                p2_curr_pts = p2_curr_pts[np.argsort(acrossSq2[:M3C2Meta['maxneigh']])]
                p2_scanPos = p2_scanPos[np.argsort(acrossSq2[:M3C2Meta['maxneigh']])]
            p2_CoG, p2_local_Cxx = get_local_mean_and_Cxx_nocorr(M3C2Meta, p2_curr_pts, p2_scanPos, epoch=epoch_i, tf=True)

            p1_p2_CoG_Cxx = np.zeros((6, 6))
            p1_p2_CoG_Cxx[0:3, 0:3] = p1_local_Cxx
            p1_p2_CoG_Cxx[3:6, 3:6] = p2_local_Cxx

            M3C2_dist = n.dot(p2_CoG - p1_CoG)
            F = np.hstack([-n, n])
            M3C2_unc = np.dot(F, np.dot(p1_p2_CoG_Cxx, F))

            M3C2_vals[cp_idx, epoch_i] = M3C2_dist
            M3C2_uncs[cp_idx, epoch_i] = M3C2_unc

    val_dict = {"val_%s" % dt.timestamp(): M3C2_vals[:, i] for i, dt in enumerate(p_dates[1:])}
    unc_dict = {"unc_%s" % dt.timestamp(): M3C2_uncs[:, i] for i, dt in enumerate(p_dates[1:])}
    merge_dict = dict()
    merge_dict.update(val_dict)
    merge_dict.update(unc_dict)
    return_dict[idx] = merge_dict
    pbarQueue.put((0, -1))  # point processing
    for p_shm in pj_shm:
        p_shm.close()

def get_local_mean_and_Cxx_nocorr(M3C2Meta, curr_pts, curr_pos, epoch, tf=True):
    nPts = curr_pts.shape[0]

    A = np.tile(np.eye(3), (nPts, 1))
    ATP = np.zeros((3, 3 * nPts))
    tfM = M3C2Meta['tfM'][epoch] if tf else np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    dx = np.zeros((nPts,), dtype=np.float)
    dy = np.zeros((nPts,), dtype=np.float)
    dz = np.zeros((nPts,), dtype=np.float)
    rrange = np.zeros((nPts,), dtype=np.float)
    sinscan = np.zeros((nPts,), dtype=np.float)
    cosscan = np.zeros((nPts,), dtype=np.float)
    cosyaw = np.zeros((nPts,), dtype=np.float)
    sinyaw = np.zeros((nPts,), dtype=np.float)
    sigmaRange = np.zeros((nPts,), dtype=np.float)
    sigmaYaw = np.zeros((nPts,), dtype=np.float)
    sigmaScan = np.zeros((nPts,), dtype=np.float)

    for scanPosId in np.unique(curr_pos):
        scanPos = np.array(M3C2Meta['spInfos'][epoch-1][scanPosId-1]['origin'])
        scanPosPtsIdx = np.arange(curr_pos.size) # == scanPosId

        dd = curr_pts[scanPosPtsIdx, :] - scanPos[np.newaxis, :]
        dlx, dly, dlz = dd[:, 0], dd[:, 1], dd[:, 2]
        yaw = np.arctan2(dly, dlx)
        planar_dist = np.hypot(dlx, dly)
        scan = np.pi / 2 - np.arctan(dlz / planar_dist)
        rrange[scanPosPtsIdx] = np.hypot(planar_dist, dlz)
        sinscan[scanPosPtsIdx] = np.sin(scan)
        cosscan[scanPosPtsIdx] = np.cos(scan)
        sinyaw[scanPosPtsIdx] = np.sin(yaw)
        cosyaw[scanPosPtsIdx] = np.cos(yaw)

        dr = curr_pts[scanPosPtsIdx, :] - M3C2Meta['redPoint'][epoch-1]
        dx[scanPosPtsIdx] = dr[:, 0]
        dy[scanPosPtsIdx] = dr[:, 1]
        dz[scanPosPtsIdx] = dr[:, 2]

        sigmaRange[scanPosPtsIdx] = np.array(
            np.sqrt(M3C2Meta['spInfos'][epoch-1][scanPosId-1]['sigma_range']**2 +
                    M3C2Meta['spInfos'][epoch-1][scanPosId-1]['ppm'] * 1e-6 * rrange[scanPosPtsIdx]**2))  # a + b*d
        sigmaYaw[scanPosPtsIdx] = np.array(M3C2Meta['spInfos'][epoch-1][scanPosId-1]['sigma_yaw'])
        sigmaScan[scanPosPtsIdx] = np.array(M3C2Meta['spInfos'][epoch-1][scanPosId-1]['sigma_scan'])

    if tf:
        Cxx = M3C2Meta['Cxx'][epoch-1]
        SigmaXiXj = (dx ** 2 * Cxx[0, 0] +  # a11a11
                     2 * dx * dy * Cxx[0, 1] +  # a11a12
                     dy ** 2 * Cxx[1, 1] +  # a12a12
                     2 * dy * dz * Cxx[1, 2] +  # a12a13
                     dz ** 2 * Cxx[2, 2] +  # a13a13
                     2 * dz * dx * Cxx[0, 2] +  # a11a13
                     2 * (dx * Cxx[0, 9] +  # a11tx
                          dy * Cxx[1, 9] +  # a12tx
                          dz * Cxx[2, 9]) +  # a13tx
                     Cxx[9, 9])  # txtx

        SigmaYiYj = (dx ** 2 * Cxx[3, 3] +  # a21a21
                     2 * dx * dy * Cxx[3, 4] +  # a21a22
                     dy ** 2 * Cxx[4, 4] +  # a22a22
                     2 * dy * dz * Cxx[4, 5] +  # a22a23
                     dz ** 2 * Cxx[5, 5] +  # a23a23
                     2 * dz * dx * Cxx[3, 5] +  # a21a23
                     2 * (dx * Cxx[3, 10] +  # a21ty
                          dy * Cxx[4, 10] +  # a22ty
                          dz * Cxx[5, 10]) +  # a23ty
                     Cxx[10, 10])  # tyty

        SigmaZiZj = (dx ** 2 * Cxx[6, 6] +  # a31a31
                     2 * dx * dy * Cxx[6, 7] +  # a31a32
                     dy ** 2 * Cxx[7, 7] +  # a32a32
                     2 * dy * dz * Cxx[7, 8] +  # a32a33
                     dz ** 2 * Cxx[8, 8] +  # a33a33
                     2 * dz * dx * Cxx[6, 8] +  # a31a33
                     2 * (dx * Cxx[6, 11] +  # a31tz
                          dy * Cxx[7, 11] +  # a32tz
                          dz * Cxx[8, 11]) +  # a33tz
                     Cxx[11, 11])  # tztz

        SigmaXiYj = (Cxx[9, 10] +  # txty
                     dx * Cxx[0, 10] +  # a11ty
                     dy * Cxx[1, 10] +  # a12ty
                     dz * Cxx[2, 10] +  # a13ty
                     dx * (Cxx[3, 9] +
                           Cxx[0, 3] * dx +
                           Cxx[1, 3] * dy +
                           Cxx[2, 3] * dz) +
                     dy * (Cxx[4, 9] +
                           Cxx[0, 4] * dx +
                           Cxx[1, 4] * dy +
                           Cxx[2, 4] * dz) +
                     dz * (Cxx[5, 9] +
                           Cxx[0, 5] * dx +
                           Cxx[1, 5] * dy +
                           Cxx[2, 5] * dz)
                     )

        SigmaXiZj = (Cxx[9, 11] +  # txtz
                     dx * Cxx[0, 11] +  # a11tz
                     dy * Cxx[1, 11] +  # a12tz
                     dz * Cxx[2, 11] +  # a13tz
                     dx * (Cxx[6, 9] +
                           Cxx[0, 6] * dx +
                           Cxx[1, 6] * dy +
                           Cxx[2, 6] * dz) +
                     dy * (Cxx[7, 9] +
                           Cxx[0, 7] * dx +
                           Cxx[1, 7] * dy +
                           Cxx[2, 7] * dz) +
                     dz * (Cxx[8, 9] +
                           Cxx[0, 8] * dx +
                           Cxx[1, 8] * dy +
                           Cxx[2, 8] * dz)
                     )

        SigmaYiZj = (Cxx[10, 11] +  # tytz
                     dx * Cxx[6, 10] +  # a21tx
                     dy * Cxx[7, 10] +  # a22tx
                     dz * Cxx[8, 10] +  # a23tx
                     dx * (Cxx[3, 11] +
                           Cxx[3, 6] * dx +
                           Cxx[3, 7] * dy +
                           Cxx[3, 8] * dz) +
                     dy * (Cxx[4, 11] +
                           Cxx[4, 6] * dx +
                           Cxx[4, 7] * dy +
                           Cxx[4, 8] * dz) +
                     dz * (Cxx[5, 11] +
                           Cxx[5, 6] * dx +
                           Cxx[5, 7] * dy +
                           Cxx[5, 8] * dz)
                     )
        C11 = np.sum(SigmaXiXj)  # sum over all j
        C12 = np.sum(SigmaXiYj)  # sum over all j
        C13 = np.sum(SigmaXiZj)  # sum over all j
        C22 = np.sum(SigmaYiYj)  # sum over all j
        C23 = np.sum(SigmaYiZj)  # sum over all j
        C33 = np.sum(SigmaZiZj)  # sum over all j
        local_Cxx = np.array([[C11, C12, C13], [C12, C22, C23], [C13, C23, C33]])
    else:
        local_Cxx = np.zeros((3,3))

    C11p = ((tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
                   tfM[0, 1] * sinyaw * sinscan +
                   tfM[0, 2] * cosscan) ** 2 * sigmaRange ** 2 +
                  (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
                   tfM[0, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
                  (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
                   tfM[0, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[0, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C12p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
                   tfM[1, 1] * sinyaw * sinscan +
                   tfM[1, 2] * cosscan) *
                  (tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
                   tfM[0, 1] * sinyaw * sinscan +
                   tfM[0, 2] * cosscan) * sigmaRange ** 2 +
                  (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
                   tfM[1, 1] * rrange * cosyaw * sinscan) *
                  (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
                   tfM[0, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
                  (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
                   tfM[0, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[0, 2] * rrange * sinscan) *
                  (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
                   tfM[1, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[1, 2] * rrange * sinscan) * sigmaScan ** 2)

    C22p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
                   tfM[1, 1] * sinyaw * sinscan +
                   tfM[1, 2] * cosscan) ** 2 * sigmaRange ** 2 +
                  (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
                   tfM[1, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
                  (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
                   tfM[1, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[1, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C23p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
                   tfM[1, 1] * sinyaw * sinscan +
                   tfM[1, 2] * cosscan) *
                  (tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
                   tfM[2, 1] * sinyaw * sinscan +
                   tfM[2, 2] * cosscan) * sigmaRange ** 2 +
                  (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
                   tfM[1, 1] * rrange * cosyaw * sinscan) *
                  (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
                   tfM[2, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
                  (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
                   tfM[2, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[2, 2] * rrange * sinscan) *
                  (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
                   tfM[1, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[1, 2] * rrange * sinscan) * sigmaScan ** 2)

    C33p = ((tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
                   tfM[2, 1] * sinyaw * sinscan +
                   tfM[2, 2] * cosscan) ** 2 * sigmaRange ** 2 +
                  (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
                   tfM[2, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
                  (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
                   tfM[2, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[2, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C13p = ((tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
                   tfM[2, 1] * sinyaw * sinscan +
                   tfM[2, 2] * cosscan) *
                  (tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
                   tfM[0, 1] * sinyaw * sinscan +
                   tfM[0, 2] * cosscan) * sigmaRange ** 2 +
                  (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
                   tfM[2, 1] * rrange * cosyaw * sinscan) *
                  (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
                   tfM[0, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
                  (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
                   tfM[2, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[2, 2] * rrange * sinscan) *
                  (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
                   tfM[0, 1] * rrange * sinyaw * cosscan +
                   -1 * tfM[0, 2] * rrange * sinscan) * sigmaScan ** 2)
    local_Cxx[0,0] += np.sum(C11p)
    local_Cxx[0,1] += np.sum(C12p)
    local_Cxx[0,2] += np.sum(C13p)
    local_Cxx[1,0] += np.sum(C12p)
    local_Cxx[1,1] += np.sum(C22p)
    local_Cxx[1,2] += np.sum(C23p)
    local_Cxx[2,1] += np.sum(C23p)
    local_Cxx[2,0] += np.sum(C13p)
    local_Cxx[2,2] += np.sum(C33p)
    # Get mean without correlation (averages out anyway, or something...)
    for pii in range(nPts):
        Cxx = np.array([[C11p[pii], C12p[pii], C13p[pii]],
                        [C12p[pii], C22p[pii], C23p[pii]],
                        [C13p[pii], C23p[pii], C33p[pii]]])
        if np.linalg.det(Cxx) == 0:
            Cxx = np.eye(3)
        Cix = np.linalg.inv(Cxx)
        ATP[:, pii*3:(pii+1)*3] = np.diag(np.diag(Cix))
    N = np.dot(ATP, A)
    Qxx = np.linalg.inv(N)  # can only have > 0 in main diagonal!
    # pts_m = curr_pts.mean(axis=0)
    l = (curr_pts).flatten(order='c')
    # l = (curr_pts-pts_m).flatten(order='c')
    # mean = np.dot(Qxx, np.dot(ATP, l)) + pts_m
    mean = np.dot(Qxx, np.dot(ATP, l))

    return mean, local_Cxx/(nPts)


def updatePbar(total, queue, maxProc):
    desc = "Processing core points"
    pCount = 0
    pbar = tqdm(total=total, ncols=100, desc=desc + " (%02d/%02d Process(es))" % (pCount, maxProc))
    while True:
        inc, process = queue.get()
        pbar.update(inc)
        if process != 0:
            pCount += process
            pbar.set_description(desc + " (%02d/%02d Process(es))" % (pCount, maxProc))


def main(pi_files, core_point_file, CxxFiles, p_dates, outFile):

    VZ_2000_sigmaRange = 0.005
    VZ_2000_ppm = 0
    VZ_2000_sigmaScan = 0.00027 /4
    VZ_2000_sigmaYaw = 0.00027 /4

    SP1 = {'origin': [0.0, 0.0, 0.0],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}

    Cxx_i = []
    tfM_i = []
    refPointMov_i = []
    for epoch_id in range(len(pi_files)-1):

        matfile = sio.loadmat(CxxFiles[epoch_id])
        Cxx = matfile['Cxx']
        refPointMov = matfile['redPoi']
        tf = matfile['xhat']
        tfM = tf_helper.opk2R(tf[0, 0], tf[1,0], tf[2,0])
        tfM = np.hstack((tfM, np.vstack((tf[3,0], tf[4,0], tf[5,0]))))
        Cxx = tf_helper.CxxOPK_XYZ2Cxx14(tf[0, 0], tf[1,0], tf[2,0], Cxx)
        Cxx = np.pad(Cxx, ((0, 6), (0, 6)), mode='constant', constant_values=0)
        #Cxx = np.zeros(Cxx.shape) #uncomment to ignore coregistration errors
        Cxx_i.append(Cxx)
        tfM_i.append(tfM)
        refPointMov_i.append(refPointMov)

    M3C2Meta = {'spInfos': [[SP1] * len(pi_files)],
                'tfM': tfM_i,
                'Cxx': Cxx_i,
                'redPoint': refPointMov_i,
                'searchrad': .5,
                'maxdist': 3,
                'minneigh': 5,
                'maxneigh': 100000,
                'leg_ref_err': 0.02}

    NUM_THREADS = 10
    NUM_BLOCKS = 60

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        pass
    elif gettrace():
        print('Debugging mode detected, running only single thread mode')
        NUM_THREADS = 1

    LEAF_SIZE = 1024 #32

    # load file
    print("Loading point clouds")
    pi_coords = []
    pi_positions = []
    pi_kdtrees = []

    for p_idx, p_file in enumerate(pi_files):
        p_coords, _, p_positions, _, _ = read_from_las(p_file)
        p_coords, unique_idx = np.unique(p_coords, axis=0, return_index=True)
        p_positions = p_positions[unique_idx]
        p_pickle_file = p_file.replace(".las", "_kd.pickle")

        # build kd tree
        if p_idx > 0: # transform second and subsequent point clouds
            p_coords = p_coords - refPointMov_i[p_idx-1]
            p_coords = np.dot(tfM_i[p_idx-1][:3, :3], p_coords.T).T + tfM_i[p_idx-1][:, 3] + refPointMov_i[p_idx-1]
        if not os.path.exists(p_pickle_file):
            print("Building kd-Tree for PC %d" % p_idx)
            p_kdtree = spatial.KDTree(p_coords, leafsize=LEAF_SIZE)
            picklebig(p_kdtree, p_pickle_file)
        else:
            print("Loading pre-built kd-Tree for PC %d" % p_idx)
            p_kdtree = unpicklebig(p_pickle_file)
        pi_coords.append(p_coords)
        pi_positions.append(p_positions)
        pi_kdtrees.append(p_kdtree)

    # load query points
    query_coords, query_norms, _, _, _ = read_from_las(core_point_file)
    idx_randomized = np.arange(query_coords.shape[0])
    np.random.shuffle(idx_randomized)
    query_coords = query_coords[idx_randomized, :]
    query_norms = query_norms[idx_randomized, :]

    if query_norms is None:
        print("Core point point cloud needs normals set. Exiting.")
        exit(-1)
    subsample = False
    if subsample:
        sub_idx = np.random.choice(np.arange(0, query_coords.shape[0]), 1000)
        query_coords = query_coords[sub_idx]
        query_norms = query_norms[sub_idx]
    query_coords_subs = np.array_split(query_coords, NUM_BLOCKS)
    query_norms_subs = np.array_split(query_norms, NUM_BLOCKS)
    print("Total: %d core points" % (query_coords.shape[0]))

    # start mp
    manager = mp.Manager()
    return_dict = manager.dict()


    # prepare shared memory
    pi_coords_shm = []
    for p_idx, p_coords in enumerate(pi_coords):
        p_coords_shm = mp.shared_memory.SharedMemory(create=True, size=p_coords.nbytes)
        p_coords_sha = np.ndarray(p_coords.shape, dtype=p_coords.dtype, buffer=p_coords_shm.buf)
        p_coords_sha[:] = p_coords[:]
        pi_coords_shm.append(p_coords_shm)
        p_coords = None # free memory

    max_dist = M3C2Meta['maxdist']
    search_radius = M3C2Meta['searchrad']
    effective_search_radius = math.hypot(max_dist, search_radius)


    print("Querying neighbours")
    pbarQueue = mp.Queue()
    pbarProc = mp.Process(target=updatePbar, args=(query_coords.shape[0], pbarQueue, NUM_THREADS))
    pbarProc.start()
    procs = []

    last_started_idx = -1
    running_ps = []
    while True:
        if len(running_ps) < NUM_THREADS:
            last_started_idx += 1
            if last_started_idx < len(query_coords_subs):
                curr_subs = query_coords_subs[last_started_idx]
                pi_idx = []
                for p_kdtree in pi_kdtrees:
                    p_idx = p_kdtree.query_ball_point(curr_subs, r=effective_search_radius, workers=(NUM_THREADS - len(running_ps)))
                    pi_idx.append(p_idx)

                p = mp.Process(target=process_corepoint_list, args=(
                    curr_subs, query_norms_subs[last_started_idx],
                    pi_idx, [shm.name for shm in pi_coords_shm], [coords.shape for coords in pi_coords], pi_positions,
                    p_dates,
                    M3C2Meta, last_started_idx, return_dict,
                    pbarQueue))
                procs.append(p)

                procs[last_started_idx].start()
                running_ps.append(last_started_idx)
                pi_idx = None # free memory
            else:
                break
        for running_p in running_ps:
            if not procs[running_p].is_alive():
                running_ps.remove(running_p)
        time.sleep(1)

    for p in procs:
        p.join()
    print("\nAll threads terminated.")
    pbarQueue.put(1)
    pbarProc.terminate()
    for shm in pi_coords_shm:
        shm.close()
        shm.unlink()

    out_attrs = {key: np.empty(query_coords.shape[0], dtype=val.dtype) for key, val in return_dict[0].items()}
    for key in out_attrs:
        curr_start = 0
        for i in range(NUM_BLOCKS):
            curr_len = return_dict[i][key].shape[0]
            out_attrs[key][curr_start:curr_start + curr_len] = return_dict[i][key]
            curr_start += curr_len
    out_attrs['normals'] = query_norms
    # Backup saving strategies are commented out
    # picklebig(query_coords, outFile + ".coords.pickle")
    # picklebig(out_attrs, outFile + ".pickle")
    # np.save(outFile + ".coords.npy", query_coords)
    # np.save(outFile + ".npy", out_attrs)
    write_to_las(outFile, query_coords, out_attrs)


if __name__ == '__main__':
    import glob

    tbegin = time.time()
    tiles_pre = list(glob.glob(r"pointclouds_gnd\*_gnd.las"))
    tiles_pre = sorted(tiles_pre)
    step = 20
    for chunk in range(len(tiles_pre)//step + 1):
        print("Running for Epochs %s - %s (Chunk %s/%s)" % (1+chunk*step,(chunk+1)*step, chunk+1, len(tiles_pre)//step + 1))
        tiles_curr = [tiles_pre[0]] + tiles_pre[1+chunk*step:1+(chunk+1)*step]
        if len(tiles_curr) == 1:
            print("Empty chunk.")
            break
        outFile = r"change_full_info_%02d.las" % chunk
        CxxFiles = [r"trafos\vals_" + ep.replace("_gnd.las", r"\Cxx.mat").replace(r"pointclouds_gnd\ScanPos001 - SINGLESCANS - ", "") for ep in tiles_curr[1:]]  # first one does not have any CXX
        p_dates = [datetime.strptime(ep.replace("_gnd.las","")[-13:],'%y%m%d_%H%M%S') for ep in tiles_curr]
        print("Reference Epoch (needed in follow-up scripts): %s" % p_dates[0].timestamp())
        tstart = time.time()
        pi_files = tiles_curr
        core_point_file =r"corepoints_with_normals.las"
        if os.path.exists(outFile):
            print("Skipping chunk because outFile exists!")
            continue
        main(pi_files, core_point_file, CxxFiles, p_dates, outFile)
        tend = time.time()
        print("== > Finished chunk (Took %.3f s)." % ((tend - tstart)))
    print("Finished all chunks.")
