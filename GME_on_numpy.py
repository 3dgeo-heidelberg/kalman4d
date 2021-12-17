from sklearn.mixture import GaussianMixture
import tqdm
import numpy as np
from datetime import datetime

# from https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
def datetime2year(dt):
    year_part = dt - datetime(year=dt.year, month=1, day=1)
    year_length = (
        datetime(year=dt.year + 1, month=1, day=1)
        - datetime(year=dt.year, month=1, day=1)
    )
    return dt.year + year_part / year_length

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

def main(infile, ks, outFile, useevery=1):
    ## override feature space from numpy file
    f = np.load(infile)
    vals = f[:, 3:]
    labels = np.full((vals.shape[0], len(ks)), np.nan)
    coords = f[:, :3]

    not_all_nans_in_row = np.logical_not(np.all(np.isnan(vals), axis=0))
    X = vals[:, not_all_nans_in_row]

    for kidx, k in enumerate(tqdm.tqdm(ks)):
        nan_indicator = np.logical_not(np.isnan(np.sum(X, axis=1)))
        gme = GaussianMixture(n_components=k, random_state=0).fit(X[nan_indicator, :])
        labels[nan_indicator, kidx] = gme.predict(X[nan_indicator, :])

    out_attrs = {}
    for kidx, k in enumerate(ks):
        out_attrs["label_%d" % k] = labels[:, kidx]

    write_to_las(outFile, coords, out_attrs)

if __name__ == '__main__':
    infile = r"kalman_25cm_Proc_q0_05.npy"
    outfile = r"GMM_clusters_from_tsfresh.las"
    ks = [20, 30, 50, 100,150] # list(range(3, 10)) + [20, 30, 50]

    main(infile, ks=ks, outFile=outfile, useevery=1)