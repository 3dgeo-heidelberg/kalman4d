from sklearn.cluster import KMeans
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

def read_from_las(path, useevery=1):
    from laspy.file import File as LasFile
    inFile = LasFile(path, mode='r')
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]
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
    normals = normals[::useevery, :]
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
            vals[:, dateidx] = inFile.points["point"]["val_%s" % field_date][::useevery]
            uncs[:, dateidx] = inFile.points["point"]["unc_%s" % field_date][::useevery]
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


def main(infile, ks, outFile, useevery=1):
    coords, normals, vals, uncs, dates = read_from_las(infile, useevery)
    labels = np.full((coords.shape[0], len(ks)), np.nan)

    ## override feature space from numpy file
    f = np.load(infile.replace(".las", ".npy"))
    vals = f[:, 3:]

    for kidx, k in enumerate(tqdm.tqdm(ks)):
        X = vals
        nan_indicator = np.logical_not(np.isnan(np.sum(X, axis=1)))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X[nan_indicator, :])
        labels[nan_indicator, kidx] = kmeans.labels_

    out_attrs = {}
    for kidx, k in enumerate(ks):
        out_attrs["label_%d" % k] = labels[:, kidx]

    write_to_las(outFile, coords, out_attrs)

if __name__ == '__main__':
    infile = r"kalman_25cm_Proc_q0_05.las"
    outfile = r"kMeans_cluster_based_on_change_values.las"
    ks = [20, 30, 50, 100,150] # list(range(3, 10)) + [20, 30, 50]

    main(infile, ks=ks, outFile=outfile, useevery=1)