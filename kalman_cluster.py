import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np

def read_from_las(path, readAttrs, useevery=1):
    from laspy.file import File as LasFile
    inFile = LasFile(path, mode='r')
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]
    try:
        normals = getattr(inFile, 'normals', None)
        normals = normals[::useevery, :]
    except:
        normals = None
    if normals is None:
        try:
            n0 = inFile.points["point"]["NormalX"]
            n1 = inFile.points["point"]["NormalY"]
            n2 = inFile.points["point"]["NormalZ"]
            normals = np.stack((n0,n1,n2)).T
            normals = normals[::useevery, :]
        except:
            normals = None
    las_fields = sorted(list(inFile.points["point"].dtype.fields))
    loaded_attrs = {}
    for attr in readAttrs:
        loaded_attrs[attr] = inFile.points["point"][attr][::useevery]

    return coords, normals, loaded_attrs

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

def main(infile, outfile, ks, attrs):
    coords, normals, attrs = read_from_las(infile, attrs)

    labels = np.full((coords.shape[0], len(ks)), np.nan)

    for kidx, k in enumerate(tqdm.tqdm(ks)):
        X = np.stack([item for key, item in attrs.items()]).T
        nan_indicator = np.logical_not(np.isnan(np.sum(X, axis=1)))
        gme = GaussianMixture(n_components=k, random_state=0).fit(X[nan_indicator, :])
        labels[nan_indicator, kidx] = gme.predict(X[nan_indicator, :])


    out_attrs = {}
    for kidx, k in enumerate(ks):
        out_attrs["label_%d" % k] = labels[:, kidx]

    write_to_las(outfile, coords, out_attrs)


if __name__ == '__main__':
    infile = r"kalman_25cm_Proc_q0_05.las"
    outfile = r"GMM_clusters_from_featurespace.las"
    ks = [20, 30, 50, 100, 150]
    attrs_static = [
        'maxvel',
        'maxacc',
        'minvel',
        'minacc',
        'accatmin',
        'accatmax',
        'mindispl',
        'maxdispl',
    ]
    attrs_full_ts = [
        'meanvel',
        'meanacc',
        'meanabsslope',
        'totalcurv',
        'min_res_q_magn'
    ]
    attrs_time = [
        'tmaxvel',
        'tmaxacc',
        'tminvel',
        'tminacc',
    ]
    attrs_final = [
        'lastacc',
        'lastvel',
        'change'
    ]
    attrs = attrs_final +  attrs_time + attrs_full_ts + attrs_static
    main(infile, outfile, ks=ks, attrs=attrs)