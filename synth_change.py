import shutil

import numpy as np
from scipy import spatial
import tqdm
import subprocess
from pathlib import Path
import laspy

SCENE = r"""
<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="synth-scene" name="Synth">
        <part>
            <filter type="objloader">
				<param type="string" key="filepath" value="{meshfile}"/>
				<param type="string" key="up" value="z" />
				<!--<param type="string" key="matfile" value="D:\lwiniwarter\Projects\ep-m3c2\mp_imp\mesh.mtl" />
				<param type="string" key="matname" value="Plane" /> -->
            </filter>

        </part>
    </scene>
</document>
"""

SURVEY = r"""
<?xml version="1.0" encoding="UTF-8"?>
<document>
	<!-- default scanner settings -->
	<scannerSettings id="profile1" active="true" pulseFreq_hz="100000" scanFreq_hz="30" scanAngle_deg="100" headRotatePerSec_deg="3.0"/>
    <survey name="synth-survey" scene="%s#synth-scene" platform="C:\Users\Lukas\Documents\Data\vals2021\10_synth_mesh\platforms.xml#tripod" scanner="C:\Users\Lukas\Documents\Data\vals2021\10_synth_mesh\scanners_tls.xml#riegl_vz400">
        <FWFSettings binSize_ns="0.2" beamSampleQuality="1" />
        <leg>
            <platformSettings x="300.0" y="50" z="0.0" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="0.0" verticalAngleMax_deg="30" headRotateStart_deg="0" headRotateStop_deg="360" trajectoryTimeInterval_s="1.0"/>
        </leg>
    </survey>
</document>
"""


outfile = r'synth_scene\epoch_%d.obj'
SCENE_FILE = r'synth_scene\scene.xml'
SURVEY_FILE = r'synth_scene\survey.xml'
HELIOS_DIR = Path(r'helios\hpp_110')
OUTDIR = r"synth_mesh"

SIGMA_ALPHA = SIGMA_BETA = 0.001 * np.pi/180.
SIGMA_GAMMA = 0.005 * np.pi/180.
SIGMA_X_Y_Z = 0.002
SIGMA_M = 0.00001

def find_playback_dir(survey_path, WORKING_DIR):
    playback = Path(WORKING_DIR) / 'output' / 'Survey Playback'
    with open(Path(WORKING_DIR) / survey_path, 'r') as sf:
        for line in sf:
            if '<survey name' in line:
                survey_name = line.split('name="')[1].split('"')[0]
    if not (playback / survey_name).is_dir():
        raise Exception('Could not locate output directory')
    last_run_dir = sorted(list((playback / survey_name).glob('*')), key=lambda f: f.stat().st_ctime, reverse=True)[0]
    return last_run_dir / 'points'


def read_from_las(path):
    inFile = laspy.read(path)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    try:
        n0 = inFile.points.array["NormalX"]
        n1 = inFile.points.array["NormalY"]
        n2 = inFile.points.array["NormalZ"]
        normals = np.stack((n0,n1,n2)).T
    except:
        normals = None

    scanpos = inFile.points.array["point_source_id"]

    extra_dims = list(inFile.points.point_format.extra_dimension_names)
    if "Amplitude" in extra_dims:
        amp = inFile.points.array["Amplitude"]
    else:
        amp = None

    if "Deviation" in extra_dims:
        dev = inFile.points.array["Deviation"]
    else:
        dev = None
    return coords, normals, scanpos, amp, dev

def write_to_las(path, points, attrdict):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")

    for attrname in attrdict:
        try:
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

grid_x, grid_y = np.meshgrid(np.arange(0, 100, 1), np.arange(0, 100, 1))


tri = spatial.Delaunay(np.stack([grid_x.flatten(), grid_y.flatten()]).T)
face_ids = tri.simplices

faces_string = []
for fid, (p1, p2, p3) in enumerate(face_ids):
    faces_string.append(f"f {p1 + 1} {p2 + 1} {p3 + 1}\n")

phi = -120 * np.pi/180.
tfM = np.array([
    [np.cos(phi), 0, np.sin(phi)],
    [0, 1, 0],
    [-np.sin(phi), 0, np.cos(phi)],
])

output_files = []
output_tfM = ['time,alpha,beta,gamma,tx,ty,tz,ppm\n']

displs = 0.05 * (np.sin(np.linspace(-np.pi/2, np.pi/2, 40)) + 1)/2

for ep_id, ep_displMax in enumerate(tqdm.tqdm(displs)):
    zs = (grid_y-50) / np.max(grid_y-50) * ep_displMax

    vertice_string = []
    for x, y, z in zip(grid_x.flatten(), grid_y.flatten(), zs.flatten()):
        x, y, z = tfM @ np.array([[x], [y], [z]])
        vertice_string.append(f"v {float(x):.4f} {float(y):.4f} {float(z):.10f}\n")

    with open(outfile % ep_id, 'w') as o:
        o.writelines(vertice_string)
        o.writelines(faces_string)


    with open(SURVEY_FILE, 'w') as o:
        o.writelines(SURVEY % SCENE_FILE)
    with open(SCENE_FILE, 'w') as o:
        o.writelines(SCENE.format(meshfile=(outfile % ep_id)))

    p = subprocess.Popen([str(HELIOS_DIR / "run" / "helios.exe"), SURVEY_FILE, '--rebuildScene', '--lasOutput'], stdout=subprocess.PIPE,
                         cwd=OUTDIR)
    stdout = p.communicate()
    p.wait()
    # find output file
    output_file = find_playback_dir(SURVEY_FILE, OUTDIR) / "leg000_points.las"
    shutil.move(output_file, OUTDIR+r'\ep%02d.las' % ep_id)
    output_files.append(OUTDIR+r'\ep%02d.las' % ep_id)

    # generate transformation
    tf_params = np.random.multivariate_normal(mean=np.array([0,0,0,0,0,0,0]), cov=np.diag(np.square([
                                                                                     SIGMA_ALPHA, SIGMA_BETA, SIGMA_GAMMA,
                                                                                     SIGMA_X_Y_Z, SIGMA_X_Y_Z, SIGMA_X_Y_Z,
                                                                                     SIGMA_M])))
    output_tfM.append(f"{ep_id},"
                      f"{tf_params[0]:.10e},"
                      f"{tf_params[1]:.10e},"
                      f"{tf_params[2]:.10e},"
                      f"{tf_params[3]:.10e},"
                      f"{tf_params[4]:.10e},"
                      f"{tf_params[5]:.10e},"
                      f"{tf_params[6]:.10e}\n")


with open(OUTDIR +r'\transform.csv', 'w') as of:
    of.writelines(output_tfM)

with open(OUTDIR +r'\transformCOV.csv', 'w') as of:
    for line in np.diag(np.square([SIGMA_ALPHA, SIGMA_BETA, SIGMA_GAMMA,
                         SIGMA_X_Y_Z, SIGMA_X_Y_Z, SIGMA_X_Y_Z,
                         SIGMA_M])):
        of.write(','.join(['%.8e' % e for e in line]) + '\n')

shutil.rmtree(OUTDIR +r'\output')

coords, normals, scanpos, amp, dev = read_from_las(output_files[0])
normv = tfM @ np.array([0, 0, 1])

write_to_las(output_files[0], coords,
             {"NormalX": np.full((coords.shape[0]), normv[0]),
              "NormalY": np.full((coords.shape[0]), normv[1]),
              "NormalZ": np.full((coords.shape[0]), normv[2]),
              })