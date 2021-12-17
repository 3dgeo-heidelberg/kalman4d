README
======

This folder contains the code used to run M3C2-EP and Kalman Filtering for the submitted manuscript "Full 4D Change Analysis of Topographic Point Cloud Time Series using Kalman Filtering".

Prerequisites:
--------------
A python >=3.8 installation (I suggest miniconda from https://docs.conda.io/en/latest/miniconda.html)
The following packages:
- matplotlib
- numpy
- pandas
- tqdm
- tsfresh
- scikit-learn
- filterpy
- laspy<2.0.0
- pdal

The ICPmat2CxxMat.exe is a MatLAB executable to transfer the covariance information created by OPALS ICP (https://opals.geo.tuwien.ac.at/html/stable/ModuleICP.html) to a .mat-File that is readable by python/numpy.

The tf_helper.py is a script to transform a 6x6 covariance matrix (rigid Helmert-Transform with fixed scale) to a full 12x12 covariance matrix as required by M3C2-EP.

apply_filter.bat is a batch file applying the ground point filter used in pre-preparation of the data. It relies on PDAL.

Running the scripts:
--------------------

1) multitemp_change.py
The M3C2-EP point cloud distance calculation is carried out using the script 
multitemp_change.py
Input parameters are found at the end of the file:

outFile is the file name of the result point cloud

tiles_pre is a list of the input point cloud las files (not transformed to align)
CxxFiles is a list of the result .mat Files from ICPmat2CxxMat.exe. An example .mat File is also supplied.
p_dates is a list of datetime objects
The order of elements in the last three lists has to match.

core_point_file is the input core point cloud las file.

This script has multiprocessing enabled and can run on multiple threads in parallel. Adjust the variable NUM_THREADS and NUM_BLOCKS to fit the power of your machine. For 71 epochs, 200.000 core points and a 64-core machine with 256GB RAM, 10 threads and 50 blocks were a sensible solution.

2) kalman_m3c2.py
The second script is kalman_m3c2.py, and also has input parameters at the end of the file:
infile is a list/iterable of input point clouds, should the number of epochs require multiple las-Files to be written (blocksize in multitemp_change.py)
ref_epoch is the UNIX timestamp of the reference epoch (written to stdout by multitemp_change.py)
Q_vals is a list of sigma values for the discrete white noise used to model state covariance
outfile is the output point cloud. Note that if too many epochs are exported, laspy will fail to write.

3) kalman_cluster.py
This script is used to take the attributes exported by kalman_m3c2.py and apply a clustering method. Again, inputs/outputs
and the parameters to be used for clustering are defined at the end of the file.

4) GME_on_numpy.py
As the number of parameters extracted with tsfresh is too large for a las-File, we export them to a .npy file. This script does the
same as kalman_cluster.py but takes a numpy file as input instead. The las-File is still required for the coordinates.

5) kmeans_cluster.py
This script takes the smoothed change values themselves and uses them for clustering. Again, input/output/options are at the end of the file.

The workflow looks like this:

                        ┌────────────────┐
                      ┌─┴───────────────┐│
                      │                 ││       ┌────────────────────────┐
                      │   epoch-wise    ││       │                        │
                      │   point clouds  ││ ────► │ 1)multitemp_change.py  │
                      │   & trafos      ││       │                        │
                      │                 ├┘       │                        │
                      └─────────────────┘        └───────────┬────────────┘
                                                             │
                                                 ┌───────────▼────────────┐
                                                 │                        │
                                                 │  point cloud with      │
                                                 │   change per epoch     │
                                                 │                        │
                                                 └───────────┬────────────┘
                                                             │
                                                 ┌───────────▼────────────┐
                                                 │                        │
                                                 │2 ) kalman_m3c2.py      │
                                                 │                        │
                                                 │                        │
                                                 └──┬────┬────────────────┘
                                                    │    │
                                 ┌──────────────────┘    │
                                 ▼                       ▼
                         ┌─────────────────┐         ┌─────────────────┐
                         │                 │         │                 │
                         │ point cloud     │         │ .npy file with  │
                         │ with smoothed   │         │ additional      │
                         │ timeseries      │         │ features        │
                         │                 │         │                 │
                         └──┬─────────┬────┴──────┐  └────────┬────────┘
                            │         │           ├──────┐    │
         ┌──────────────────▼┐  ┌─────▼───────────┴─┐  ┌─▼────▼──────────┐
         │                   │  │                   │  │                 │
         │                   │  │                   │  │                 │
         │3)kmeans_cluster.py│  │4)kmeans_cluster.py│  │5)GME_on_numpy.py│
         │                   │  │                   │  │                 │
         │                   │  │                   │  │                 │
         └───────────┬───────┘  └───────┬───────────┘  └────┬────────────┘
                     │                  │                   │
                     │                  │                   │
                     │                  │                   │
                     └───────►┌─────────▼──────┐◄───────────┘
                              │                │
                              │ cluster result │
                              │                │
                              └────────────────┘


