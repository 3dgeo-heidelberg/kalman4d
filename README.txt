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

The tf_helper_T.py is a script to transform a 6x6 covariance matrix (rigid Helmert-Transform with fixed scale) to a full 12x12 covariance matrix as required by M3C2-EP. It follows the nomenclature given in the formula collecton of Joeckel, R., Gruber, F. J. (2020). Formelsammlung für das Vermessungswesen. Germany: Springer Fachmedien Wiesbaden.

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

3) kmeans_cluster.py
This script takes the smoothed change values themselves and uses them for clustering. Again, input/output/options are at the end of the file.

4) kromer_m3c2.py
This script runs on the same inputs as kalman_m3c2.py, but implements temporal median smoothing or linear interpolation. The output can be changed in Line 115/116.

5) Any script with 'plot' is used to recreate the figures in the manuscript.

7) For the synthetic example, the HELIOS++ Software is required: https://github.com/3dgeo-heidelberg/helios/releases/tag/v1.1.0
Then, synth_change.py will create a point cloud time series, which can be used for M3C2-EP using multitemp_change_synth.py, and then processed in the kalman filter with kalman_m3c2_synth.py.
Similarly, linear interpolation or temporal median filtering can be carried out with synth_kromer_m3c2.py. Results are then quantified with kalman_synth_residuals.py and plotted with plot_kalman_qchoice_synth.py.
