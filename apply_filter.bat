cd pointclouds
for /r %%i in (*.las) do (
echo %%i
pdal translate "%%i" ^
-o "%%~ni_gnd.las" ^
outlier smrf range  ^
--readers.las.use_eb_vlr="true" ^
--filters.outlier.method="statistical" ^
--filters.outlier.mean_k=8 --filters.outlier.multiplier=10.0 ^
--filters.smrf.ignore="Classification[7:7]"  ^
--filters.smrf.cell="0.5"  ^
--filters.smrf.slope="2"  ^
--filters.range.limits="Classification[2:2],Deviation[0:50]" ^
--writers.las.extra_dims="all" ^
--writers.las.scale_x="0.00025" ^
--writers.las.scale_y="0.00025" ^
--writers.las.scale_z="0.00025" ^
--writers.las.offset_x="auto" ^
--writers.las.offset_y="auto" ^
--writers.las.offset_z="auto" ^
--verbose 2
)
cd ..
