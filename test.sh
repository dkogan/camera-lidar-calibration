#!/bin/zsh

set -e

DIR=$1
if { [[ -z "$DIR" ]] } {
    echo "Usage: $0 DIRECTORY_TEST_DATA" > /dev/stderr;
    exit 1;
}
if {! [[ -d "$DIR" ]]} {
    echo "Usage: $0 DIRECTORY_TEST_DATA" > /dev/stderr;
    echo "'$DIR' is not a directory I can read" > /dev/stderr;
    exit 1;
}

make all tests || exit 1

Nfailed=0

echo "====== running test-bitarray"
./test-bitarray || Nfailed=$((Nfailed+1))


echo "====== running test-transformation-uncertainty.py"
models=($DIR/2024-calibration/results-intrinsics/multisense/{left,right,aux}_camera/camera-0-SPLINED.cameramodel)
dump=/tmp/clc-context.pickle
bag_glob="$DIR/2024-calibration/images-and-lidar-*.bag"

./fit.py \
  --rt-vehicle-lidar0 0.01,0.02,0.03,-5.1,0.2,0.3 \
  --dump $dump \
  --topic  /vl_points_0,/vl_points_1,/vl_points_2,/multisense/left/image_mono,/multisense/right/image_mono,/multisense/aux/image_color \
  --bag $bag_glob \
  $models && \
./test-transformation-uncertainty.py \
  --topic /vl_points_1,/multisense/right/image_mono \
  --isector 3 \
  --Nsamples 40 \
  --context $dump \
|| Nfailed=$((Nfailed+1))

./lidar-segmentation-auto-test.py $DIR \
|| Nfailed=$((Nfailed+1))


echo "===== SUMMARY: Nfailed=$Nfailed"
exit $Nfailed
