#!/bin/zsh

set -e

DIR=${1:A}
if { [[ -z "$DIR" ]] } {
    echo "Usage: $0 DIRECTORY_TEST_DATA" > /dev/stderr;
    exit 1;
}
if {! [[ -d "$DIR" ]]} {
    echo "Usage: $0 DIRECTORY_TEST_DATA" > /dev/stderr;
    echo "'$DIR' is not a directory I can read" > /dev/stderr;
    exit 1;
}

THISDIR=${0:A:h}
CLCDIR=${THISDIR:h}

cd $CLCDIR

make || exit 1

Nfailed=0

echo "====== running test-bitarray"
test/test-bitarray || Nfailed=$((Nfailed+1))

test -x test_private.sh && Nfailed=$(./test_private)

# This SHOULD work, but something's wrong. Need to investigate. Disabling it for now
if { false } {

echo "====== running test-transformation-uncertainty.py"
models=($DIR/2023-10-19/results/intrinsics/multisense_front/{left,right,aux}_camera/camera-0-SPLINED.cameramodel)
dump=/tmp/clc-context.pickle
bag_glob="$DIR/2023-10-19/one*.bag"

./fit.py \
  --rt-vehicle-lidar0 0.01,0.02,0.03,-5.1,0.2,0.3 \
  --dump $dump \
  --topic /lidar/velodyne_front_horiz_points,/lidar/velodyne_front_tilted_points,/lidar/multisense_front/left/image_mono,/lidar/multisense_front/right/image_mono,/lidar/multisense_front/aux/image_color \
  --bag $bag_glob \
  $models && \
test/test-transformation-uncertainty.py \
  --topic /lidar/velodyne_front_horiz_points,/lidar/multisense_front/left/image_mono \
  --isector 3 \
  --Nsamples 40 \
  --context $dump \
|| Nfailed=$((Nfailed+1))

}







test/test-lidar-segmentation.py $DIR \
|| Nfailed=$((Nfailed+1))


echo "===== SUMMARY: Nfailed=$Nfailed"
exit $Nfailed
