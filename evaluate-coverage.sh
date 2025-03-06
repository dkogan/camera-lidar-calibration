#!/bin/zsh

set -x
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

rm *.gcda 2>/dev/null || true

make clean
CCXXFLAGS=--coverage LDFLAGS=--coverage make -j

./test.sh $DIR

gcov -r -n clc.c lidar-segmentation.c
