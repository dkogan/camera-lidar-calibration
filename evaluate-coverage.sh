#!/bin/zsh

set -x
set -e

rm *.gcda 2>/dev/null || true

make clean
CCXXFLAGS=--coverage LDFLAGS=--coverage make -j

./test.sh

gcov -r -n clc.c lidar-segmentation.c
