include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := camera_lidar_calibration
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += \
  -lm

CFLAGS    += --std=gnu99 -ggdb3
CCXXFLAGS += -Wno-missing-field-initializers

LIB_SOURCES += point_segmentation.c

camera_lidar_calibration_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
camera_lidar_calibration_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

_camera_lidar_calibration$(PY_EXT_SUFFIX): camera_lidar_calibration_pywrap.o libcamera_lidar_calibration.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

DIST_PY3_MODULES := _camera_lidar_calibration$(PY_EXT_SUFFIX)

all: _camera_lidar_calibration$(PY_EXT_SUFFIX)

include $(MRBUILD_MK)/Makefile.common.footer
