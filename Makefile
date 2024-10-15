include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := camera_lidar_calibration
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += \
  -lm

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers

LIB_SOURCES += point_segmentation.c
point-segmentation.o: point_segmentation.usage.h
%.usage.h: %.usage
	< $^ sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.usage.h


BIN_SOURCES += point-segmentation-test.c

camera_lidar_calibration_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
camera_lidar_calibration_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

camera_lidar_calibration$(PY_EXT_SUFFIX): camera_lidar_calibration_pywrap.o libcamera_lidar_calibration.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

DIST_PY3_MODULES := camera_lidar_calibration$(PY_EXT_SUFFIX)

all: camera_lidar_calibration$(PY_EXT_SUFFIX)

include $(MRBUILD_MK)/Makefile.common.footer
