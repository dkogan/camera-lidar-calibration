include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := camera_lidar_calibration
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += \
  -lm

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers

LIB_SOURCES += point_segmentation.c eig.c
point-segmentation.o: point_segmentation.usage.h
%.usage.h: %.usage
	< $^ sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.usage.h


BIN_SOURCES += point-segmentation-test.c

include $(MRBUILD_MK)/Makefile.common.footer
