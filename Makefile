include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := clc
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += -lm -lmrcal -ldogleg -lopencv_core -l mrgingham

CFLAGS    += --std=gnu99 -ggdb3
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable

# I need the bleeding-edge mrcal
MRCAL     ?= /home/dima/projects/mrcal
CCXXFLAGS += -I$(MRCAL)/..
LDFLAGS   += -L$(MRCAL) -Wl,-rpath=$(MRCAL)

CFLAGS += -I/usr/include/suitesparse


LIB_SOURCES += lidar-segmentation.c clc.c mrgingham-c-bridge.cc

clc-pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
clc-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

_clc$(PY_EXT_SUFFIX): clc-pywrap.o libclc.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

mrgingham-c-bridge.o: CXXFLAGS += -I/usr/include/mrgingham -I/usr/include/opencv4/

DIST_PY3_MODULES := _clc$(PY_EXT_SUFFIX)

all: _clc$(PY_EXT_SUFFIX)

include $(MRBUILD_MK)/Makefile.common.footer
