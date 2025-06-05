include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := clc
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += -lm -lmrcal -ldogleg -lopencv_core -lmrgingham -lopencv_calib3d -lopencv_core

CFLAGS    += --std=gnu99 -ggdb3
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable

# The code currently has some "unsigned" things that will become signed in the
# near future. Making these not unsigned will make the warnings go away. In the
# mean time, I simply silence them
CCXXFLAGS += -Wno-sign-compare

# There're a few commented-out chunks of code that throw benign warnings that I
# silence here
CCXXFLAGS += -Wno-comment

CFLAGS += -I/usr/include/suitesparse


LIB_SOURCES += lidar-segmentation.c clc.c mrgingham-c-bridge.cc opencv-c-bridge.cc

clc-pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
clc-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

clc/_clc$(PY_EXT_SUFFIX): clc-pywrap.o libclc.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $^ -o $@

mrgingham-c-bridge.o: CXXFLAGS += -I/usr/include/mrgingham -I/usr/include/opencv4/
opencv-c-bridge.o:    CXXFLAGS += -I/usr/include/opencv4/

DIST_PY3_MODULES := clc

all: clc/_clc$(PY_EXT_SUFFIX)



# rules to build the tests. The tests are conducted via test.sh
BIN_SOURCES += \
	test/test-bitarray.c

$(BIN_SOURCES:.c=.o): CFLAGS += -I.

include $(MRBUILD_MK)/Makefile.common.footer
