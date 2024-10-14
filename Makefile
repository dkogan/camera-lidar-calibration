all: point-segmentation
.PHONY: all


LDLIBS = -lm
CFLAGS = -Wall -Wextra -ggdb3 -O0

CFLAGS += -Wno-unused-parameter -Wno-unused-function

%: %.o
	gcc $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	gcc -c $(CFLAGS) -o $@ $<



point-segmentation: eig.o
point-segmentation.o: point_segmentation.usage.h

# Text-include rules. I construct these from plain ASCII files to handle line
# wrapping
%.usage.h: %.usage
	< $^ sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.usage.h

.SECONDARY:
