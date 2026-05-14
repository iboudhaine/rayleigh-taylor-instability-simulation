CC      ?= gcc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L
LDFLAGS ?= -lm

run: run.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f run

.PHONY: clean
