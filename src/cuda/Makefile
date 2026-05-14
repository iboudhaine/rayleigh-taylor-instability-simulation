NVCC   ?= nvcc
NVFLAGS ?= -O3 -std=c++14 -Xcompiler "-Wall -Wextra"

run: run.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f run

.PHONY: clean
