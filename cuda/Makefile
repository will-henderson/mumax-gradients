# Builds mumax3 cuda kernels and create GO wrappers for the compute capabilities listed in CUDA_CC. 
# If CUDA_CC is not defined, then CUDA_CC is set to "30".
#
# Examples:
#
#    make
#    make CUDA_CC=70
#    make CUDA_CC="30 32 35 37 50 52 53 60 61 62 70"
#    make NVCC=/usr/local/cuda-9.2/bin/nvcc CUDA_CC=72
#
# Different CUDA versions support different compute capabilities, as shown in the list below.
# CUDA SDK  7.5 support for compute capability 20 21 30 32 35 37 50 52 53
# CUDA SDK  8.0 support for compute capability 20 21 30 32 35 37 50 52 53 60 61 62
# CUDA SDK  9.0 support for compute capability       30 32 35 37 50 52 53 60 61 62 70
# CUDA SDK  9.1 support for compute capability       30 32 35 37 50 52 53 60 61 62 70
# CUDA SDK  9.2 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72
# CUDA SDK 10.0 support for compute capability       30 32 35 37 50 52 53 60 61 62 70 72 75

SHELL = /bin/bash

# When CUDA_CC is not an environment variable and is not set on the command line, use compute capability 3.0
ifeq ($(CUDA_CC),)
    CUDA_CC = 30
endif

NVCC = nvcc
NVCCFLAGS = -std c++03 --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx

CUDAFILES := $(wildcard *.cu)
WRAPPERS := $(CUDAFILES:.cu=_wrapper.go)


.PHONY: all wrappers fatbin clean realclean

all: wrappers
	go install -v

wrappers: $(WRAPPERS)

%_wrapper.go: %.cu cuda2go
	@ rm -f $(basename $<)*.ptx
	@ for cc in $(CUDA_CC); do \
		echo $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
		     $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
	done
	@ ./cuda2go $< > /dev/null
	@ gofmt -w -s -l $@ > /dev/null

cuda2go: cuda2go.go
	go build $<

fatbin:
	$(MAKE) realclean
	$(MAKE) CUDA_CC="30 32 35 37 50 52 53 60 61 62 70 72 75"

clean:
	rm -vf *.ptx

realclean:
	rm -vf *_wrapper.go *.ptx cuda2go
