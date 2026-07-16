# Portable build for sycl-gpu-dsp.
#
# Auto-detects the three things that differ from machine to machine: the acpp
# (AdaptiveCpp) compiler, the CUDA toolkit, and the GPU compute capability.
# Override any of them on the command line or in the environment, e.g.
#
#   make                                   # detect everything, build all programs
#   make CUDA_PATH=/usr/local/cuda-12.6    # pin a specific CUDA (must match acpp)
#   make SM_ARCH=sm_80                      # target a different GPU arch
#   make ACPP=/opt/acpp/bin/acpp            # use a specific acpp
#   make print-config                       # show what was detected, build nothing
#   make host-tests                         # build tests that need no GPU/acpp
#
# `make print-config` is the single source of truth the shell scripts query,
# so build flags live here and nowhere else.

# --- Toolchain detection ----------------------------------------------------

# acpp: prefer one already on PATH, else $ACPP_HOME/bin, else ~/adaptivecpp/bin.
ACPP_HOME ?= $(HOME)/adaptivecpp
ACPP ?= $(shell command -v acpp 2>/dev/null || \
                echo $(firstword $(wildcard $(ACPP_HOME)/bin/acpp /usr/bin/acpp)))

# CUDA toolkit: honor an explicit CUDA_PATH; else derive it from nvcc on PATH;
# else fall back to the conventional /usr/local/cuda. It must be the CUDA
# version acpp was built against, hence the override.
CUDA_PATH ?= $(shell d=`command -v nvcc 2>/dev/null`; \
                     if [ -n "$$d" ]; then dirname `dirname $$d`; \
                     else echo /usr/local/cuda; fi)

# GPU arch: ask the driver for the first device's compute capability
# (e.g. "7.5" -> sm_75); fall back to sm_75 (Turing) if nvidia-smi is absent.
SM_ARCH ?= $(shell cc=`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1`; \
                   if [ -n "$$cc" ]; then echo sm_`echo $$cc | tr -d '.'`; \
                   else echo sm_75; fi)

CXX ?= g++

# --- Layout & flags ---------------------------------------------------------

SRC        := src
TESTS      := tests
BUILD      := build
TEST_BUILD := $(BUILD)/tests

ACPP_FLAGS    := -O2
SYCL_TARGET   := --acpp-targets=generic
CUDA_TARGET   := --acpp-targets=cuda:$(SM_ARCH)
CUDA_INCLUDES := -I$(CUDA_PATH)/include
CUDA_LIBS     := -L$(CUDA_PATH)/lib64 -lcufft -lcudart
HOST_FLAGS    := -O1 -std=c++17 -I$(SRC) -I$(TESTS)

# Program groups by how they must be compiled.
GENERIC_PROGS := 01_usm 02_window 03_dft 04_fft dedisp dmsearch
CUDA_PROGS    := stageB_cufft stageC_spectrogram iq2spectrogram
HOST_PROGS    := stageA_load

GENERIC_BINS := $(addprefix $(BUILD)/,$(GENERIC_PROGS))
CUDA_BINS    := $(addprefix $(BUILD)/,$(CUDA_PROGS))
HOST_BINS    := $(addprefix $(BUILD)/,$(HOST_PROGS))
ALL_BINS     := $(GENERIC_BINS) $(CUDA_BINS) $(HOST_BINS)

# Every header is a dependency of every program: they're small and shared, and
# a change to a constants header should rebuild everything that used it.
HEADERS := $(wildcard $(SRC)/*.hpp)

# --- Default target ---------------------------------------------------------

.PHONY: all
all: $(ALL_BINS)

# --- Build rules ------------------------------------------------------------

$(GENERIC_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) $< -o $@

$(CUDA_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(ACPP) $(ACPP_FLAGS) $(CUDA_TARGET) -I$(SRC) $< -o $@ \
	  $(CUDA_INCLUDES) $(CUDA_LIBS)

$(HOST_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(HOST_FLAGS) $< -o $@

$(BUILD) $(TEST_BUILD):
	mkdir -p $@

# --- Tests ------------------------------------------------------------------

HOST_TEST_SRCS := $(TESTS)/host/main.cpp \
                  $(TESTS)/host/test_sigmf_meta.cpp \
                  $(TESTS)/host/test_cli_util.cpp \
                  $(TESTS)/host/test_dsp_math.cpp

.PHONY: host-tests gpu-tests tests
host-tests: $(TEST_BUILD)/host_tests
gpu-tests:  $(TEST_BUILD)/test_fft_vs_dft \
            $(TEST_BUILD)/test_window_kernel \
            $(TEST_BUILD)/test_cufft_batch
tests: host-tests gpu-tests

$(TEST_BUILD)/host_tests: $(HOST_TEST_SRCS) $(HEADERS) | $(TEST_BUILD)
	$(CXX) $(HOST_FLAGS) $(HOST_TEST_SRCS) -o $@

$(TEST_BUILD)/test_fft_vs_dft $(TEST_BUILD)/test_window_kernel: \
    $(TEST_BUILD)/%: $(TESTS)/gpu/%.cpp $(HEADERS) | $(TEST_BUILD)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) -I$(TESTS) $< -o $@

$(TEST_BUILD)/test_cufft_batch: $(TESTS)/gpu/test_cufft_batch.cpp $(HEADERS) | $(TEST_BUILD)
	$(ACPP) $(ACPP_FLAGS) $(CUDA_TARGET) -I$(SRC) -I$(TESTS) \
	  $(CUDA_INCLUDES) $(CUDA_LIBS) $< -o $@

# --- Introspection & cleanup ------------------------------------------------

# Machine-readable one-value queries used by the shell scripts.
.PHONY: print-acpp print-cuda-path print-sm-arch print-config
print-acpp:      ; @echo $(ACPP)
print-cuda-path: ; @echo $(CUDA_PATH)
print-sm-arch:   ; @echo $(SM_ARCH)

print-config:
	@echo "acpp       = $(ACPP)"
	@echo "CUDA_PATH  = $(CUDA_PATH)"
	@echo "SM_ARCH    = $(SM_ARCH)"
	@echo "CXX        = $(CXX)"

.PHONY: clean
clean:
	rm -rf $(BUILD)
