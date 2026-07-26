# Portable build for sycl-gpu-dsp.
#
# Auto-detects the two things that differ from machine to machine: the acpp
# (AdaptiveCpp) compiler and the CUDA toolkit. Override either on the command
# line or in the environment, e.g.
#
#   make                                   # detect everything, build all programs
#   make CUDA_PATH=/usr/local/cuda-12.6    # pin a specific CUDA (must match acpp)
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

CXX ?= g++

# Fail with a clear message (instead of a cryptic shell error) if a rule that
# needs acpp runs and detection came up empty. Expanded in recipes, so
# acpp-free targets (host tests, print-config, clean) still work without it.
require-acpp = $(if $(ACPP),,$(error acpp not found -- install AdaptiveCpp, \
  source env/acpp-env.sh, or pass ACPP=/path/to/acpp))

# --- Layout & flags ---------------------------------------------------------

SRC        := src
TESTS      := tests
BUILD      := build
TEST_BUILD := $(BUILD)/tests

ACPP_FLAGS    := -O3
SYCL_TARGET   := --acpp-targets=generic
CUDA_INCLUDES := -I$(CUDA_PATH)/include
CUDA_LIBS     := -L$(CUDA_PATH)/lib64 -lcufft -lcudart
HOST_FLAGS    := -O1 -std=c++17 -I$(SRC) -I$(TESTS)
HOST_OPT_FLAGS := -O3 -std=c++17 -I$(SRC)

# Every program compiles through acpp's `generic` target (JIT, fastest and
# most portable -- see doc/performance.md in the AdaptiveCpp repo). CUDA_PROGS
# differ only in that they call cuFFT directly and so need cufft/cudart on
# the include/link line; that's independent of the SYCL compilation target.
GENERIC_PROGS := 01_usm 02_window 03_dft 04_fft dedisp dmsearch radar_pulses
CUDA_PROGS    := stageB_cufft stageC_spectrogram iq2spectrogram
HOST_PROGS    := stageA_load
# Host-only but performance-sensitive (hundreds-of-millions-of-lines-scale
# CSV parsing) -- gets -O3 rather than HOST_PROGS' fast-compiling -O1.
HOST_OPT_PROGS := csv2sigmf

GENERIC_BINS  := $(addprefix $(BUILD)/,$(GENERIC_PROGS))
CUDA_BINS     := $(addprefix $(BUILD)/,$(CUDA_PROGS))
HOST_BINS     := $(addprefix $(BUILD)/,$(HOST_PROGS))
HOST_OPT_BINS := $(addprefix $(BUILD)/,$(HOST_OPT_PROGS))
ALL_BINS      := $(GENERIC_BINS) $(CUDA_BINS) $(HOST_BINS) $(HOST_OPT_BINS)

# Every header is a dependency of every program: they're small and shared, and
# a change to a constants header should rebuild everything that used it.
HEADERS := $(wildcard $(SRC)/*.hpp)

# --- Default target ---------------------------------------------------------

.PHONY: all
all: $(ALL_BINS)

# --- Build rules ------------------------------------------------------------

$(GENERIC_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(require-acpp)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) $< -o $@

$(CUDA_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(require-acpp)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) $< -o $@ \
	  $(CUDA_INCLUDES) $(CUDA_LIBS)

$(HOST_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(HOST_FLAGS) $< -o $@

$(HOST_OPT_BINS): $(BUILD)/%: $(SRC)/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(HOST_OPT_FLAGS) $< -o $@

$(BUILD) $(TEST_BUILD):
	mkdir -p $@

# --- Tests ------------------------------------------------------------------

HOST_TEST_SRCS := $(TESTS)/host/main.cpp \
                  $(TESTS)/host/test_sigmf_meta.cpp \
                  $(TESTS)/host/test_cli_util.cpp \
                  $(TESTS)/host/test_dsp_math.cpp \
                  $(TESTS)/host/test_csv_iq.cpp

.PHONY: host-tests gpu-tests tests
host-tests: $(TEST_BUILD)/host_tests
gpu-tests:  $(TEST_BUILD)/test_fft_vs_dft \
            $(TEST_BUILD)/test_window_kernel \
            $(TEST_BUILD)/test_radar_envelope \
            $(TEST_BUILD)/test_cufft_batch
tests: host-tests gpu-tests

$(TEST_BUILD)/host_tests: $(HOST_TEST_SRCS) $(HEADERS) | $(TEST_BUILD)
	$(CXX) $(HOST_FLAGS) $(HOST_TEST_SRCS) -o $@

$(TEST_BUILD)/test_fft_vs_dft $(TEST_BUILD)/test_window_kernel $(TEST_BUILD)/test_radar_envelope: \
    $(TEST_BUILD)/%: $(TESTS)/gpu/%.cpp $(HEADERS) | $(TEST_BUILD)
	$(require-acpp)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) -I$(TESTS) $< -o $@

$(TEST_BUILD)/test_cufft_batch: $(TESTS)/gpu/test_cufft_batch.cpp $(HEADERS) | $(TEST_BUILD)
	$(require-acpp)
	$(ACPP) $(ACPP_FLAGS) $(SYCL_TARGET) -I$(SRC) -I$(TESTS) \
	  $(CUDA_INCLUDES) $(CUDA_LIBS) $< -o $@

# --- Documentation -----------------------------------------------------------

# API reference generated from the Doxygen comments in src/*.hpp (the
# reusable headers get full @brief/@param/@return docs; the .cpp programs
# built from them get a @file brief only -- see Doxyfile). Output isn't
# committed; open build/docs/html/index.html after running this.
.PHONY: docs
docs:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not found -- install it to build docs"; exit 1; }
	doxygen Doxyfile
	@echo "docs: open build/docs/html/index.html"

# --- Introspection & cleanup ------------------------------------------------

# Machine-readable one-value queries used by the shell scripts.
.PHONY: print-acpp print-cuda-path print-config
print-acpp:      ; @echo $(ACPP)
print-cuda-path: ; @echo $(CUDA_PATH)

print-config:
	@echo "acpp       = $(ACPP)"
	@echo "CUDA_PATH  = $(CUDA_PATH)"
	@echo "CXX        = $(CXX)"

.PHONY: clean
clean:
	rm -rf $(BUILD)
