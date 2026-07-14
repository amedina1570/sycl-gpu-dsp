# sycl-gpu-dsp

GPU-accelerated digital signal processing in SYCL (AdaptiveCpp), targeting NVIDIA
GPUs via the CUDA backend. Built as a hands-on progression from SYCL fundamentals
to a complete spectral pipeline that recovers and removes interstellar dispersion
from real radio-astronomy data.

## Highlights

- USM memory management, in-order queues, cooperative work-items, local memory
- A hand-written radix-2 Cooley-Tukey FFT, validated against a naive DFT
- A batched spectral pipeline using **cuFFT via SYCL interop**
  (`AdaptiveCpp_enqueue_custom_operation` + `cufftSetStream`)
- Incoherent dedispersion and a blind **dispersion-measure (DM) search**

## Quick start: IQ file -> spectrogram image

`iq2spectrogram` combines stage A/C (load, window, batched cuFFT) and the
Python viewer into one command: feed it a SigMF `ci16_le`/`cf32_le` IQ file,
get a spectrogram PNG back. Sample rate, center frequency, and datatype are
auto-read from a `.sigmf-meta` sidecar when present.

    acpp -O2 --acpp-targets=cuda:sm_75 src/iq2spectrogram.cpp -o build/iq2spectrogram \
      -I/usr/local/cuda-12.6/include -L/usr/local/cuda-12.6/lib64 -lcufft -lcudart

    ./build/iq2spectrogram path/to/recording.sigmf-data
    # -> writes <stem>_spectrogram.{bin,json,png} next to your working directory

Or use the wrapper, which builds it for you (only when the source has
changed) before running it:

    ./run_iq2spectrogram.sh path/to/recording.sigmf-data [options...]

Options: `--nfft`, `--hop`, `--fs`, `--fc`, `--datatype`, `-o/--out PREFIX`,
`--no-plot` (skip PNG rendering), `--viewer PATH`, `--python PATH`,
`--chunk-mb` (see below). Run with `--help` for the full list.

### Large files

`iq2spectrogram` streams the input in bounded chunks rather than loading the
whole file and building one giant batched cuFFT call: it sizes chunks off
`--chunk-mb` (default 256 MB of GPU memory per chunk; raise it for fewer,
larger chunks and more throughput if your GPU has room, lower it if you're
memory-constrained) and appends each chunk's rows straight to the output
`.bin`, so host RAM and GPU VRAM usage stay flat regardless of file size — a
multi-hundred-GB recording processes the same way a 10 MB one does, just
slower. The viewer (`view/view_spec.py`) memory-maps the `.bin` and
max-hold-decimates it down to the figure's pixel width instead of loading it
whole, so plotting a huge spectrogram doesn't itself blow out RAM.

## Demonstration: Crab pulsar giant pulse

Using a 0.2 s, 20 Msps recording of a Crab pulsar giant pulse from the Dwingeloo
Radio Telescope (SigMF, `ci16_le`, centered at 410 MHz), the pipeline:

1. Loads the raw complex int16 IQ
2. Frames it (8192-pt FFT, 75% overlap), applies a Hann window, and runs a
   batched cuFFT to build a spectrogram
3. Recovers the dispersion sweep of the giant pulse across the band
4. Dedisperses to collapse the sweep into a single pulse
5. Runs a DM search, independently recovering **DM ~= 56.75 pc/cm^3**
   (literature value for the Crab: 56.7)

Data: Crab giant pulse, Stichting CAMRAS / M. Fine & T. J. Dijkema,
Zenodo DOI [10.5281/zenodo.13143544](https://doi.org/10.5281/zenodo.13143544),
CC BY-SA 4.0. Not included in this repo — download separately.

## Software requirements

- **Linux** (native or WSL2) — the examples below assume Ubuntu 24.04
- **NVIDIA GPU + driver** with a CUDA compute capability matching your
  target (this repo defaults to `sm_75`, Turing); adjust for your card
- **CUDA Toolkit** (`nvcc`, `cufft`, `cudart`) — developed against 12.6
- **[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)** built
  against LLVM 18 + CUDA 12.6, providing the `acpp` SYCL compiler
- **Python 3** with `numpy` and `matplotlib` — only needed to render
  spectrogram/dedispersion/DM-search PNGs via `view/*.py`
- **g++** with C++17 support — only needed to build the host-side unit
  tests (`tests/host/`); GPU tests and all `acpp`-compiled programs don't
  need it

## Build environment

- WSL2 (Ubuntu 24.04), NVIDIA RTX 2070 (Turing, sm_75)
- AdaptiveCpp built against LLVM 18 + CUDA 12.6
- NVIDIA driver on Windows; CUDA toolkit (not driver) inside WSL

Activate the toolchain:

    source env/acpp-env.sh

## Programs

| File | Description |
|------|-------------|
| `src/01_usm.cpp`        | USM allocations, in-order queue, explicit transfers |
| `src/02_window.cpp`     | Index-driven Hann window, device-side math |
| `src/03_dft.cpp`        | Naive O(N^2) DFT — correctness baseline |
| `src/04_fft.cpp`        | Single-workgroup radix-2 FFT (local memory + barriers) |
| `src/stageA_load.cpp`   | SigMF `ci16_le` loader + sanity stats |
| `src/stageB_cufft.cpp`  | cuFFT interop skeleton (synthetic data) |
| `src/stageC_spectrogram.cpp` | Full pipeline: IQ -> windowed batched cuFFT -> spectrogram |
| `src/iq2spectrogram.cpp` | **Combined CLI**: any IQ file in -> spectrogram PNG out (stages A+C+viewer) |
| `src/dedisp.cpp`        | Incoherent dedispersion + pulse profile |
| `src/dmsearch.cpp`      | Blind DM search maximizing pulse SNR |
| `view/*.py`             | Matplotlib viewers for the outputs |

## Compiling

Pure-SYCL programs (generic JIT target):

    acpp -O2 --acpp-targets=generic src/01_usm.cpp -o build/01_usm

cuFFT-interop programs (integrated CUDA target + cuFFT link):

    acpp -O2 --acpp-targets=cuda:sm_75 src/stageC_spectrogram.cpp -o build/stageC \
      -I/usr/local/cuda-12.6/include \
      -L/usr/local/cuda-12.6/lib64 -lcufft -lcudart

Adjust `sm_75` and the CUDA path for your GPU/toolkit.

## Testing

Shared host-side logic (SigMF metadata parsing, CLI validation, dispersion
math, SNR scoring) and GPU kernels (Hann window, radix-2 FFT vs. naive DFT,
batched cuFFT interop) are split into reusable headers in `src/` and covered
by a [doctest](https://github.com/doctest/doctest)-based suite in `tests/`:

    ./tests/run_tests.sh          # host tests + GPU tests (needs acpp + GPU)
    ./tests/run_tests.sh --host   # host tests only, no acpp/GPU required

Host tests build with plain `g++` and run anywhere. GPU tests build with
`acpp` (`source env/acpp-env.sh` first) and execute real SYCL/cuFFT kernels
on the GPU, matching the compile flags used above.

`tests/test_chunking.sh` is a separate end-to-end smoke test for
`iq2spectrogram`'s chunked/streaming processing (large-file support): it
generates a synthetic IQ file, runs it once with the default `--chunk-mb`
and once forced into many small chunks, and checks the two runs produce
byte-identical `.bin` output, that the recovered tone lands in the expected
bin, and that `view_spec.py` can plot the result. It confirms chunking
doesn't change *what* gets computed, not that the DSP math itself is
correct — that's what the doctest GPU suite above is for.

    ./tests/test_chunking.sh      # needs acpp + GPU, ~1-2 min

## License

MIT (code). The Crab pulsar dataset is CC BY-SA 4.0 and is not redistributed here.
