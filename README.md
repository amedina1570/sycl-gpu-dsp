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

## License

MIT (code). The Crab pulsar dataset is CC BY-SA 4.0 and is not redistributed here.
