# sycl-gpu-dsp

GPU-accelerated I/Q signal processing in SYCL (AdaptiveCpp), targeting NVIDIA
GPUs via the CUDA backend. At its core is a general-purpose pipeline that
turns any SigMF IQ recording (`ci16_le`/`cf32_le`, from a few MB to
multi-hundred-GB captures — see "Large files" below) into a windowed,
batched-FFT spectrogram, built up from SYCL fundamentals (USM, kernels,
local memory) through a hand-written FFT and cuFFT interop. A worked
radio-astronomy example — interstellar dedispersion of a Crab pulsar giant
pulse — demonstrates the pipeline end-to-end on a real capture.

## Highlights

- USM memory management, in-order queues, cooperative work-items, local memory
- A hand-written radix-2 Cooley-Tukey FFT, validated against a naive DFT
- A general **IQ file -> spectrogram** pipeline using **cuFFT via SYCL
  interop** (`AdaptiveCpp_enqueue_custom_operation` + `cufftSetStream`),
  streamed in bounded chunks so file size isn't limited by host/GPU memory
- A worked example built on top: incoherent dedispersion and a blind
  **dispersion-measure (DM) search**, applied to a pulsar recording
- GPU-accelerated **RADAR pulse-train detection** (width/PRI/PRF/duty
  cycle) for pulsed captures

## Quick start: IQ file -> spectrogram image

`iq2spectrogram` combines stage A/C (load, window, batched cuFFT) and the
Python viewer into one command: feed it a SigMF `ci16_le`/`cf32_le` IQ file,
get a spectrogram PNG back. Sample rate, center frequency, and datatype are
auto-read from a `.sigmf-meta` sidecar when present.

    make iq2spectrogram        # auto-detects acpp, CUDA, and your GPU's arch

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

### Quick-look: FFT + time domain + I/Q

For sanity-checking a capture without running the full spectrogram
pipeline, `view/view_iq_snapshot.py` reads a short segment straight from a
raw SigMF IQ file and plots three panels: FFT spectrum, time-domain I/Q
traces, and an I/Q (constellation) scatter. Same `.sigmf-meta`
auto-detection as `iq2spectrogram`.

    python3 view/view_iq_snapshot.py path/to/recording.sigmf-data \
      --offset 0 --nsamp 4096
    # -> writes <stem>_snapshot.png

### RADAR-like pulse trains

`view/view_radar_pulses.py` extracts pulse parameters from a pulsed
capture: it reads a segment (default 5ms, long enough to span several
repetitions — override with `--duration` or `--nsamp`), detects pulses on
the amplitude envelope (threshold at the midpoint between the noise floor
and peak by default, `--threshold-frac` to tune), and reports pulse width,
PRI, PRF, and duty cycle. It reuses `view_iq_snapshot`'s loader and FFT
computation, swapping the raw I/Q time trace for an envelope plot with
detected pulses shaded — a dense multi-pulse I/Q trace isn't readable, the
envelope is what actually shows the pulse train.

    python3 view/view_radar_pulses.py path/to/recording.sigmf-data \
      --duration 5e-3
    # -> writes <stem>_radar.png and <stem>_radar.json (extracted parameters)

For longer segments than comfortably fit in NumPy, `radar_pulses` is a
GPU-accelerated port of the same detection logic (envelope + boxcar
smoothing on SYCL, thresholding/edge-pairing/stats on host), same CLI
surface, same `.sigmf-meta` auto-detection:

    make radar_pulses
    ./build/radar_pulses path/to/recording.sigmf-data --duration 5e-3
    # -> writes <stem>_radar.json and <stem>_radar_envelope.bin

    python3 view/view_radar_pulses.py <stem>_radar.json
    # -> plots from the precomputed result instead of recomputing in NumPy
    #    (re-reads only a small I/Q slice from the original file)

## Worked example: Crab pulsar giant pulse

`iq2spectrogram` itself is data-agnostic — point it at any SigMF IQ capture
(a Wi-Fi packet, a cellular downlink, whatever) and it produces a
spectrogram. This example instead runs the full radio-astronomy chain on
top of that pipeline: a 0.2 s, 20 Msps recording of a Crab pulsar giant
pulse from the Dwingeloo Radio Telescope (SigMF, `ci16_le`, centered at 410
MHz).

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

## Building

Everything builds through a single auto-detecting Makefile, so the same
commands work on any machine with the toolchain below — no paths to edit:

    make                  # build every program into build/
    make iq2spectrogram   # just the spectrogram pipeline
    make host-tests       # tests that need no GPU/acpp
    make print-config     # show the detected acpp / CUDA path
    make clean

The Makefile auto-detects two machine-specific things and lets you override
either of them (they must be consistent with each other — in particular
`CUDA_PATH` has to be the CUDA release your AdaptiveCpp was built against).
Everything compiles through acpp's `generic` (JIT) target, so there's no GPU
arch to pin ahead of time:

| Variable | Detected from | Override example |
|----------|---------------|------------------|
| `ACPP`      | `acpp` on `PATH`, else `$ACPP_HOME/bin` (default `~/adaptivecpp`) | `make ACPP=/opt/acpp/bin/acpp` |
| `CUDA_PATH` | the `nvcc` on `PATH`, else `/usr/local/cuda` | `make CUDA_PATH=/usr/local/cuda-12.6` |

`source env/acpp-env.sh` first if `acpp` isn't already on your `PATH` (it
adds `$ACPP_HOME/bin`; override `ACPP_HOME` for a non-default install).

## Software requirements

- **Linux** (native or WSL2) — the examples below assume Ubuntu 24.04. On
  WSL2, the NVIDIA driver lives on the Windows side; install the CUDA
  *toolkit* (not another driver) inside WSL
- **NVIDIA GPU + driver** — no specific compute capability to target ahead
  of time; the build uses acpp's `generic` JIT target, which compiles for
  whatever device it finds at runtime
- **CUDA Toolkit** (`nvcc`, `cufft`, `cudart`) — developed against 12.6
- **[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)** built
  against LLVM 18 + CUDA 12.6, providing the `acpp` SYCL compiler
- **Python 3** with `numpy` and `matplotlib` — only needed to render
  spectrogram/dedispersion/DM-search PNGs via `view/*.py`
- **g++** with C++17 support — only needed to build the host-side unit
  tests (`tests/host/`); GPU tests and all `acpp`-compiled programs don't
  need it

## Programs

**General-purpose IQ -> spectrogram pipeline** (works on any SigMF capture):

| File | Description |
|------|-------------|
| `src/stageA_load.cpp`   | SigMF `ci16_le` loader + sanity stats |
| `src/stageB_cufft.cpp`  | cuFFT interop skeleton (synthetic data) |
| `src/stageC_spectrogram.cpp` | Fixed-size demo pipeline: IQ -> windowed batched cuFFT -> spectrogram |
| `src/iq2spectrogram.cpp` | **Combined CLI**: any IQ file in -> spectrogram PNG out, streamed in bounded chunks (stages A+C+viewer) |
| `view/view_spec.py`     | Matplotlib spectrogram viewer, memory-maps + downsamples large `.bin` output |
| `view/view_iq_snapshot.py` | Quick-look plot (FFT spectrum, time-domain I/Q, I/Q scatter) for a short segment of any SigMF IQ file |

**SYCL fundamentals** (the building blocks the pipeline above is written from):

| File | Description |
|------|-------------|
| `src/01_usm.cpp`        | USM allocations, in-order queue, explicit transfers |
| `src/02_window.cpp`     | Index-driven Hann window, device-side math |
| `src/03_dft.cpp`        | Naive O(N^2) DFT — correctness baseline |
| `src/04_fft.cpp`        | Single-workgroup radix-2 FFT (local memory + barriers) |

**Worked example: pulsar dedispersion**, built on top of the general pipeline's spectrogram output:

| File | Description |
|------|-------------|
| `src/dedisp.cpp`        | Incoherent dedispersion + pulse profile |
| `src/dmsearch.cpp`      | Blind DM search maximizing pulse SNR |
| `view/view_dedisp.py`, `view/view_dmsearch.py` | Matplotlib viewers for the dedispersion/DM-search outputs |

**Worked example: RADAR pulse-train detection**:

| File | Description |
|------|-------------|
| `src/radar_pulses.cpp`  | GPU-accelerated envelope + boxcar smoothing (SYCL), threshold/edge-detection/stats (host); width/PRI/PRF/duty cycle |
| `view/view_radar_pulses.py` | Matplotlib viewer -- either all-NumPy standalone on a raw IQ file, or plots `radar_pulses`' precomputed `.json`/`.bin` output |

## Compiling

Build any single program by name — every program compiles through the same
`generic` SYCL target; the Makefile just also links `cufft`/`cudart` for the
three that call cuFFT directly (`stageB_cufft`, `stageC_spectrogram`,
`iq2spectrogram`):

    make build/01_usm
    make build/stageC_spectrogram

See `make print-config` for the flags it resolved, and the [Building](#building)
section above for overriding the detected acpp / CUDA path.

## Testing

Shared host-side logic (SigMF metadata parsing, CLI validation, dispersion
math, SNR scoring, pulse-train detection/stats) and GPU kernels (Hann
window, radix-2 FFT vs. naive DFT, batched cuFFT interop, pulse envelope +
smoothing) are split into reusable headers in `src/` and covered by a
[doctest](https://github.com/doctest/doctest)-based suite in `tests/`:

    ./tests/run_tests.sh          # host tests + GPU tests (needs acpp + GPU)
    ./tests/run_tests.sh --host   # host tests only, no acpp/GPU required

Host tests build with plain `g++` and run anywhere. GPU tests build with
`acpp` (`source env/acpp-env.sh` first) and execute real SYCL/cuFFT kernels
on the GPU, matching the compile flags used above.

`tests/test_chunking.sh` is a separate end-to-end smoke test for
`iq2spectrogram`'s chunked/streaming processing (large-file support): it
generates a synthetic IQ file, runs it once with the default `--chunk-mb`
and once forced into many small chunks, and checks that the two runs produce
byte-identical `.bin` output, that the recovered tone lands in the expected
bin, that a handful of frames match an independent numpy computation of the
same windowed FFT (straight from the raw file, bypassing the chunking logic
entirely), and that `view_spec.py` can plot the result. Byte-identical
wide-vs-narrow chunking only proves the two runs agree with *each other*; a
bug that shifts every chunk's read position by the same fixed offset would
affect both equally and slip through undetected by any spectrogram-based
check — magnitude spectra are shift-invariant, so that specific bug class is
instead caught by a plain unit test on the seek arithmetic
(`chunk_start_sample` in `tests/host/test_cli_util.cpp`).

    ./tests/test_chunking.sh      # needs acpp + GPU, ~1-2 min

`tests/test_radar_pulses.sh` is the equivalent end-to-end smoke test for
`radar_pulses`: it synthesizes a rectangular-envelope pulse train (known
PRI/width) as a complex-baseband IQ file, runs `radar_pulses` on it, checks
the reported width/PRI/PRF/duty-cycle against the known ground truth,
cross-checks against the same file run through the all-NumPy
`view_radar_pulses.py` path (two independent implementations should agree),
and confirms `view_radar_pulses.py` can plot the precomputed `.json` result
without error.

    ./tests/test_radar_pulses.sh  # needs acpp + GPU, numpy; ~10 s

## License

MIT (code). The Crab pulsar dataset is CC BY-SA 4.0 and is not redistributed here.
