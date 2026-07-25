# Learning sycl-gpu-dsp

This document explains *why* the code is built the way it is — the SYCL/GPU
programming concepts, the DSP theory behind each algorithm, and the
conventions a contributor should follow when adding to the project. The
[README](../README.md) is the quick-reference (how to build, how to run
each tool); this is the deeper walkthrough.

It's organized in the same order the repository is: SYCL fundamentals
first, then the general pipeline built from them, then the three
domain-specific tools built on top of *that*, then the conventions tying it
all together. If you're new to SYCL, read Part 1 in order. If you already
know SYCL and want the DSP theory, Parts 1–3 still have it woven in
section by section. If you're about to add a new tool, skip to Part 4.

## Prerequisites

You don't need to know SYCL going in — Part 1 introduces each concept the
first time the code needs it. You do need enough C++ to read templates and
lambdas comfortably, and it helps (but isn't required) to have seen a
Fourier transform before. Complex numbers show up throughout: an I/Q
sample *is* a complex number, `I + jQ`, and most of the DSP theory below is
just "what happens when you do arithmetic on those."

---

## Part 1 — SYCL fundamentals

Four small, self-contained programs, each introducing one new SYCL concept
on top of the last. None of them are the production code path — they're
what the production code (Part 2) is built *from*. Build and run each one
as you read it:

```
make build/01_usm && ./build/01_usm
```

### 1.1 USM and queues — `src/01_usm.cpp`

A GPU has its own memory, separate from the host's RAM. Before you can run
anything on it, you need a way to (a) allocate memory there and (b) move
data across the PCIe bus between host and device. SYCL calls this **USM**
(Unified Shared Memory), and there are three flavors:

- **`sycl::malloc_device`** — memory that lives only on the device. Fastest
  to access from a kernel, but the host can't read or write it directly;
  you move data in and out explicitly with `queue::memcpy`. This is what
  every program in this repo uses (see `sycl_util.hpp`'s
  `malloc_device_checked` — a thin wrapper that also checks the allocation
  didn't silently fail, which the raw SYCL call doesn't do for you).
- **`sycl::malloc_host`** — the reverse: host-resident but device-visible.
  Rare in this codebase; mostly useful for tiny amounts of data touched by
  both sides.
- **`sycl::malloc_shared`** — a single allocation that migrates
  automatically between host and device as needed. Simpler to write, but
  the automatic migration costs performance you don't control. Not used
  here — this project favors explicit `malloc_device` + `memcpy` for
  predictable performance (see AdaptiveCpp's own
  [performance guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md),
  which this project follows: device USM for control, never buffers).

A **queue** is how you submit work to a device — kernels, memory copies,
everything. Every queue in this repo is constructed
`sycl::property::queue::in_order{}`: operations submitted to it run in the
order you submitted them, so `memcpy` → `kernel` → `memcpy` back is
automatically correctly sequenced without you having to manually declare
dependencies between them. The tradeoff is you give up out-of-order
scheduling within that queue — a non-issue for the mostly-linear
pipelines here.

One more thing every queue in this repo does: it's constructed with an
**async exception handler** —

```cpp
sycl::queue q{
    [](sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try { std::rethrow_exception(e); }
        catch (const sycl::exception& ex) {
          std::cerr << "asynchronous SYCL exception: " << ex.what() << "\n";
        }
      }
    },
    sycl::property::queue::in_order{}};
```

This matters because a failure inside a kernel or a copy doesn't throw
where you'd expect — `q.wait()` alone won't surface it. Without this
handler, a failing kernel just... does nothing, silently. `01_usm.cpp`
does the simplest possible thing with all this: allocate three device
buffers, copy two arrays in, add them element-wise in a `parallel_for`
kernel, copy the result back, verify it.

### 1.2 Kernels and device-side math — `src/02_window.cpp`

A kernel is the function that runs once per work-item (roughly: once per
output element) on the device. `parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {...})`
launches `N` of them, each knowing its own index. `02_window.cpp` uses
this to compute a **Hann window**:

```
w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
```

**Why window at all?** Any real-world signal you capture is finite —
you're always looking through a rectangular "aperture" of `N` samples,
whether you multiply by an explicit window or not (an implicit rectangular
window is still a window). Truncating a signal abruptly is equivalent, in
the frequency domain, to convolving its true spectrum with a `sinc`
function — which has significant sidelobes. The practical symptom is
**spectral leakage**: energy at one frequency smears into neighboring
bins, and a strong signal's sidelobes can bury a weaker signal nearby.
Multiplying by a window that tapers to zero at both ends (Hann does; a
plain rectangular window doesn't) trades some main-lobe width (frequency
resolution) for much lower sidelobes — less leakage.

*Which* window to use is a real tradeoff, not a fixed answer — see §3.3
below for why this project uses two different windows in two different
places on purpose.

### 1.3 The DFT — `src/03_dft.cpp`

The Discrete Fourier Transform turns `N` time-domain samples into `N`
frequency-domain bins:

```
X[k] = sum_{n=0}^{N-1} x[n] * exp(-i*2*pi*k*n/N)
```

Bin `k` measures how much of the signal's energy is at frequency
`k * fs/N`. Computed directly, this is `O(N²)` — a nested loop, `N`
frequencies times `N` samples each. `dft_lib.hpp`'s `naive_dft_mag` does
exactly that, one GPU thread per output bin, each thread doing the full
`N`-term inner sum itself. It's clearly labeled a **correctness
baseline** — its only job is to be obviously, unambiguously right, so it
can be the ground truth the FFT (next) is checked against
(`tests/gpu/test_fft_vs_dft.cpp`).

### 1.4 The FFT — `src/04_fft.cpp`

`O(N²)` is fine for `N=1024` and painful for `N=1,000,000`. The **Fast
Fourier Transform** computes the identical result in `O(N log N)` by
exploiting a symmetry: the DFT of a sequence can be built from the DFTs of
its even- and odd-indexed halves. Apply that recursively (this project
implements the classic **radix-2 Cooley-Tukey** decomposition) and you get
`log₂N` stages, each doing `N/2` **butterfly** operations — a butterfly
combines one even-indexed and one odd-indexed partial result using one
complex multiply (by a "twiddle factor") and two adds:

```
a' = a + w*b
b' = a - w*b
```

Two SYCL concepts show up here for the first time:

- **`sycl::local_accessor`** — fast, on-chip memory shared by every
  work-item in a **work-group**, instead of the slower device-global
  memory every other kernel in this repo reads/writes through. The whole
  transform for one FFT happens in local memory.
- **`nd_range` + barriers** — because each butterfly stage depends on the
  *previous* stage's output, every work-item in the group has to finish
  stage `s` before any of them start stage `s+1`. `it.barrier(...)`
  enforces that. This is why the FFT needs `nd_range`/work-groups where
  `01_usm.cpp`/`02_window.cpp` didn't: those kernels had no
  inter-work-item dependencies at all, one thread's output never depended
  on another's.

Before the butterfly stages run, the input is reordered by **bit-reversal
permutation** (index `n`'s bits reversed gives its position in the
reordered array) — a standard part of the in-place radix-2 algorithm, so
the butterflies can operate in-place without needing extra buffers.

**Important scope note:** this FFT only works for one work-group's worth
of data — `N/2` must not exceed the device's max work-group size (often
1024, so `N` up to ~2048). It's also *not* what the production pipeline
actually uses for its FFTs — that's cuFFT, next in Part 2. `04_fft.cpp` is
explicitly a teaching artifact (see the README's own "SYCL fundamentals"
categorization): it exists to show what an FFT *is*, built entirely from
the SYCL primitives introduced in 1.1–1.3, not to be the fastest possible
FFT. Higher-radix (radix-4/8) or split-radix variants would do fewer
arithmetic operations per point, and a production-grade implementation
would also handle non-power-of-two sizes and transforms spanning many
work-groups — that's a genuinely hard problem, which is exactly why real
FFT libraries (cuFFT) exist instead of everyone hand-rolling one.

---

## Part 2 — The production pipeline

Part 1 built understanding from first principles. The actual pipeline
that processes real files uses a different, much faster FFT — NVIDIA's
**cuFFT** library — accessed from SYCL through interop. This part walks
through how that's wired together and how it scales to files far bigger
than GPU memory.

### 2.1 Loading SigMF — `src/stageA_load.cpp`

Every tool here reads **SigMF**, a simple convention for IQ recordings:
a raw binary `.sigmf-data` file (interleaved `I,Q,I,Q,...` samples, either
16-bit signed integers — `ci16_le` — or 32-bit floats — `cf32_le`) plus an
optional `.sigmf-meta` JSON sidecar describing sample rate, center
frequency, and datatype. `stageA_load.cpp` is the minimal version: open,
read, decode `ci16_le` samples to `[-1, 1]` floats
(`dsp::decode_i16`, dividing by 32768), report basic stats. It reads the
*whole* file into a `std::vector` up front — fine for a small file, a
correctness/design stepping stone toward the streaming version in §2.4.

### 2.2 cuFFT via SYCL interop — `src/stageB_cufft.cpp`, `src/cufft_interop.hpp`

cuFFT is a plain CUDA library, not a SYCL one. AdaptiveCpp lets you call
into it anyway via **interop**: `cgh.AdaptiveCpp_enqueue_custom_operation`
hands you a `sycl::interop_handle`, from which
`ih.get_native_queue<sycl::backend::cuda>()` gets you the actual native
CUDA stream the SYCL queue is running on — at which point you can call
`cufftSetStream` + `cufftExecC2C` directly, and because it's on the *same*
stream, it's correctly sequenced with everything else the SYCL queue is
doing, no extra synchronization needed. `cufft_interop.hpp` wraps this
pattern once (`enqueue_exec_c2c_forward`) since `stageB_cufft.cpp`,
`stageC_spectrogram.cpp`, and `iq2spectrogram.cpp` all need it.

Why cuFFT instead of `04_fft.cpp`'s hand-rolled radix-2 kernel? Vendor FFT
libraries implement mixed-radix decompositions (combining radix-2/3/5/7
stages to handle arbitrary sizes efficiently, not just powers of two),
Stockham-style autosort (avoiding the explicit bit-reversal pass), and
hardware-tuned kernels per GPU architecture. There's no realistic
hand-written kernel that beats that on NVIDIA hardware — so the pipeline
uses it, and the hand-rolled FFT stays a teaching example.

### 2.3 Assembling a spectrogram — `src/stageC_spectrogram.cpp`

A spectrogram answers "what frequencies are present, *and when*" — a 2D
image of frequency (one axis) vs. time (the other), color-coded by power.
It's built by repeating the same recipe on many overlapping, fixed-length
**frames** of the signal:

1. Slice out `NFFT` samples starting `HOP` samples after the previous
   frame (`HOP < NFFT` means frames overlap — this project defaults to
   75% overlap, `HOP = NFFT/4`, giving smoother time resolution than
   non-overlapping frames would).
2. Multiply by the window (§1.2 — Hann here).
3. FFT it (§2.2 — cuFFT, batched across all frames in one call).
4. Convert complex bins to power in dB: `10*log10(|X[k]|² + floor)` (the
   small `floor` in `dsp::db_from_power` keeps a zero-power bin from
   producing `-inf`).
5. **fftshift**: an FFT's raw bin order is `[0, +1, +2, ..., -2, -1]` in
   frequency (DC first, then increasing positive frequency, wrapping to
   the most-negative frequency at the far end) — fftshift reorders bins to
   `[-N/2, ..., -1, 0, +1, ..., N/2-1]` so a plotted spectrogram reads
   left-to-right as low-to-high frequency, with DC in the middle, matching
   how you'd naturally expect to read it.

`stageC_spectrogram.cpp` does all of this as one fixed-size demo (whole
file loaded, one giant batched cuFFT call) — a stepping stone toward the
one below, which is the version to actually use.

### 2.4 Streaming files bigger than memory — `src/iq2spectrogram.cpp`

`stageC_spectrogram.cpp`'s "load the whole file, one batched FFT call"
approach breaks down fast: a multi-GB recording needs a multi-GB batch
buffer *and* a multi-GB spectrogram output buffer, on top of the input —
easily exceeding both host RAM and GPU VRAM. `iq2spectrogram.cpp` is the
same recipe (§2.3), but run in **bounded chunks**: `cli::resolve_chunk_frames`
(`cli_util.hpp`) works out how many frames fit in a `--chunk-mb` budget of
GPU memory, and the main loop seeks to each chunk's byte offset, reads
just those samples, runs the same framing→window→FFT→dB kernels on
**fixed, reused** device buffers, and appends the result straight to the
output `.bin`. Memory usage stays flat regardless of file size — a 10 MB
file and a 100 GB file are processed the same way, just for longer.
(Verified, not just argued: running the same file once with default
chunking and once forced into many more, smaller chunks produces
byte-identical output — chunking is purely a memory-bounding strategy, not
a behavior change. See `tests/test_chunking.sh`.)

This file also introduces two conventions used throughout the rest of the
project (see Part 4 for the general pattern): reading `--fs`/`--fc`/
`--datatype` from a `.sigmf-meta` sidecar when present, and writing a
small JSON sidecar (`<prefix>_spectrogram.json`) alongside its binary
output describing exactly what it wrote — so the next tool in the chain
(the Python viewer, or another program) doesn't need those parameters
repeated on its own command line.

### 2.5 Getting data in: `src/csv2sigmf.cpp`

Not every I/Q recording arrives as SigMF — some vendor test equipment
exports plain CSV instead (a header row, then one `<I>,<Q>` integer pair
per line). `csv2sigmf` converts one into a real `.sigmf-data`/
`.sigmf-meta` pair, so every tool above reads it unchanged. The interesting
part is *how*: at the scale this format shows up in practice (hundreds of
millions of lines — the NIST TN 2159 dataset this was built against and
validated on is a 307-million-line, 2.4 GB file), `std::getline` +
`std::stringstream` per line would take tens of minutes. Instead,
`csv_iq.hpp`'s `parse_iq_line` is a hand-rolled pointer-arithmetic parser,
and the main loop reads the file in large (64 MB) blocks, only ever
carrying a genuinely-partial trailing line over to the next block. Same
conversion on the real 2.4 GB file: about 13 seconds.

---

## Part 3 — Three domains built on the pipeline

Everything above is general-purpose — it works on any SigMF capture. This
project's original scope was inspecting I/Q signals across three specific
domains, each with its own tool(s) built on top of the general pipeline.

### 3.1 Radioastronomy — `src/dedisp.cpp`, `src/dmsearch.cpp`

Pulses from a real pulsar don't arrive at all frequencies simultaneously —
interstellar plasma delays lower frequencies more than higher ones, so a
giant pulse's arrival sweeps diagonally across a spectrogram (a "dispersion
sweep") instead of appearing as a single vertical line. The delay follows
the standard cold-plasma law:

```
delay(s) = K * DM * (f_GHz)^-2
```

where `DM` (dispersion measure, pc/cm³) characterizes how much plasma the
signal passed through, and `K` is a physical constant
(`dsp::DM_DELAY_CONST_S`). **Incoherent dedispersion**
(`dedisp.cpp`) undoes this: for each frequency channel, shift its samples
backward in time by that channel's predicted delay (relative to the top of
the band), so the pulse's energy lines up in a single frame again instead
of being smeared diagonally across many.

The catch: dedispersion only *un-smears* the signal if you already know
the right DM. `dmsearch.cpp` does a **blind DM search** — try many trial
DM values, dedisperse with each, and score how sharply each one collapses
the pulse using `dsp::compute_snr` (`(peak - median) / stddev` of the
frequency-summed profile — a real pulse gives a high score at the *correct*
DM and a low, flat score everywhere else). The DM with the best score wins.
On the bundled Crab giant-pulse recording, the blind search recovers
DM ≈ 56.75 pc/cm³, matching the literature value (56.7) independently.

### 3.2 RADAR — `src/radar_pulses.cpp`

A pulsed RADAR-like signal's defining feature isn't its spectrum, it's its
*timing*: pulse width, how often pulses repeat (PRI — pulse repetition
interval, and its reciprocal PRF — pulse repetition frequency), and duty
cycle (fraction of time actually transmitting). `radar_pulses.cpp`
computes the signal's **envelope** (`|I + jQ|`, the instantaneous
magnitude — note this needs *both* I and Q; a real-valued signal's naive
magnitude oscillates with the carrier and isn't a stable envelope, a
mistake actually made and caught while building this tool's test data,
before it was ever committed — the fix was to synthesize a proper
complex-baseband tone instead), lightly smooths it (boxcar/moving-average — matching NumPy's
`convolve(..., mode='same')` exactly, edge-tapering included, so the SYCL
and NumPy implementations agree bit-for-bit), thresholds it (`floor +
threshold_frac * (peak - floor)`, using the 10th/99.9th percentiles as
robust floor/peak estimates), and pairs up rising/falling edges into
complete pulses — dropping any pulse cut off at either end of the analyzed
segment, since its true width/timing can't be measured. From the surviving
pulses: width = mean `(falling - rising)/fs`, PRI = mean spacing between
consecutive rising edges, PRF = `1/PRI`, duty cycle = `width/PRI`.

### 3.3 RF — `view/view_welch.py`

Not every signal is pulsed — a lot of real RF traffic (cellular uplink,
Wi-Fi, anything continuous or quasi-continuous) is better characterized by
its steady-state spectral content: how wide is the occupied channel, and
how far above the noise floor does it sit? A single windowed FFT snapshot
(like `view_iq_snapshot.py` uses for a quick look) is a noisy estimate —
its variance doesn't improve no matter how much data you feed it, because
you're only ever looking at one realization of the spectrum. **Welch's
method** fixes this: split the segment into many overlapping sub-segments,
FFT each one separately, and average the resulting power spectra together.
More segments averaged = lower variance, at the cost of frequency
resolution (`fs/nperseg` — a shorter sub-segment means more of them to
average, but coarser bins).

This is also where the **window choice actually matters as a tradeoff**,
concretely, not just in the abstract (§1.2): `view_welch.py` defaults to
**Blackman-Harris**, not the Hann window the rest of the pipeline uses, and
that's deliberate. Hann has lower sidelobes than a rectangular window but
still leaks meaningfully (~-31 dB); Blackman-Harris trades main-lobe width
for far deeper suppression (~-92 dB). For a spectrogram, Hann's tighter
main lobe (better time/frequency resolution) is worth it. For measuring
*occupied bandwidth* — where a strong in-band signal's own sidelobes can
otherwise get mistaken for real spectral content further out — the deeper
suppression matters more than the resolution. Same underlying math
(window before FFT), different point on the resolution/leakage tradeoff,
chosen for what each tool is actually measuring.

**A real worked example.** This tool was validated against a real capture:
a 5-second, 307.2M-sample, 61.44 Msps LTE uplink recording (NIST TN 2159).
Run with default settings, its "occupied bandwidth" measurement locked
onto a narrow, isolated −21 MHz spike — technically the single loudest
point in the spectrum, but *not* the actual signal of interest (a much
wider, lower-amplitude ~10-18 MHz plateau around 0 Hz — the real PUSCH
channel, matching the expected ~18 MHz for a 20 MHz LTE channel at 100
resource blocks). That's a real, physically-meaningful spur (most likely
LO leakage), and the tool's peak-finding needed a `--obw-smooth-khz` option
— smoothing the search *before* locating the peak, so a single narrow spur
can't hijack the measurement away from a wider channel elsewhere in the
band. But that same smoothing, applied by default, broke the simple case
where the signal of interest genuinely *is* narrowband (a synthetic test
tone) — diluting its own peak into the noise floor. The fix: smoothing
defaults to *off*, and is an opt-in flag for the "wideband channel with an
unrelated spur nearby" case specifically. Real data breaking a
reasonable-looking default, twice in a row (this and a related edge-padding
bug — see the file's own comments), is a good reminder that DSP heuristics
that look right on paper are worth checking against messy real data, not
just clean synthetic test signals.

---

## Part 4 — Conventions for contributors

Adding a new tool? These are the patterns already established across the
codebase — following them means your tool composes with everything else
here for free.

**CLI shape.** Every CLI program follows the same skeleton: an `Args`
struct, a `usage()` function, `parse_args()` (fills `Args` from `argv`,
`--help`/`-h` prints usage and exits 0), and `validate_args()` (checks the
parsed values are sane, returns `false` with a clear stderr message rather
than proceeding on bad input). See `iq2spectrogram.cpp` or
`radar_pulses.cpp` for the fullest examples.

**`.sigmf-meta` auto-detection.** Any tool that reads a SigMF file should
try `sigmf::read_file`/`json_number`/`json_string` (`sigmf_meta.hpp`) for
`core:sample_rate`/`core:frequency`/`core:datatype` before falling back to
CLI flags or hard defaults — so a user who already has a sidecar doesn't
have to repeat those values on every command.

**JSON sidecars for cross-tool handoff.** A tool that produces output
another tool (often a Python viewer) will consume should write a small
JSON sidecar describing exactly what it computed — parameters, results,
paths to any binary output — rather than requiring the consumer to
recompute or guess. `view_spec.py` and `view_radar_pulses.py` both support
this "precomputed" mode as an alternative to their standalone (recompute
everything) path.

**Checked device allocations.** Always use
`sycl_util::malloc_device_checked<T>(count, q, "name")` instead of calling
`sycl::malloc_device` directly — the raw call returns `nullptr` (not an
exception) on failure, which otherwise turns into a null-pointer kernel
crash with no useful message.

**Fail loud, not quiet.** Validate inputs at the boundary (CLI args, file
reads) and bail with a clear stderr message and non-zero exit rather than
proceeding on bad data. Device-memory guards (checking a requested
allocation against `sycl::info::device::global_mem_size` before making it)
follow the same philosophy — see `dedisp.cpp`/`dmsearch.cpp`/
`radar_pulses.cpp`.

**Build target.** Every program compiles through AdaptiveCpp's `generic`
(JIT) target — see the Makefile's own comment and
[AdaptiveCpp's performance guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md)
for why. Programs that call cuFFT directly additionally link
`cufft`/`cudart`; that's independent of the SYCL compilation target.

**Testing philosophy — three layers, not one:**

1. **Unit tests** (`tests/host/*.cpp`, `tests/gpu/*.cpp`, doctest) for pure
   logic pulled out of `main()` into headers — CLI validation
   (`cli_util.hpp`), math (`dsp_math.hpp`), the actual kernel bodies
   (`dft_lib.hpp`/`fft_lib.hpp`/`radar_lib.hpp`). If it's testable without
   `argv` or file I/O, it belongs in a header, not inline in `main()`.
2. **Integration smoke tests** (`tests/test_*.sh`) that build and run the
   real binary end-to-end against a synthetic input with a *known* answer
   (a tone at a known bin, a pulse train with known PRI/width), and often
   cross-check two independent implementations of the same thing against
   each other (e.g. `radar_pulses.cpp` vs. `view_radar_pulses.py`'s
   NumPy path) — agreement between two different implementations is much
   stronger evidence than either one alone.
3. **Real-data validation**, where possible — the strongest check of all,
   and the one synthetic data can't provide. Every domain tool in this
   project was checked against a real, independently-produced ground truth
   at least once: `dmsearch.cpp` against the literature DM for the Crab
   pulsar, `radar_pulses.cpp`/`view_welch.py` against the NIST LTE capture
   (cross-checked against that recording's own hardware-logged burst
   timestamps — 433/433 matched within ±2 ms). Synthetic tests catch
   implementation bugs; real data catches *wrong assumptions* that a clean
   synthetic signal would never expose (real-valued vs. complex envelopes,
   a spur vs. the actual channel, and so on — see §3.2 and §3.3).

---

## Glossary

- **USM** — Unified Shared Memory, SYCL's memory-allocation model (device/host/shared).
- **In-order queue** — a SYCL queue where submitted operations execute in submission order.
- **Kernel** — the function executed once per work-item on the device.
- **Work-group / work-item** — a work-item is one kernel invocation; a work-group is a set of them that can cooperate via local memory and barriers.
- **DFT / FFT** — Discrete/Fast Fourier Transform; the FFT computes the same result as the DFT in `O(N log N)` instead of `O(N²)`.
- **Butterfly** — the basic combine step of a Cooley-Tukey FFT: `a±w*b`.
- **Window function** — a taper applied to a finite signal segment before an FFT, trading frequency resolution for reduced spectral leakage.
- **Spectral leakage** — energy from one frequency smearing into neighboring bins due to finite-length analysis.
- **Spectrogram** — a time-vs-frequency-vs-power image built from many overlapping windowed FFTs.
- **fftshift** — reordering FFT bins from `[0..N/2-1, -N/2..-1]` to `[-N/2..N/2-1]` for natural low-to-high plotting.
- **SigMF** — a simple convention for IQ file storage: raw binary samples (`ci16_le`/`cf32_le`) plus a JSON metadata sidecar.
- **Dispersion measure (DM)** — a measure of how much interstellar plasma a radio signal passed through, in pc/cm³; determines frequency-dependent arrival delay.
- **Incoherent dedispersion** — undoing dispersion delay by shifting each frequency channel's samples in time.
- **PRI / PRF** — pulse repetition interval / frequency (period and rate of a pulsed signal); duty cycle = pulse width / PRI.
- **Welch's method** — PSD estimation by averaging the FFTs of many overlapping sub-segments, trading frequency resolution for lower variance.
- **Occupied bandwidth (OBW)** — the bandwidth around a spectrum's peak that stays within some threshold (e.g. 10 dB) of it.
- **PSD** — power spectral density, power per unit frequency (dB/Hz here).
