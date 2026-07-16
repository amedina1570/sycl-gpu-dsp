// Parameters of the bundled Crab-pulsar worked example, shared by the
// dedispersion (dedisp.cpp) and DM-search (dmsearch.cpp) programs so the two
// can't drift apart. NFFT/HOP must match the framing iq2spectrogram/stageC
// used to write the spectrogram .bin these programs read back, or the
// row-major (frames x NFFT) layout is misinterpreted.
//
// The recording is a 0.2 s, 20 Msps SigMF ci16_le capture of a Crab giant
// pulse from the Dwingeloo Radio Telescope, centered at 410 MHz.
#pragma once
#include "dsp_constants.hpp"

namespace crab {

constexpr int    NFFT = dsp::DEFAULT_NFFT;                        // 8192
constexpr int    HOP  = dsp::DEFAULT_NFFT / dsp::DEFAULT_HOP_DIVISOR; // 2048
constexpr double FS   = 20e6;    // sample rate, Hz
constexpr double FC   = 410e6;   // center frequency, Hz
constexpr double DM   = 56.7;    // literature Crab dispersion measure, pc/cm^3

// Blind DM-search sweep: trials from DM_MIN to DM_MAX in DM_STEP increments.
constexpr double DM_MIN  = 40.0;
constexpr double DM_MAX  = 75.0;
constexpr double DM_STEP = 0.25;

} // namespace crab
