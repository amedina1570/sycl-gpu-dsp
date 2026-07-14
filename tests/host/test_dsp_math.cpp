#include "doctest.h"
#include "dsp_math.hpp"

TEST_CASE("decode_i16 normalizes signed 16-bit samples to [-1, 1]") {
  CHECK(dsp::decode_i16(0) == doctest::Approx(0.0f));
  CHECK(dsp::decode_i16(-32768) == doctest::Approx(-1.0f));
  CHECK(dsp::decode_i16(32767) == doctest::Approx(1.0f).epsilon(0.001));
  CHECK(dsp::decode_i16(16384) == doctest::Approx(0.5f));
}

TEST_CASE("dm_shift_samples is zero at the reference frequency, for any DM") {
  const double fref = 410e6 + (8192.0/2)*20e6/8192.0;
  CHECK(dsp::dm_shift_samples(56.7, fref, fref, 2048.0/20e6) == 0);
  CHECK(dsp::dm_shift_samples(0.0, 410e6, fref, 2048.0/20e6) == 0);
}

TEST_CASE("dm_shift_samples is zero everywhere when DM is zero") {
  CHECK(dsp::dm_shift_samples(0.0, 350e6, 410e6, 2048.0/20e6) == 0);
  CHECK(dsp::dm_shift_samples(0.0, 470e6, 410e6, 2048.0/20e6) == 0);
}

TEST_CASE("compute_dm_shifts is all-zero for DM=0") {
  auto shift = dsp::compute_dm_shifts(0.0, 410e6, 20e6, 8192, 2048.0/20e6);
  REQUIRE(shift.size() == 8192);
  for (int s : shift) CHECK(s == 0);
}

TEST_CASE("compute_dm_shifts decreases monotonically with frequency (positive DM)") {
  // Crab-pulse parameters from dedisp.cpp/dmsearch.cpp. Lower-frequency
  // channels (small k) are further from the top-of-band reference than
  // higher-frequency channels (large k), so |shift| should shrink as k grows.
  constexpr int NFFT = 8192;
  auto shift = dsp::compute_dm_shifts(56.7, 410e6, 20e6, NFFT, 2048.0/20e6);
  REQUIRE(shift.size() == NFFT);
  for (int k = 1; k < NFFT; ++k)
    CHECK(shift[k] <= shift[k-1]);
  CHECK(shift[0] > shift[NFFT-1]);
}

TEST_CASE("compute_snr on a synthetic profile with one clear pulse") {
  std::vector<float> values = {1,1,1,1,1,1,1,1,1,100};
  dsp::SnrStats s = dsp::compute_snr(values);
  CHECK(s.median == doctest::Approx(1.0f));
  CHECK(s.mean == doctest::Approx(10.9f));
  CHECK(s.stddev == doctest::Approx(29.7f));
  CHECK(s.peak == doctest::Approx(100.0f));
  CHECK(s.snr == doctest::Approx(10.0f/3.0f));
}

TEST_CASE("compute_snr handles degenerate inputs without dividing by zero") {
  CHECK(dsp::compute_snr({}).snr == doctest::Approx(0.0f));

  dsp::SnrStats flat = dsp::compute_snr(std::vector<float>(5, 3.0f));
  CHECK(flat.stddev == doctest::Approx(0.0f));
  CHECK(flat.snr == doctest::Approx(0.0f));

  dsp::SnrStats single = dsp::compute_snr(std::vector<float>{7.0f});
  CHECK(single.median == doctest::Approx(7.0f));
  CHECK(single.peak == doctest::Approx(7.0f));
  CHECK(single.snr == doctest::Approx(0.0f));
}
