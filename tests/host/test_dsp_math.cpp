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

TEST_CASE("percentile matches numpy's linear-interpolation method") {
  std::vector<float> v = {1, 2, 3, 4, 5};
  CHECK(dsp::percentile(v, 0.0) == doctest::Approx(1.0f));
  CHECK(dsp::percentile(v, 50.0) == doctest::Approx(3.0f));
  CHECK(dsp::percentile(v, 100.0) == doctest::Approx(5.0f));
  CHECK(dsp::percentile(std::vector<float>{1, 2}, 50.0) == doctest::Approx(1.5f));
  CHECK(dsp::percentile(std::vector<float>{7}, 50.0) == doctest::Approx(7.0f));
  CHECK(dsp::percentile({}, 50.0) == doctest::Approx(0.0f));
}

TEST_CASE("detect_pulses finds two clean rectangular pulses") {
  // 40 samples, baseline 0, two 5-sample-wide pulses of amplitude 1 at
  // [5,10) and [20,25) -- rising[i] is the pulse's first high sample,
  // falling[i] is one past its last (matches Python detect_pulses).
  std::vector<float> env(40, 0.0f);
  for (int i = 5; i < 10; ++i) env[i] = 1.0f;
  for (int i = 20; i < 25; ++i) env[i] = 1.0f;

  dsp::PulseDetection det = dsp::detect_pulses(env, 0.5f, 3);
  REQUIRE(det.rising.size() == 2);
  REQUIRE(det.falling.size() == 2);
  CHECK(det.rising[0] == 5);   CHECK(det.falling[0] == 10);
  CHECK(det.rising[1] == 20);  CHECK(det.falling[1] == 25);
  CHECK(det.threshold == doctest::Approx(0.5f));
}

TEST_CASE("detect_pulses drops pulses cut off at either boundary") {
  // Active from sample 0 (no rising edge captured) through sample 4, then a
  // clean pulse at [10,15), then active again from 30 through the end (no
  // falling edge captured). Only the clean middle pulse should survive.
  std::vector<float> env(35, 0.0f);
  for (int i = 0; i < 5; ++i) env[i] = 1.0f;
  for (int i = 10; i < 15; ++i) env[i] = 1.0f;
  for (int i = 30; i < 35; ++i) env[i] = 1.0f;

  dsp::PulseDetection det = dsp::detect_pulses(env, 0.5f, 3);
  REQUIRE(det.rising.size() == 1);
  CHECK(det.rising[0] == 10);
  CHECK(det.falling[0] == 15);
}

TEST_CASE("detect_pulses filters out pulses shorter than min_pulse_samples") {
  std::vector<float> env(20, 0.0f);
  env[5] = 1.0f;                      // 1-sample blip
  for (int i = 10; i < 15; ++i) env[i] = 1.0f;  // 5-sample real pulse

  dsp::PulseDetection det = dsp::detect_pulses(env, 0.5f, 3);
  REQUIRE(det.rising.size() == 1);
  CHECK(det.rising[0] == 10);
}

TEST_CASE("pulse_stats on a synthetic two-pulse train matches hand-computed PRI/PRF") {
  dsp::PulseDetection det;
  det.rising = {5, 20};
  det.falling = {10, 25};
  dsp::PulseStats s = dsp::pulse_stats(det, 1000.0);  // fs = 1000 Hz

  CHECK(s.n_pulses == 2);
  CHECK(s.pulse_width_mean_s == doctest::Approx(0.005f));
  CHECK(s.pulse_width_std_s == doctest::Approx(0.0f));
  REQUIRE(s.has_pri);
  CHECK(s.pri_mean_s == doctest::Approx(0.015f));
  CHECK(s.prf_hz == doctest::Approx(1.0f / 0.015f));
  CHECK(s.duty_cycle == doctest::Approx(0.005f / 0.015f));
}

TEST_CASE("pulse_stats handles 0 and 1 detected pulses without PRI") {
  dsp::PulseStats none = dsp::pulse_stats(dsp::PulseDetection{}, 1000.0);
  CHECK(none.n_pulses == 0);
  CHECK_FALSE(none.has_pri);

  dsp::PulseDetection one;
  one.rising = {5};
  one.falling = {10};
  dsp::PulseStats s = dsp::pulse_stats(one, 1000.0);
  CHECK(s.n_pulses == 1);
  CHECK(s.pulse_width_mean_s == doctest::Approx(0.005f));
  CHECK_FALSE(s.has_pri);
}
