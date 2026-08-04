#include "doctest.h"
#include "cli_util.hpp"

TEST_CASE("is_power_of_two accepts powers of two") {
  CHECK(cli::is_power_of_two(1));
  CHECK(cli::is_power_of_two(2));
  CHECK(cli::is_power_of_two(4096));
  CHECK(cli::is_power_of_two(8192));
  CHECK(cli::is_power_of_two(1 << 20));
}

TEST_CASE("is_power_of_two rejects non-powers-of-two") {
  CHECK_FALSE(cli::is_power_of_two(0));
  CHECK_FALSE(cli::is_power_of_two(-8));
  CHECK_FALSE(cli::is_power_of_two(3));
  CHECK_FALSE(cli::is_power_of_two(4095));
  CHECK_FALSE(cli::is_power_of_two(6));
}

TEST_CASE("resolve_hop defaults to 75% overlap (nfft/4) when unset") {
  CHECK(cli::resolve_hop(8192, 0) == 2048);
  CHECK(cli::resolve_hop(1024, 0) == 256);
}

TEST_CASE("resolve_hop passes through an explicit hop") {
  CHECK(cli::resolve_hop(8192, 512) == 512);
  CHECK(cli::resolve_hop(8192, 1) == 1);
}

TEST_CASE("is_supported_datatype accepts ci16_le and cf32_le") {
  CHECK(cli::is_supported_datatype("ci16_le"));
  CHECK(cli::is_supported_datatype("cf32_le"));
}

TEST_CASE("is_supported_datatype rejects anything else") {
  CHECK_FALSE(cli::is_supported_datatype(""));
  CHECK_FALSE(cli::is_supported_datatype("cu8"));
  CHECK_FALSE(cli::is_supported_datatype("CI16_LE"));
}

TEST_CASE("parse_int accepts only complete in-range integers") {
  auto v = cli::parse_int("8192");
  REQUIRE(v.has_value());
  CHECK(*v == 8192);
  CHECK_FALSE(cli::parse_int("8192x").has_value());
  CHECK_FALSE(cli::parse_int("999999999999999999999").has_value());
  CHECK_FALSE(cli::parse_int("3.14").has_value());
}

TEST_CASE("parse_long accepts signed complete integers") {
  auto v = cli::parse_long("-42");
  REQUIRE(v.has_value());
  CHECK(*v == -42);
  CHECK_FALSE(cli::parse_long("").has_value());
  CHECK_FALSE(cli::parse_long("12 samples").has_value());
}

TEST_CASE("parse_double rejects malformed and non-finite values") {
  auto v = cli::parse_double("20e6");
  REQUIRE(v.has_value());
  CHECK(*v == doctest::Approx(20e6));
  CHECK_FALSE(cli::parse_double("20e6Hz").has_value());
  CHECK_FALSE(cli::parse_double("nan").has_value());
  CHECK_FALSE(cli::parse_double("inf").has_value());
}

TEST_CASE("parse_float rejects out-of-range values") {
  auto v = cli::parse_float("0.5");
  REQUIRE(v.has_value());
  CHECK(*v == doctest::Approx(0.5f));
  CHECK_FALSE(cli::parse_float("1e100").has_value());
}

TEST_CASE("resolve_chunk_frames fits the requested memory budget") {
  // bytes/frame = 8192 * 12 = 98304; 256 MB budget -> floor(268435456/98304)
  CHECK(cli::resolve_chunk_frames(256, 8192, 10'000'000) == 2730);
}

TEST_CASE("resolve_chunk_frames clamps to total_frames when the budget is generous") {
  CHECK(cli::resolve_chunk_frames(4096, 8192, 100) == 100);
}

TEST_CASE("resolve_chunk_frames never returns fewer than one frame") {
  CHECK(cli::resolve_chunk_frames(0, 8192, 1000) == 1);
  CHECK(cli::resolve_chunk_frames(1, 1 << 20, 1000) == 1);
}

TEST_CASE("chunk_sample_count covers exactly the hop-spaced frame range") {
  CHECK(cli::chunk_sample_count(1, 2048, 8192) == 8192);
  CHECK(cli::chunk_sample_count(5, 2048, 8192) == 16384);
  CHECK(cli::chunk_sample_count(2730, 2048, 8192) == 5597184);
}

TEST_CASE("chunk_start_sample is exactly f0*hop, not off by any constant") {
  CHECK(cli::chunk_start_sample(0, 2048) == 0);
  CHECK(cli::chunk_start_sample(1, 2048) == 2048);
  CHECK(cli::chunk_start_sample(2730, 2048) == 5591040);
  CHECK(cli::chunk_start_sample(989, 42) == 41538);
}
