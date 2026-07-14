#include "doctest.h"
#include "sigmf_meta.hpp"

TEST_CASE("json_number finds a top-level numeric field") {
  std::string text = R"({"core:sample_rate": 20000000.0, "core:frequency": 410000000.0})";
  auto v = sigmf::json_number(text, "core:sample_rate");
  REQUIRE(v.has_value());
  CHECK(*v == doctest::Approx(20000000.0));
}

TEST_CASE("json_number handles keys whose name contains a colon") {
  // The "core:" prefix means a naive find(':', pos) from the key's start
  // would match the colon inside the key itself, not the separator.
  std::string text = R"({"core:frequency": 410e6})";
  auto v = sigmf::json_number(text, "core:frequency");
  REQUIRE(v.has_value());
  CHECK(*v == doctest::Approx(410e6));
}

TEST_CASE("json_number returns nullopt for a missing key") {
  std::string text = R"({"core:sample_rate": 20e6})";
  CHECK_FALSE(sigmf::json_number(text, "core:frequency").has_value());
}

TEST_CASE("json_string extracts a quoted value") {
  std::string text = R"({"core:datatype": "ci16_le", "core:sample_rate": 20e6})";
  auto v = sigmf::json_string(text, "core:datatype");
  REQUIRE(v.has_value());
  CHECK(*v == "ci16_le");
}

TEST_CASE("json_string returns nullopt for a missing key") {
  std::string text = R"({"core:sample_rate": 20e6})";
  CHECK_FALSE(sigmf::json_string(text, "core:datatype").has_value());
}

TEST_CASE("json_number and json_string coexist in a realistic sidecar") {
  std::string text = R"({
    "global": {
      "core:datatype": "ci16_le",
      "core:sample_rate": 20000000,
      "core:version": "1.0.0"
    },
    "captures": [{"core:frequency": 410000000, "core:sample_start": 0}]
  })";
  CHECK(*sigmf::json_string(text, "core:datatype") == "ci16_le");
  CHECK(*sigmf::json_number(text, "core:sample_rate") == doctest::Approx(20000000));
  CHECK(*sigmf::json_number(text, "core:frequency") == doctest::Approx(410000000));
}

TEST_CASE("read_file returns nullopt for a nonexistent path") {
  CHECK_FALSE(sigmf::read_file("/nonexistent/path/does-not-exist.sigmf-meta").has_value());
}
