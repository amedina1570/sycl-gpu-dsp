#include "doctest.h"
#include "csv_iq.hpp"
#include <cstring>

namespace {
// Convenience wrapper so tests can pass a C string and get back an
// offset instead of juggling raw pointers.
bool parse(const char* line, long long& i, long long& q, size_t* consumed = nullptr) {
  const char* begin = line;
  const char* end = line + std::strlen(line);
  const char* next = csv::parse_iq_line(begin, end, i, q);
  if (!next) return false;
  if (consumed) *consumed = (size_t)(next - begin);
  return true;
}
}  // namespace

TEST_CASE("parse_iq_line parses a normal comma-separated pair with no trailing newline") {
  long long i = 0, q = 0;
  REQUIRE(parse("26,-11", i, q));
  CHECK(i == 26);
  CHECK(q == -11);
}

TEST_CASE("parse_iq_line parses a newline-terminated pair and stops after the newline") {
  long long i = 0, q = 0;
  size_t consumed = 0;
  REQUIRE(parse("16,-48\n", i, q, &consumed));
  CHECK(i == 16);
  CHECK(q == -48);
  CHECK(consumed == 7);  // stops right after '\n', ready for the next line
}

TEST_CASE("parse_iq_line tolerates CRLF line endings") {
  long long i = 0, q = 0;
  REQUIRE(parse("-30,1\r\n", i, q));
  CHECK(i == -30);
  CHECK(q == 1);
}

TEST_CASE("parse_iq_line tolerates surrounding whitespace around each number") {
  long long i = 0, q = 0;
  REQUIRE(parse(" 26 , -11 \n", i, q));
  CHECK(i == 26);
  CHECK(q == -11);
}

TEST_CASE("parse_iq_line rejects a missing comma") {
  long long i = 0, q = 0;
  CHECK_FALSE(parse("26 -11\n", i, q));
}

TEST_CASE("parse_iq_line rejects a non-numeric line (e.g. the CSV header row)") {
  long long i = 0, q = 0;
  CHECK_FALSE(parse("I Data,Q Data\n", i, q));
}

TEST_CASE("parse_iq_line rejects an empty line") {
  long long i = 0, q = 0;
  CHECK_FALSE(parse("", i, q));
}

TEST_CASE("parse_iq_line rejects trailing garbage before the line ends") {
  long long i = 0, q = 0;
  CHECK_FALSE(parse("26,-11x\n", i, q));
}

TEST_CASE("parse_iq_line scans consecutive lines from one buffer") {
  const char text[] = "1,2\n3,-4\n5,6";
  const char* p = text;
  const char* end = text + sizeof(text) - 1;  // exclude the implicit '\0'
  long long i, q;

  p = csv::parse_iq_line(p, end, i, q);
  REQUIRE(p != nullptr);
  CHECK(i == 1); CHECK(q == 2);

  p = csv::parse_iq_line(p, end, i, q);
  REQUIRE(p != nullptr);
  CHECK(i == 3); CHECK(q == -4);

  p = csv::parse_iq_line(p, end, i, q);
  REQUIRE(p != nullptr);
  CHECK(i == 5); CHECK(q == 6);
  CHECK(p == end);  // last line had no trailing newline
}
