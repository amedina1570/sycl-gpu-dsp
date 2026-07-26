/**
 * @file csv_iq.hpp
 * @brief Fast, dependency-free CSV line parser for `"<int>,<int>"` I/Q pairs.
 *
 * Used by csv2sigmf.cpp to convert vendor CSV IQ dumps (one signed-integer
 * sample pair per line, no quoting/escaping) to raw SigMF at the speed a
 * multi-hundred-million-line file demands -- `std::getline` +
 * `std::stringstream` per line would take far too long at that scale.
 */
#pragma once

/// @namespace csv
/// @brief Minimal, high-throughput CSV I/Q parsing support for csv2sigmf.cpp.
namespace csv {

/// Parse one line `"<int>,<int>"` starting at `begin`. Whitespace around
/// either number is skipped. The line may end in `"\n"`, `"\r\n"`, or
/// simply at `end` (the true end of the file often has no trailing
/// newline).
///
/// Calling convention for streaming a large buffer: only pass `end` as
/// either (a) the position immediately after a `'\n'` already found in the
/// buffer (so intermediate lines always terminate within `[begin, end)`
/// and never spuriously hit the end-of-buffer case), or (b) the true end
/// of the whole input, for the final -- possibly newline-less -- line.
///
/// @param begin Start of the line to parse.
/// @param end End of the available buffer (see calling convention above).
/// @param[out] i Parsed first (I) value, valid only if this returns non-null.
/// @param[out] q Parsed second (Q) value, valid only if this returns non-null.
/// @return A pointer just past the parsed line (past the newline, if any),
/// or `nullptr` if `[begin, end)` does not start with a complete, valid line.
inline const char* parse_iq_line(const char* begin, const char* end,
                                  long long& i, long long& q) {
  const char* p = begin;

  auto skip_ws = [&]() {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
  };
  auto parse_int = [&](long long& out) -> bool {
    skip_ws();
    const char* start = p;
    bool neg = false;
    if (p < end && *p == '-') { neg = true; ++p; }
    long long v = 0;
    bool any_digit = false;
    while (p < end && *p >= '0' && *p <= '9') {
      v = v * 10 + (*p - '0');
      ++p;
      any_digit = true;
    }
    if (!any_digit) { p = start; return false; }
    out = neg ? -v : v;
    return true;
  };

  if (!parse_int(i)) return nullptr;
  skip_ws();
  if (p >= end || *p != ',') return nullptr;
  ++p;
  if (!parse_int(q)) return nullptr;
  skip_ws();
  if (p < end && *p == '\r') ++p;
  if (p < end && *p == '\n') return p + 1;
  if (p == end) return p;  // last line of the input, no trailing newline
  return nullptr;          // trailing garbage before the line ends
}

} // namespace csv
