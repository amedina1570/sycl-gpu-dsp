/**
 * @file sigmf_meta.hpp
 * @brief Minimal SigMF `.sigmf-meta` sidecar parsing.
 *
 * Reads a flat JSON text and pulls scalar `"key": value` / `"key": "string"`
 * fields without a full JSON parser. Shared by iq2spectrogram.cpp,
 * radar_pulses.cpp, csv2sigmf.cpp, and their tests.
 */
#pragma once
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

/// @namespace sigmf
/// @brief Minimal SigMF `.sigmf-meta` JSON sidecar reader/writer support.
namespace sigmf {

/// Read an entire file into a string.
/// @param path Path to read.
/// @return File contents, or `std::nullopt` if it couldn't be opened.
inline std::optional<std::string> read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

/// Scrape a numeric field out of a SigMF JSON text, e.g. `"core:sample_rate"`.
///
/// SigMF keys like `"core:datatype"` contain a colon themselves, so the
/// search for the key/value separator must start *after* the closing quote
/// of the key, not from the key's start position.
/// @param text Full `.sigmf-meta` JSON text.
/// @param key Key name, without quotes (e.g. `"core:sample_rate"`).
/// @return The parsed value, or `std::nullopt` if the key is missing or its
/// value isn't a valid number (a malformed/null/string value is
/// deliberately *not* reported as `0.0` -- see the `strtod` end-pointer
/// check in the implementation).
inline std::optional<double> json_number(const std::string& text, const std::string& key) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return std::nullopt;
  pos = text.find(':', pos + key.size() + 2);
  if (pos == std::string::npos) return std::nullopt;
  // strtod returns 0.0 both for a genuine "0" and for "no digits parsed"
  // (e.g. the value is null/a string/malformed); check `end` to tell those
  // apart instead of silently treating a bad sidecar value as zero.
  const char* start = text.c_str() + pos + 1;
  char* end = nullptr;
  double v = std::strtod(start, &end);
  if (end == start) return std::nullopt;
  return v;
}

/// Scrape a string field out of a SigMF JSON text, e.g. `"core:datatype"`.
/// @param text Full `.sigmf-meta` JSON text.
/// @param key Key name, without quotes (e.g. `"core:datatype"`).
/// @return The unquoted string value, or `std::nullopt` if the key is
/// missing or its value isn't a quoted string.
inline std::optional<std::string> json_string(const std::string& text, const std::string& key) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return std::nullopt;
  pos = text.find(':', pos + key.size() + 2);
  if (pos == std::string::npos) return std::nullopt;
  auto q1 = text.find('"', pos);
  if (q1 == std::string::npos) return std::nullopt;
  auto q2 = text.find('"', q1 + 1);
  if (q2 == std::string::npos) return std::nullopt;
  return text.substr(q1 + 1, q2 - q1 - 1);
}

} // namespace sigmf
