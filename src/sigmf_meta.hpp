// Minimal SigMF `.sigmf-meta` sidecar parsing: reads a flat JSON text and
// pulls scalar "key": value / "key": "string" fields without a full JSON
// parser. Shared by iq2spectrogram.cpp and its tests.
#pragma once
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace sigmf {

inline std::optional<std::string> read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

// SigMF keys like "core:datatype" contain a colon themselves, so the search
// for the key/value separator must start *after* the closing quote of the
// key, not from the key's start position.
inline std::optional<double> json_number(const std::string& text, const std::string& key) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return std::nullopt;
  pos = text.find(':', pos + key.size() + 2);
  if (pos == std::string::npos) return std::nullopt;
  return std::strtod(text.c_str() + pos + 1, nullptr);
}

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
