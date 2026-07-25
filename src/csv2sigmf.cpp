/**
 * @file csv2sigmf.cpp
 * @brief Convert a vendor CSV I/Q dump into raw SigMF (`.sigmf-data` +
 * `.sigmf-meta`).
 *
 * One `"<I>,<Q>"` integer pair per line, optional header row, so the rest
 * of this project's tools (iq2spectrogram, radar_pulses, dedisp,
 * dmsearch, ...) can read it unchanged.
 *
 * Built for the NIST TN 2159 "AWS-3 LTE Waveforms" dataset's `IQ.csv`
 * format (header `"I Data,Q Data"`, signed-integer relative-unit samples,
 * no embedded sample rate/frequency -- hence `--fs` is required here,
 * unlike the SigMF-native tools which can fall back to a `.sigmf-meta`
 * sidecar), but works for any CSV in that shape.
 *
 * Reads in large blocks and hand-parses each line (csv_iq.hpp) rather than
 * `std::getline` + `std::stringstream`, which would be far too slow at the
 * hundreds-of-millions-of-lines scale this format shows up at. See
 * docs/TUTORIAL.md §2.5.
 */
#include "csv_iq.hpp"
#include "cli_util.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string input;
  std::string out_prefix;
  std::string datatype = "ci16_le";
  double fs = 0.0;  // required -- no sidecar to fall back to
  double fc = 0.0;
};

void usage(const char* prog) {
  printf(
    "usage: %s <input.csv> --fs HZ [options]\n"
    "      --fs HZ          sample rate (required -- CSV has no embedded rate)\n"
    "      --fc HZ          center frequency (default 0)\n"
    "      --datatype T     ci16_le | cf32_le (default ci16_le)\n"
    "  -o, --out PREFIX     output file prefix (default: <input stem>)\n",
    prog);
}

bool parse_args(int argc, char** argv, Args& a) {
  if (argc < 2) return false;
  std::string first = argv[1];
  if (first == "-h" || first == "--help") {
    usage(argv[0]);
    std::exit(0);
  }
  a.input = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); std::exit(1); }
      return argv[++i];
    };
    if (s == "-o" || s == "--out")       a.out_prefix = next(s.c_str());
    else if (s == "--fs")                a.fs = std::stod(next(s.c_str()));
    else if (s == "--fc")                a.fc = std::stod(next(s.c_str()));
    else if (s == "--datatype")          a.datatype = next(s.c_str());
    else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
    else { fprintf(stderr, "unknown option: %s\n", s.c_str()); return false; }
  }
  return true;
}

bool validate_args(const Args& a) {
  if (!cli::is_supported_datatype(a.datatype)) {
    fprintf(stderr, "unsupported datatype '%s' (supported: ci16_le, cf32_le)\n", a.datatype.c_str());
    return false;
  }
  if (a.fs <= 0.0) {
    fprintf(stderr, "--fs is required and must be positive (this CSV format has no embedded sample rate)\n");
    return false;
  }
  return true;
}

bool write_sigmf_meta(const fs::path& meta_path, const Args& a) {
  std::ofstream jf(meta_path);
  if (!jf) return false;
  jf << std::setprecision(17)
     << "{\n"
     << "  \"global\": {\n"
     << "    \"core:datatype\": \"" << a.datatype << "\",\n"
     << "    \"core:sample_rate\": " << a.fs << ",\n"
     << "    \"core:version\": \"1.0.0\"\n"
     << "  },\n"
     << "  \"captures\": [\n"
     << "    { \"core:sample_start\": 0, \"core:frequency\": " << a.fc << " }\n"
     << "  ],\n"
     << "  \"annotations\": []\n"
     << "}\n";
  return (bool)jf;
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { usage(argv[0]); return 1; }

  fs::path inpath(a.input);
  if (!fs::exists(inpath)) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }
  if (!validate_args(a)) return 1;
  if (a.out_prefix.empty()) a.out_prefix = inpath.stem().string();

  FILE* in = fopen(a.input.c_str(), "rb");
  if (!in) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }

  fs::path data_path = a.out_prefix + ".sigmf-data";
  fs::path meta_path = a.out_prefix + ".sigmf-meta";
  FILE* out = fopen(data_path.string().c_str(), "wb");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", data_path.string().c_str());
    fclose(in);
    return 1;
  }

  const bool is_ci16 = (a.datatype == "ci16_le");
  constexpr size_t READ_BLOCK = 64 * 1024 * 1024;
  constexpr size_t FLUSH_SAMPLES = 16 * 1024 * 1024;

  std::vector<char> readtmp(READ_BLOCK);
  std::vector<char> buf;  // carried-over partial line + newly read data
  std::vector<int16_t> out_i16;
  std::vector<float>   out_f32;
  if (is_ci16) out_i16.reserve(FLUSH_SAMPLES * 2);
  else         out_f32.reserve(FLUSH_SAMPLES * 2);

  size_t carry_len = 0, line_no = 0, nsamp = 0;
  bool first_line = true, ok = true;

  for (;;) {
    size_t got = fread(readtmp.data(), 1, READ_BLOCK, in);
    bool eof = (got < READ_BLOCK);  // short read -> no more data after this block

    buf.resize(carry_len + got);
    std::memcpy(buf.data() + carry_len, readtmp.data(), got);
    const char* base = buf.data();
    const char* buf_end = base + buf.size();

    // Scan only through the last newline in this block; the remainder is a
    // genuinely partial line, carried to the next iteration -- UNLESS this
    // is the final block (eof), in which case buf_end really is end of
    // file and csv::parse_iq_line's own end-of-buffer acceptance correctly
    // handles a last line with no trailing newline.
    const char* safe_end = buf_end;
    if (!eof) {
      const char* last_nl = nullptr;
      for (const char* p = buf_end; p > base;) {
        --p;
        if (*p == '\n') { last_nl = p; break; }
      }
      safe_end = last_nl ? last_nl + 1 : base;
    }

    const char* p = base;
    while (p < safe_end) {
      ++line_no;
      if (first_line) {
        first_line = false;
        // Header row heuristic: a real sample line starts with a digit or '-'.
        if (!(*p == '-' || (*p >= '0' && *p <= '9'))) {
          const char* nl = (const char*)std::memchr(p, '\n', (size_t)(safe_end - p));
          p = nl ? nl + 1 : safe_end;
          continue;
        }
      }

      long long iv = 0, qv = 0;
      const char* next = csv::parse_iq_line(p, safe_end, iv, qv);
      if (!next) {
        fprintf(stderr, "csv2sigmf: malformed line %zu in %s\n", line_no, a.input.c_str());
        ok = false;
        break;
      }
      p = next;

      if (is_ci16) {
        if (iv < -32768 || iv > 32767 || qv < -32768 || qv > 32767) {
          fprintf(stderr,
            "csv2sigmf: line %zu: value out of ci16_le range (I=%lld, Q=%lld) -- "
            "use --datatype cf32_le if this file needs a wider range\n",
            line_no, iv, qv);
          ok = false;
          break;
        }
        out_i16.push_back((int16_t)iv);
        out_i16.push_back((int16_t)qv);
      } else {
        out_f32.push_back((float)iv);
        out_f32.push_back((float)qv);
      }
      ++nsamp;
    }
    if (!ok) break;

    if (is_ci16 && out_i16.size() >= FLUSH_SAMPLES * 2) {
      fwrite(out_i16.data(), sizeof(int16_t), out_i16.size(), out);
      out_i16.clear();
    } else if (!is_ci16 && out_f32.size() >= FLUSH_SAMPLES * 2) {
      fwrite(out_f32.data(), sizeof(float), out_f32.size(), out);
      out_f32.clear();
    }

    carry_len = (size_t)(buf_end - safe_end);
    if (carry_len > 0) std::memmove(buf.data(), safe_end, carry_len);
    if (eof) break;
  }

  if (ok) {
    if (is_ci16 && !out_i16.empty()) fwrite(out_i16.data(), sizeof(int16_t), out_i16.size(), out);
    if (!is_ci16 && !out_f32.empty()) fwrite(out_f32.data(), sizeof(float), out_f32.size(), out);
  }

  fclose(in);
  bool write_ok = (fclose(out) == 0);

  if (!ok || !write_ok) {
    fprintf(stderr, "csv2sigmf: conversion failed\n");
    return 1;
  }
  if (nsamp == 0) {
    fprintf(stderr, "csv2sigmf: no samples found in %s (empty file, or every line was rejected)\n",
            a.input.c_str());
    return 1;
  }

  if (!write_sigmf_meta(meta_path, a)) {
    fprintf(stderr, "cannot write %s\n", meta_path.string().c_str());
    return 1;
  }

  printf("converted %zu complex samples (%.3f s at %.3f Msps)\n",
         nsamp, nsamp / a.fs, a.fs / 1e6);
  printf("wrote %s and %s\n", data_path.string().c_str(), meta_path.string().c_str());
  return 0;
}
