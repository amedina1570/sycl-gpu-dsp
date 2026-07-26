/**
 * @file radar_pulses.cpp
 * @brief GPU-accelerated pulse-train detection for RADAR-like SigMF IQ
 * captures.
 *
 * A port of view/view_radar_pulses.py's numeric core (envelope, threshold,
 * edge detection, width/PRI/PRF/duty-cycle) to SYCL, so segments larger
 * than comfortably fit in NumPy can be processed.
 *
 * Reads sample rate / center frequency / datatype from a `.sigmf-meta`
 * sidecar when present (falling back to CLI flags / defaults, same
 * convention as iq2spectrogram.cpp), and writes a JSON sidecar + envelope
 * `.bin` that view_radar_pulses.py can plot directly (its `--json` mode)
 * instead of recomputing detection in Python. view_radar_pulses.py keeps
 * working standalone on raw IQ files exactly as it always has -- this is
 * an additional, faster path for longer segments, not a replacement. See
 * docs/TUTORIAL.md §3.2.
 */
#include "sigmf_meta.hpp"
#include "cli_util.hpp"
#include "dsp_math.hpp"
#include "radar_lib.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <optional>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string input;
  std::string out_prefix;
  std::string datatype;   // "ci16_le" or "cf32_le"
  long   offset = 0;       // starting complex sample
  double duration = 5e-3;  // seconds; used to derive nsamp when nsamp is unset
  long   nsamp = 0;        // 0 => resolved from duration*fs after metadata is known
  double fs = 0.0;
  double fc = 0.0;
  float  threshold_frac = 0.5f;
  int    smooth_samples = 5;
  long   min_pulse_samples = 3;
};

void usage(const char* prog) {
  printf(
    "usage: %s <iq-file> [options]\n"
    "  -o, --out PREFIX          output file prefix (default: <input stem>)\n"
    "      --offset N            starting sample, complex samples (default 0)\n"
    "      --duration SEC        segment length in seconds (default 5e-3)\n"
    "      --nsamp N             segment length in samples, overrides --duration\n"
    "      --fs HZ               sample rate (default: read from .sigmf-meta, else 20e6)\n"
    "      --fc HZ               center frequency (default: read from .sigmf-meta, else 0)\n"
    "      --datatype T          ci16_le | cf32_le (default: read from .sigmf-meta, else ci16_le)\n"
    "      --threshold-frac F    detection threshold, fraction between noise floor and peak (default 0.5)\n"
    "      --smooth-samples N    envelope smoothing window in samples (default 5)\n"
    "      --min-pulse-samples N discard detections shorter than this many samples (default 3)\n",
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
    if (s == "-o" || s == "--out")              a.out_prefix = next(s.c_str());
    else if (s == "--offset")                   a.offset = std::stol(next(s.c_str()));
    else if (s == "--duration")                 a.duration = std::stod(next(s.c_str()));
    else if (s == "--nsamp")                    a.nsamp = std::stol(next(s.c_str()));
    else if (s == "--fs")                       a.fs = std::stod(next(s.c_str()));
    else if (s == "--fc")                       a.fc = std::stod(next(s.c_str()));
    else if (s == "--datatype")                 a.datatype = next(s.c_str());
    else if (s == "--threshold-frac")           a.threshold_frac = std::stof(next(s.c_str()));
    else if (s == "--smooth-samples")           a.smooth_samples = std::stoi(next(s.c_str()));
    else if (s == "--min-pulse-samples")        a.min_pulse_samples = std::stol(next(s.c_str()));
    else if (s == "-h" || s == "--help")        { usage(argv[0]); std::exit(0); }
    else { fprintf(stderr, "unknown option: %s\n", s.c_str()); return false; }
  }
  return true;
}

// Fill any fields the user left unset from a SigMF .sigmf-meta sidecar (if
// present), then apply the hard defaults and derived values. Same pattern
// as iq2spectrogram.cpp's resolve_metadata.
void resolve_metadata(Args& a, const fs::path& inpath) {
  fs::path meta_path;
  {
    std::string s = inpath.string();
    auto pos = s.rfind(".sigmf-data");
    meta_path = (pos != std::string::npos) ? s.substr(0, pos) + ".sigmf-meta"
                                            : s + ".sigmf-meta";
  }
  if (auto text = sigmf::read_file(meta_path.string())) {
    if (a.fs == 0.0)        if (auto v = sigmf::json_number(*text, "core:sample_rate")) a.fs = *v;
    if (a.fc == 0.0)        if (auto v = sigmf::json_number(*text, "core:frequency"))   a.fc = *v;
    if (a.datatype.empty()) if (auto v = sigmf::json_string(*text, "core:datatype"))    a.datatype = *v;
    printf("read metadata from %s\n", meta_path.string().c_str());
  }
  if (a.fs == 0.0) a.fs = 20e6;
  if (a.datatype.empty()) a.datatype = "ci16_le";
  if (a.nsamp == 0) a.nsamp = (long)std::llround(a.duration * a.fs);
  if (a.out_prefix.empty()) a.out_prefix = inpath.stem().string();
}

bool validate_args(const Args& a) {
  if (!cli::is_supported_datatype(a.datatype)) {
    fprintf(stderr, "unsupported datatype '%s' (supported: ci16_le, cf32_le)\n", a.datatype.c_str());
    return false;
  }
  if (a.offset < 0) { fprintf(stderr, "--offset must be non-negative\n"); return false; }
  if (a.nsamp <= 0) { fprintf(stderr, "segment length must be positive (check --duration/--nsamp)\n"); return false; }
  if (a.threshold_frac <= 0.0f || a.threshold_frac >= 1.0f) {
    fprintf(stderr, "--threshold-frac must be in (0, 1)\n");
    return false;
  }
  if (a.smooth_samples < 1) { fprintf(stderr, "--smooth-samples must be >= 1\n"); return false; }
  if (a.min_pulse_samples < 1) { fprintf(stderr, "--min-pulse-samples must be >= 1\n"); return false; }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { usage(argv[0]); return 1; }

  fs::path inpath(a.input);
  if (!fs::exists(inpath)) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }

  resolve_metadata(a, inpath);
  if (!validate_args(a)) return 1;

  // Async exceptions (e.g. a failed copy/kernel) are logged instead of being
  // silently dropped -- q.wait() alone does not rethrow them.
  sycl::queue q{
      [](sycl::exception_list exceptions) {
        for (const std::exception_ptr& e : exceptions) {
          try {
            std::rethrow_exception(e);
          } catch (const sycl::exception& ex) {
            std::fprintf(stderr, "asynchronous SYCL exception: %s\n", ex.what());
          }
        }
      },
      sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  const size_t bytes_per_sample = cli::bytes_per_complex_sample(a.datatype);
  const bool is_ci16 = (a.datatype == "ci16_le");

  std::ifstream f(a.input, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }
  std::streamsize file_bytes = f.tellg();
  size_t file_nsamp = (size_t)file_bytes / bytes_per_sample;
  if ((size_t)a.offset >= file_nsamp) {
    fprintf(stderr, "--offset %ld is past end of file (%zu complex samples)\n", a.offset, file_nsamp);
    return 1;
  }
  size_t nsamp = std::min((size_t)a.nsamp, file_nsamp - (size_t)a.offset);
  if (nsamp < 2) { fprintf(stderr, "segment too short (need at least 2 samples)\n"); return 1; }

  // Device-memory guard: 3 buffers of nsamp-scale (raw, envelope, smoothed).
  // Checked against total device memory rather than currently-free memory --
  // SYCL has no portable "free" memory query (same as dedisp.cpp/dmsearch.cpp).
  {
    size_t total_bytes = q.get_device().get_info<sycl::info::device::global_mem_size>();
    size_t raw_bytes = nsamp * 2 * (is_ci16 ? sizeof(int16_t) : sizeof(float));
    size_t device_bytes_needed = raw_bytes + 2 * nsamp * sizeof(float); // + d_env + d_smooth
    if (device_bytes_needed > (size_t)(total_bytes * 0.9)) {
      fprintf(stderr,
        "radar_pulses: this segment needs ~%.2f GB of GPU memory (device has ~%.2f GB total) -- "
        "reduce --duration/--nsamp.\n",
        device_bytes_needed / 1e9, total_bytes / 1e9);
      return 1;
    }
  }

  f.seekg((std::streamoff)((size_t)a.offset * bytes_per_sample));
  std::vector<float> envelope;
  if (is_ci16) {
    std::vector<int16_t> raw(nsamp * 2);
    f.read(reinterpret_cast<char*>(raw.data()), nsamp * 2 * sizeof(int16_t));
    if (!f || (size_t)f.gcount() != nsamp * 2 * sizeof(int16_t)) {
      fprintf(stderr, "short read from %s\n", a.input.c_str());
      return 1;
    }
    envelope = dsp::gpu_smoothed_envelope(q, raw, a.smooth_samples);
  } else {
    std::vector<float> raw(nsamp * 2);
    f.read(reinterpret_cast<char*>(raw.data()), nsamp * 2 * sizeof(float));
    if (!f || (size_t)f.gcount() != nsamp * 2 * sizeof(float)) {
      fprintf(stderr, "short read from %s\n", a.input.c_str());
      return 1;
    }
    envelope = dsp::gpu_smoothed_envelope(q, raw, a.smooth_samples);
  }

  dsp::PulseDetection det = dsp::detect_pulses(envelope, a.threshold_frac, (size_t)a.min_pulse_samples);
  dsp::PulseStats stats = dsp::pulse_stats(det, a.fs);

  printf("pulses detected: %d\n", stats.n_pulses);
  if (stats.n_pulses) {
    printf("  pulse width: %.3f us (+/- %.3f us)\n",
           stats.pulse_width_mean_s * 1e6, stats.pulse_width_std_s * 1e6);
  }
  if (stats.has_pri) {
    printf("  PRI: %.3f us (+/- %.3f us)\n", stats.pri_mean_s * 1e6, stats.pri_std_s * 1e6);
    printf("  PRF: %.1f Hz\n", stats.prf_hz);
    printf("  duty cycle: %.2f%%\n", stats.duty_cycle * 100.0f);
  } else {
    printf("  fewer than 2 pulses in this segment -- cannot compute PRI; try a longer --duration\n");
  }

  fs::path json_path = a.out_prefix + "_radar.json";
  fs::path bin_path  = a.out_prefix + "_radar_envelope.bin";

  std::ofstream bin_out(bin_path, std::ios::binary);
  if (!bin_out) { fprintf(stderr, "cannot write %s\n", bin_path.string().c_str()); return 1; }
  bin_out.write(reinterpret_cast<char*>(envelope.data()), envelope.size() * sizeof(float));
  bin_out.close();
  if (!bin_out) { fprintf(stderr, "write to %s failed\n", bin_path.string().c_str()); return 1; }

  // Sidecar JSON so view_radar_pulses.py can plot from precomputed results
  // instead of recomputing detection in NumPy (see its --json mode). Fields
  // that are undefined below their pulse-count threshold (matching Python's
  // None) are written as JSON null rather than a misleading 0.
  std::ofstream jf(json_path);
  jf << std::setprecision(17) << "{\n";
  jf << "  \"iq_input\": \""     << fs::absolute(inpath).string()   << "\",\n";
  jf << "  \"envelope_bin\": \"" << fs::absolute(bin_path).string() << "\",\n";
  jf << "  \"offset\": "  << a.offset << ",\n";
  jf << "  \"nsamp\": "   << nsamp    << ",\n";
  jf << "  \"fs\": "      << a.fs     << ",\n";
  jf << "  \"fc\": "      << a.fc     << ",\n";
  jf << "  \"datatype\": \"" << a.datatype << "\",\n";
  jf << "  \"threshold_frac\": " << a.threshold_frac << ",\n";
  jf << "  \"threshold\": "      << det.threshold    << ",\n";
  jf << "  \"smooth_samples\": "     << a.smooth_samples     << ",\n";
  jf << "  \"min_pulse_samples\": " << a.min_pulse_samples << ",\n";
  jf << "  \"n_pulses\": " << stats.n_pulses << ",\n";
  jf << "  \"pulse_width_mean_s\": "; if (stats.n_pulses) jf << stats.pulse_width_mean_s; else jf << "null"; jf << ",\n";
  jf << "  \"pulse_width_std_s\": ";  if (stats.n_pulses) jf << stats.pulse_width_std_s;  else jf << "null"; jf << ",\n";
  jf << "  \"pri_mean_s\": "; if (stats.has_pri) jf << stats.pri_mean_s; else jf << "null"; jf << ",\n";
  jf << "  \"pri_std_s\": ";  if (stats.has_pri) jf << stats.pri_std_s;  else jf << "null"; jf << ",\n";
  jf << "  \"prf_hz\": ";     if (stats.has_pri) jf << stats.prf_hz;     else jf << "null"; jf << ",\n";
  jf << "  \"duty_cycle\": "; if (stats.has_pri) jf << stats.duty_cycle; else jf << "null"; jf << ",\n";
  jf << "  \"rising\": [";
  for (size_t i = 0; i < det.rising.size(); ++i) jf << (i ? "," : "") << det.rising[i];
  jf << "],\n";
  jf << "  \"falling\": [";
  for (size_t i = 0; i < det.falling.size(); ++i) jf << (i ? "," : "") << det.falling[i];
  jf << "]\n";
  jf << "}\n";
  if (!jf) { fprintf(stderr, "cannot write %s\n", json_path.string().c_str()); return 1; }

  printf("wrote %s and %s\n", json_path.string().c_str(), bin_path.string().c_str());
  return 0;
}
