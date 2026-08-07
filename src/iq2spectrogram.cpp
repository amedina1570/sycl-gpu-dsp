/**
 * @file iq2spectrogram.cpp
 * @brief Single entry point: feed it an IQ file, get a spectrogram PNG.
 *
 * Combines stageA (load/parse) + stageC (windowed batched cuFFT via
 * SYCL/cuFFT interop) + view_spec.py (rendering) into one command. Reads
 * sample rate and center frequency from a SigMF `.sigmf-meta` sidecar when
 * present (falling back to CLI flags / defaults), supports `ci16_le` and
 * `cf32_le` IQ, and streams arbitrarily large files in bounded chunks. See
 * docs/TUTORIAL.md §2.4.
 */
#include "sigmf_meta.hpp"
#include "cli_util.hpp"
#include "sycl_dsp_math.hpp"
#include "dsp_math.hpp"
#include "cufft_interop.hpp"
#include "sycl_util.hpp"
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <string>
#include <type_traits>
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
  int    nfft = dsp::DEFAULT_NFFT;
  int    hop  = 0;        // 0 => resolved to nfft/4 (75% overlap) after parsing
  double fs   = 0.0;
  double fc   = 0.0;
  bool   plot = true;
  std::string viewer;     // path to view_spec.py; auto-located if empty
  std::string python = "python3";
  long   chunk_mb = 256;  // target GPU memory per chunk; controls streaming granularity
};

void usage(const char* prog) {
  printf(
    "usage: %s <iq-file> [options]\n"
    "  -o, --out PREFIX     output file prefix (default: <input stem>)\n"
    "      --nfft N         FFT size, power of two (default 8192)\n"
    "      --hop N          hop size in samples (default nfft/4, 75%% overlap)\n"
    "      --fs HZ          sample rate (default: read from .sigmf-meta, else 20e6)\n"
    "      --fc HZ          center frequency (default: read from .sigmf-meta, else 0)\n"
    "      --datatype T     ci16_le | cf32_le (default: read from .sigmf-meta, else ci16_le)\n"
    "      --no-plot        skip PNG rendering, only write .bin + .json\n"
    "      --viewer PATH    path to view_spec.py (default: auto-detected next to this binary)\n"
    "      --python PATH    python interpreter to invoke (default: python3)\n"
    "      --chunk-mb MB    stream this many MB of GPU memory worth of frames at a time\n"
    "                       (default 256; large files are processed in bounded chunks so\n"
    "                       neither host RAM nor GPU VRAM has to hold the whole spectrogram)\n",
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
    if (s == "-o" || s == "--out")   a.out_prefix = next(s.c_str());
    else if (s == "--nfft") {
      auto v = cli::parse_int(next(s.c_str()));
      if (!v) { fprintf(stderr, "invalid value for --nfft\n"); return false; }
      a.nfft = *v;
    }
    else if (s == "--hop") {
      auto v = cli::parse_int(next(s.c_str()));
      if (!v) { fprintf(stderr, "invalid value for --hop\n"); return false; }
      a.hop = *v;
    }
    else if (s == "--fs") {
      auto v = cli::parse_double(next(s.c_str()));
      if (!v) { fprintf(stderr, "invalid value for --fs\n"); return false; }
      a.fs = *v;
    }
    else if (s == "--fc") {
      auto v = cli::parse_double(next(s.c_str()));
      if (!v) { fprintf(stderr, "invalid value for --fc\n"); return false; }
      a.fc = *v;
    }
    else if (s == "--datatype")      a.datatype = next(s.c_str());
    else if (s == "--no-plot")       a.plot = false;
    else if (s == "--viewer")        a.viewer = next(s.c_str());
    else if (s == "--python")        a.python = next(s.c_str());
    else if (s == "--chunk-mb") {
      auto v = cli::parse_long(next(s.c_str()));
      if (!v) { fprintf(stderr, "invalid value for --chunk-mb\n"); return false; }
      a.chunk_mb = *v;
    }
    else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
    else { fprintf(stderr, "unknown option: %s\n", s.c_str()); return false; }
  }
  return true;
}

// Fill any fields the user left unset from a SigMF .sigmf-meta sidecar (if
// present), then apply the hard defaults and derived values.
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
  a.hop = cli::resolve_hop(a.nfft, a.hop);
  if (a.out_prefix.empty()) a.out_prefix = inpath.stem().string();
}

bool validate_args(const Args& a) {
  if (!cli::is_power_of_two(a.nfft)) {
    fprintf(stderr, "--nfft must be a positive power of two\n");
    return false;
  }
  if (a.hop <= 0) {
    fprintf(stderr, "--hop must be positive\n");
    return false;
  }
  if (a.fs <= 0.0) {
    fprintf(stderr, "--fs must be positive\n");
    return false;
  }
  if (!cli::is_supported_datatype(a.datatype)) {
    fprintf(stderr, "unsupported datatype '%s' (supported: ci16_le, cf32_le)\n",
            a.datatype.c_str());
    return false;
  }
  if (a.chunk_mb <= 0) {
    fprintf(stderr, "--chunk-mb must be positive\n");
    return false;
  }
  return true;
}

// Write the small .json sidecar describing the .bin (for the viewer).
// max_digits10 precision so fs/fc round-trip exactly (the default 6
// significant digits would corrupt e.g. fc = 1090.123e6).
bool write_sidecar_json(const fs::path& json_path, const fs::path& bin_path,
                        const Args& a, size_t nframes) {
  std::ofstream jf(json_path);
  jf << std::setprecision(17)
     << "{\n"
     << "  \"bin\": \""   << sigmf::json_escape(fs::absolute(bin_path).string()) << "\",\n"
     << "  \"nfft\": "    << a.nfft << ",\n"
     << "  \"hop\": "     << a.hop << ",\n"
     << "  \"fs\": "      << a.fs << ",\n"
     << "  \"fc\": "      << a.fc << ",\n"
     << "  \"nframes\": " << nframes << "\n"
     << "}\n";
  return jf.good();
}

// Run `python viewer json png` directly (fork/exec, no shell), so paths with
// spaces or shell metacharacters can never break or inject into a command
// line. Returns the viewer's exit code, or -1 if it could not be run.
int run_viewer(const std::string& python, const fs::path& viewer,
               const fs::path& json_path, const fs::path& png_path) {
  pid_t pid = fork();
  if (pid < 0) { perror("fork"); return -1; }
  if (pid == 0) {
    execlp(python.c_str(), python.c_str(), viewer.c_str(),
           json_path.c_str(), png_path.c_str(), (char*)nullptr);
    fprintf(stderr, "cannot exec %s\n", python.c_str());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return -1; }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { usage(argv[0]); return 1; }

  fs::path inpath(a.input);
  if (!fs::exists(inpath)) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }

  resolve_metadata(a, inpath);
  if (!validate_args(a)) return 1;

  const int NFFT = a.nfft, HOP = a.hop;

  sycl::queue q = cufft_util::make_inorder_queue();
  if (!cufft_util::require_cuda_backend(q)) return 1;

  // --- Size the job from the file length alone (no full-file read yet) ---
  std::ifstream f(a.input, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }
  std::streamsize bytes = f.tellg();
  const size_t bytes_per_sample = cli::bytes_per_complex_sample(a.datatype);

  size_t nsamp = bytes / bytes_per_sample;
  if (nsamp < (size_t)NFFT) { fprintf(stderr, "file too short for NFFT=%d\n", NFFT); return 1; }
  size_t nframes = (nsamp - NFFT) / HOP + 1;

  // --- Chunk sizing: stream `chunk_frames` frames at a time so the batch
  // and spectrogram device buffers (and their host-side copies) stay
  // bounded regardless of file size, instead of scaling with the whole
  // file. See cli_util.hpp for the byte-budget math. ---
  size_t chunk_frames = cli::resolve_chunk_frames(a.chunk_mb, NFFT, nframes);
  size_t nchunks = (nframes + chunk_frames - 1) / chunk_frames;
  size_t max_samples_per_chunk = cli::chunk_sample_count(chunk_frames, HOP, NFFT);
  printf("samples=%zu  frames=%zu  nfft=%d  hop=%d  fs=%.0f  fc=%.0f  datatype=%s\n",
         nsamp, nframes, NFFT, HOP, a.fs, a.fc, a.datatype.c_str());
  printf("streaming %zu frame(s)/chunk (~%ld MB GPU budget), %zu chunk(s) total\n",
         chunk_frames, a.chunk_mb, nchunks);

  // --- Device + host buffers, sized for the largest chunk and reused
  // across chunks (the final chunk just uses a prefix of them). ---
  cufftComplex* d_batch = sycl_util::malloc_device_checked<cufftComplex>(chunk_frames*(size_t)NFFT, q, "d_batch");
  float*        d_spec  = sycl_util::malloc_device_checked<float>(chunk_frames*(size_t)NFFT, q, "d_spec");
  int16_t* d_raw_i16 = nullptr; float* d_raw_f32 = nullptr;
  std::vector<int16_t> raw_i16;
  std::vector<float>   raw_f32;
  const bool is_ci16 = (a.datatype == "ci16_le");
  if (is_ci16) {
    raw_i16.resize(max_samples_per_chunk * 2);
    d_raw_i16 = sycl_util::malloc_device_checked<int16_t>(max_samples_per_chunk*2, q, "d_raw_i16");
  } else {
    raw_f32.resize(max_samples_per_chunk * 2);
    d_raw_f32 = sycl_util::malloc_device_checked<float>(max_samples_per_chunk*2, q, "d_raw_f32");
  }
  std::vector<float> spec_chunk(chunk_frames * (size_t)NFFT);

  fs::path bin_path  = a.out_prefix + "_spectrogram.bin";
  fs::path json_path = a.out_prefix + "_spectrogram.json";
  fs::path png_path  = a.out_prefix + "_spectrogram.png";
  std::ofstream bin_out(bin_path, std::ios::binary);
  if (!bin_out) { fprintf(stderr, "cannot write %s\n", bin_path.string().c_str()); return 1; }

  // One implementation of "read chunk from disk -> upload -> fused framing +
  // decode + Hann window", instantiated for both raw sample types; the only
  // per-type difference is dsp::decode_sample's overload.
  auto stage_chunk = [&](auto& host_raw, auto* d_raw,
                         size_t samples_here, size_t frames_here) -> bool {
    using Raw = std::remove_pointer_t<decltype(d_raw)>;
    const size_t want_bytes = samples_here * 2 * sizeof(Raw);
    f.read(reinterpret_cast<char*>(host_raw.data()), want_bytes);
    if (!f || (size_t)f.gcount() != (size_t)want_bytes) {
      fprintf(stderr, "short read from %s (wanted %zu bytes, got %zd)\n",
              a.input.c_str(), want_bytes, (ssize_t)f.gcount());
      return false;
    }
    q.memcpy(d_raw, host_raw.data(), want_bytes).wait();
    q.parallel_for(sycl::range<2>{frames_here, (size_t)NFFT}, [=](sycl::id<2> id){
      size_t fr = id[0], n = id[1];
      size_t s  = fr*HOP + n;
      float w = dsp::hann_coeff(n, NFFT);
      cufftComplex cc;
      cc.x = dsp::decode_sample(d_raw[2*s])   * w;
      cc.y = dsp::decode_sample(d_raw[2*s+1]) * w;
      d_batch[fr*NFFT + n] = cc;
    }).wait();
    return true;
  };

  // cuFFT plan, reused across chunks: every chunk has `chunk_frames` frames
  // except possibly the last, so at most two plans are ever created.
  cufftHandle plan;
  int planned_batch = 0;

  for (size_t c = 0; c < nchunks; ++c) {
    size_t f0 = c * chunk_frames;
    size_t frames_here = std::min(chunk_frames, nframes - f0);
    size_t samples_here = cli::chunk_sample_count(frames_here, HOP, NFFT);

    // Chunks overlap by NFFT-HOP samples at their boundary; re-reading that
    // small overlap from disk (via an absolute seek) is simpler than
    // carrying it over by hand and, for realistic chunk sizes, negligible
    // I/O overhead.
    f.seekg((std::streamoff)cli::chunk_start_sample(f0, HOP) * bytes_per_sample);

    bool ok = is_ci16 ? stage_chunk(raw_i16, d_raw_i16, samples_here, frames_here)
                      : stage_chunk(raw_f32, d_raw_f32, samples_here, frames_here);
    if (!ok) return 1;

    // --- Batched cuFFT via interop ---
    if ((int)frames_here != planned_batch) {
      if (planned_batch != 0) cufftDestroy(plan);
      CUFFT_CHECK(cufftPlan1d(&plan, NFFT, CUFFT_C2C, (int)frames_here));
      planned_batch = (int)frames_here;
    }
    cufftResult fft_status = CUFFT_SUCCESS;
    cufft_util::enqueue_exec_c2c_forward(q, plan, d_batch, &fft_status);

    // --- Magnitude (dB) + fftshift, 2D range ---
    q.parallel_for(sycl::range<2>{frames_here, (size_t)NFFT}, [=](sycl::id<2> id){
      size_t fr = id[0], k = id[1];
      cufftComplex cc = d_batch[fr*NFFT + k];
      float p = cc.x*cc.x + cc.y*cc.y;
      float db = dsp::db_from_power(p);
      size_t ks = (k + NFFT/2) % NFFT;    // fftshift: DC -> center
      d_spec[fr*NFFT + ks] = db;
    });
    q.wait();
    if (fft_status != CUFFT_SUCCESS) {
      fprintf(stderr, "cuFFT error %d executing chunk %zu\n", fft_status, c+1);
      return 1;
    }

    q.memcpy(spec_chunk.data(), d_spec, frames_here*(size_t)NFFT*sizeof(float)).wait();
    bin_out.write(reinterpret_cast<char*>(spec_chunk.data()), frames_here*(size_t)NFFT*sizeof(float));
    printf("chunk %zu/%zu: frames %zu..%zu\n", c+1, nchunks, f0, f0+frames_here);
  }
  bin_out.close();
  if (!bin_out) { fprintf(stderr, "write to %s failed\n", bin_path.string().c_str()); return 1; }

  if (planned_batch != 0) cufftDestroy(plan);
  sycl::free(d_batch, q); sycl::free(d_spec, q);
  if (d_raw_i16) sycl::free(d_raw_i16, q);
  if (d_raw_f32) sycl::free(d_raw_f32, q);

  if (!write_sidecar_json(json_path, bin_path, a, nframes)) {
    fprintf(stderr, "cannot write %s\n", json_path.string().c_str());
    return 1;
  }
  printf("wrote %s and %s\n", bin_path.string().c_str(), json_path.string().c_str());

  if (!a.plot) return 0;

  // --- Render PNG via view_spec.py ---
  fs::path viewer = a.viewer;
  if (viewer.empty()) {
    fs::path exe = fs::weakly_canonical(fs::path(argv[0]));
    viewer = exe.parent_path().parent_path() / "view" / "view_spec.py"; // <repo>/build/.. -> <repo>/view
  }
  if (!fs::exists(viewer)) {
    fprintf(stderr, "viewer script not found at %s (pass --viewer PATH); .bin/.json written, skipping plot\n",
            viewer.string().c_str());
    return 0;
  }

  int rc = run_viewer(a.python, viewer, json_path, png_path);
  if (rc != 0) {
    fprintf(stderr, "plotting failed (exit %d): %s %s %s %s\n", rc, a.python.c_str(),
            viewer.string().c_str(), json_path.string().c_str(), png_path.string().c_str());
    return 1;
  }
  printf("spectrogram image: %s\n", fs::absolute(png_path).string().c_str());
  return 0;
}
