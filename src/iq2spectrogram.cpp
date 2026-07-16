// iq2spectrogram: single entry point — feed it an IQ file, get a spectrogram PNG.
//
// Combines stageA (load/parse) + stageC (windowed batched cuFFT via SYCL/cuFFT
// interop) + view_spec.py (rendering) into one command. Reads sample rate and
// center frequency from a SigMF `.sigmf-meta` sidecar when present (falling
// back to CLI flags / defaults), supports ci16_le and cf32_le IQ, and shells
// out to the Python viewer to produce the final PNG.
#include "sigmf_meta.hpp"
#include "cli_util.hpp"
#include "sycl_dsp_math.hpp"
#include "dsp_math.hpp"
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <optional>
#include <filesystem>
#include <exception>

namespace fs = std::filesystem;

#define CUFFT_CHECK(x) do { cufftResult r=(x); \
  if(r!=CUFFT_SUCCESS){printf("cuFFT error %d at line %d\n",r,__LINE__);return 1;} }while(0)

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
      if (i + 1 >= argc) { printf("missing value for %s\n", flag); std::exit(1); }
      return argv[++i];
    };
    if (s == "-o" || s == "--out")   a.out_prefix = next(s.c_str());
    else if (s == "--nfft")          a.nfft = std::stoi(next(s.c_str()));
    else if (s == "--hop")           a.hop  = std::stoi(next(s.c_str()));
    else if (s == "--fs")            a.fs   = std::stod(next(s.c_str()));
    else if (s == "--fc")            a.fc   = std::stod(next(s.c_str()));
    else if (s == "--datatype")      a.datatype = next(s.c_str());
    else if (s == "--no-plot")       a.plot = false;
    else if (s == "--viewer")        a.viewer = next(s.c_str());
    else if (s == "--python")        a.python = next(s.c_str());
    else if (s == "--chunk-mb")      a.chunk_mb = std::stol(next(s.c_str()));
    else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
    else { printf("unknown option: %s\n", s.c_str()); return false; }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { usage(argv[0]); return 1; }

  fs::path inpath(a.input);
  if (!fs::exists(inpath)) { printf("cannot open %s\n", a.input.c_str()); return 1; }

  // --- Resolve fs/fc/datatype from a SigMF .sigmf-meta sidecar, if present ---
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

  if (!cli::is_power_of_two(a.nfft)) {
    printf("--nfft must be a positive power of two\n");
    return 1;
  }
  if (a.hop <= 0) {
    printf("--hop must be positive\n");
    return 1;
  }
  if (!cli::is_supported_datatype(a.datatype)) {
    printf("unsupported datatype '%s' (supported: ci16_le, cf32_le)\n", a.datatype.c_str());
    return 1;
  }
  if (a.chunk_mb <= 0) {
    printf("--chunk-mb must be positive\n");
    return 1;
  }

  const int NFFT = a.nfft, HOP = a.hop;

  sycl::queue q{
    [](sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception& ex) {
          std::printf("asynchronous SYCL exception: %s\n", ex.what());
        }
      }
    },
    sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  if (q.get_device().get_backend() != sycl::backend::cuda) {
    printf("cuFFT interop requires a SYCL queue backed by the CUDA backend\n");
    return 1;
  }

  // --- Size the job from the file length alone (no full-file read yet) ---
  std::ifstream f(a.input, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", a.input.c_str()); return 1; }
  std::streamsize bytes = f.tellg();
  const size_t bytes_per_sample = cli::bytes_per_complex_sample(a.datatype);

  size_t nsamp = bytes / bytes_per_sample;
  if (nsamp < (size_t)NFFT) { printf("file too short for NFFT=%d\n", NFFT); return 1; }
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
  cufftComplex* d_batch = sycl::malloc_device<cufftComplex>(chunk_frames*(size_t)NFFT, q);
  float*        d_spec  = sycl::malloc_device<float>(chunk_frames*(size_t)NFFT, q);
  int16_t* d_raw_i16 = nullptr; float* d_raw_f32 = nullptr;
  std::vector<int16_t> raw_i16;
  std::vector<float>   raw_f32;
  if (a.datatype == "ci16_le") {
    raw_i16.resize(max_samples_per_chunk * 2);
    d_raw_i16 = sycl::malloc_device<int16_t>(max_samples_per_chunk*2, q);
  } else {
    raw_f32.resize(max_samples_per_chunk * 2);
    d_raw_f32 = sycl::malloc_device<float>(max_samples_per_chunk*2, q);
  }
  std::vector<float> spec_chunk(chunk_frames * (size_t)NFFT);

  fs::path bin_path  = a.out_prefix + "_spectrogram.bin";
  fs::path json_path = a.out_prefix + "_spectrogram.json";
  fs::path png_path  = a.out_prefix + "_spectrogram.png";
  std::ofstream bin_out(bin_path, std::ios::binary);

  for (size_t c = 0; c < nchunks; ++c) {
    size_t f0 = c * chunk_frames;
    size_t frames_here = std::min(chunk_frames, nframes - f0);
    size_t samples_here = cli::chunk_sample_count(frames_here, HOP, NFFT);

    // Chunks overlap by NFFT-HOP samples at their boundary; re-reading that
    // small overlap from disk (via an absolute seek) is simpler than
    // carrying it over by hand and, for realistic chunk sizes, negligible
    // I/O overhead.
    f.seekg((std::streamoff)(f0 * (size_t)HOP) * bytes_per_sample);

    if (a.datatype == "ci16_le") {
      f.read(reinterpret_cast<char*>(raw_i16.data()), samples_here*2*sizeof(int16_t));
      q.memcpy(d_raw_i16, raw_i16.data(), samples_here*2*sizeof(int16_t)).wait();
      q.parallel_for(sycl::range<2>{frames_here, (size_t)NFFT}, [=](sycl::id<2> id){
        size_t fr = id[0], n = id[1];
        size_t s  = fr*HOP + n;
        float w = dsp::hann_coeff(n, NFFT);
        float i = dsp::decode_i16(d_raw_i16[2*s]);
        float qd= dsp::decode_i16(d_raw_i16[2*s+1]);
        cufftComplex cc; cc.x = i*w; cc.y = qd*w;
        d_batch[fr*NFFT + n] = cc;
      }).wait();
    } else {
      f.read(reinterpret_cast<char*>(raw_f32.data()), samples_here*2*sizeof(float));
      q.memcpy(d_raw_f32, raw_f32.data(), samples_here*2*sizeof(float)).wait();
      q.parallel_for(sycl::range<2>{frames_here, (size_t)NFFT}, [=](sycl::id<2> id){
        size_t fr = id[0], n = id[1];
        size_t s  = fr*HOP + n;
        float w = dsp::hann_coeff(n, NFFT);
        cufftComplex cc; cc.x = d_raw_f32[2*s]*w; cc.y = d_raw_f32[2*s+1]*w;
        d_batch[fr*NFFT + n] = cc;
      }).wait();
    }

    // --- Batched cuFFT via interop ---
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, NFFT, CUFFT_C2C, (int)frames_here));
    q.submit([&](sycl::handler& cgh){
      cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih){
        auto stream = ih.get_native_queue<sycl::backend::cuda>();
        cufftResult r = cufftSetStream(plan, stream);
        if (r != CUFFT_SUCCESS) {
          printf("cuFFT error %d in cufftSetStream\n", r);
          return;
        }
        r = cufftExecC2C(plan, d_batch, d_batch, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS) {
          printf("cuFFT error %d in cufftExecC2C\n", r);
        }
      });
    });

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
    cufftDestroy(plan);

    q.memcpy(spec_chunk.data(), d_spec, frames_here*(size_t)NFFT*sizeof(float)).wait();
    bin_out.write(reinterpret_cast<char*>(spec_chunk.data()), frames_here*(size_t)NFFT*sizeof(float));
    printf("chunk %zu/%zu: frames %zu..%zu\n", c+1, nchunks, f0, f0+frames_here);
  }
  bin_out.close();

  sycl::free(d_batch, q); sycl::free(d_spec, q);
  if (d_raw_i16) sycl::free(d_raw_i16, q);
  if (d_raw_f32) sycl::free(d_raw_f32, q);

  // --- Write a small .json sidecar describing the .bin (for the viewer) ---
  std::ofstream jf(json_path);
  jf << "{\n"
     << "  \"bin\": \""   << fs::absolute(bin_path).string() << "\",\n"
     << "  \"nfft\": "    << NFFT << ",\n"
     << "  \"hop\": "     << HOP << ",\n"
     << "  \"fs\": "      << a.fs << ",\n"
     << "  \"fc\": "      << a.fc << ",\n"
     << "  \"nframes\": " << nframes << "\n"
     << "}\n";
  jf.close();
  printf("wrote %s and %s\n", bin_path.string().c_str(), json_path.string().c_str());

  if (!a.plot) return 0;

  // --- Render PNG via view_spec.py ---
  fs::path viewer = a.viewer;
  if (viewer.empty()) {
    fs::path exe = fs::weakly_canonical(fs::path(argv[0]));
    viewer = exe.parent_path().parent_path() / "view" / "view_spec.py"; // <repo>/build/.. -> <repo>/view
  }
  if (!fs::exists(viewer)) {
    printf("viewer script not found at %s (pass --viewer PATH); .bin/.json written, skipping plot\n",
           viewer.string().c_str());
    return 0;
  }

  std::string cmd = a.python + " \"" + viewer.string() + "\" \"" + json_path.string() +
                     "\" \"" + png_path.string() + "\"";
  int rc = std::system(cmd.c_str());
  if (rc != 0) { printf("plotting failed (exit %d): %s\n", rc, cmd.c_str()); return 1; }
  printf("spectrogram image: %s\n", fs::absolute(png_path).string().c_str());
  return 0;
}
