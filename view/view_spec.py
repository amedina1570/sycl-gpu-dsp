import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe: works whether or not a display is attached
import matplotlib.pyplot as plt

if len(sys.argv) >= 2:
    # General mode: driven by the .json sidecar written by iq2spectrogram.
    meta_path = sys.argv[1]
    with open(meta_path) as fh:
        meta = json.load(fh)
    bin_path = meta['bin']
    NFFT, HOP, FS, FC = meta['nfft'], meta['hop'], meta['fs'], meta['fc']
    out_png = sys.argv[2] if len(sys.argv) >= 3 else meta_path.rsplit('.', 1)[0] + '.png'
    mark_pulse = False
    title = f'Spectrogram — {bin_path}'
else:
    # Legacy demo mode: reproduces the Crab giant-pulse figure from the README.
    NFFT, HOP, FS, FC = 8192, 2048, 20e6, 410e6
    bin_path = '/home/user/crab_spectrogram.bin'
    out_png = '/home/user/crab_spectrogram.png'
    mark_pulse = True
    title = 'Crab pulsar giant pulse — Dwingeloo 20 Msps'

# Memory-map instead of loading the whole .bin: for a long recording (e.g.
# iq2spectrogram's chunked output on a multi-GB IQ file) the spectrogram
# itself can be tens of GB, far more than fits in RAM or is useful at the
# ~1500-pixel-wide figure below. Only ever materialize a downsampled copy.
spec_mm = np.memmap(bin_path, dtype=np.float32, mode='r')
nframes = spec_mm.size // NFFT
spec_mm = spec_mm[:nframes*NFFT].reshape(nframes, NFFT)  # [time, freq], still memmapped

MAX_TIME_PIXELS = 4000
if nframes > MAX_TIME_PIXELS:
    # Max-hold decimation: preserves narrow transients (e.g. a giant pulse)
    # that a naive stride/average would smear out or miss entirely.
    factor = -(-nframes // MAX_TIME_PIXELS)  # ceil division
    n_out = -(-nframes // factor)
    spec = np.empty((n_out, NFFT), dtype=np.float32)
    for i in range(n_out):
        lo, hi = i*factor, min((i+1)*factor, nframes)
        spec[i] = spec_mm[lo:hi].max(axis=0)
    hop_eff = HOP * factor
else:
    spec = np.asarray(spec_mm)
    hop_eff = HOP

t = np.arange(spec.shape[0])*hop_eff/FS
f = (FC + (np.arange(NFFT)-NFFT/2)*FS/NFFT)/1e6   # MHz, fftshifted
plt.figure(figsize=(12,6))
plt.imshow(spec.T, aspect='auto', origin='lower',
           extent=[t[0], t[-1], f[0], f[-1]],
           cmap='viridis', vmax=np.percentile(spec,99.5),
           vmin=np.percentile(spec,20))
if mark_pulse:
    plt.axvline(391475/FS, color='r', ls='--', lw=0.7, alpha=0.6)  # expected pulse
plt.xlabel('Time (s)'); plt.ylabel('Frequency (MHz)')
plt.title(title)
plt.colorbar(label='Power (dB)')
plt.tight_layout(); plt.savefig(out_png, dpi=130)
print(f'saved {out_png}')
