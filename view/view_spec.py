import numpy as np, matplotlib.pyplot as plt
NFFT, HOP, FS, FC = 8192, 2048, 20e6, 410e6
spec = np.fromfile('/home/user/crab_spectrogram.bin', dtype=np.float32)
nframes = spec.size // NFFT
spec = spec.reshape(nframes, NFFT)           # [time, freq]
t = np.arange(nframes)*HOP/FS
f = (FC + (np.arange(NFFT)-NFFT/2)*FS/NFFT)/1e6   # MHz, fftshifted
plt.figure(figsize=(12,6))
plt.imshow(spec.T, aspect='auto', origin='lower',
           extent=[t[0], t[-1], f[0], f[-1]],
           cmap='viridis', vmax=np.percentile(spec,99.5),
           vmin=np.percentile(spec,20))
plt.axvline(391475/FS, color='r', ls='--', lw=0.7, alpha=0.6)  # expected pulse
plt.xlabel('Time (s)'); plt.ylabel('Frequency (MHz)')
plt.title('Crab pulsar giant pulse — Dwingeloo 20 Msps')
plt.colorbar(label='Power (dB)')
plt.tight_layout(); plt.savefig('/home/user/crab_spectrogram.png', dpi=130)
print('saved crab_spectrogram.png')
