import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe: works whether or not a display is attached
import matplotlib.pyplot as plt

# Defaults match dedisp.cpp's default output names, in the current directory.
# Usage: view_dedisp.py [dedispersed.bin] [profile.bin] [out.png]
ded_path  = sys.argv[1] if len(sys.argv) > 1 else 'crab_dedispersed.bin'
prof_path = sys.argv[2] if len(sys.argv) > 2 else 'crab_profile.bin'
out_png   = sys.argv[3] if len(sys.argv) > 3 else 'crab_dedispersed.png'

NFFT, HOP, FS, FC = 8192, 2048, 20e6, 410e6
ded = np.fromfile(ded_path, dtype=np.float32)
nframes = ded.size // NFFT
ded = ded[:nframes*NFFT].reshape(nframes, NFFT)
prof = np.fromfile(prof_path, dtype=np.float32)
t = np.arange(nframes)*HOP/FS
f = (FC + (np.arange(NFFT)-NFFT/2)*FS/NFFT)/1e6

fig,(a1,a2)=plt.subplots(2,1,figsize=(12,8),sharex=True,
                         gridspec_kw={'height_ratios':[3,1]})
a1.imshow(ded.T, aspect='auto', origin='lower',
          extent=[t[0],t[-1],f[0],f[-1]], cmap='viridis',
          vmax=np.percentile(ded,99.5), vmin=np.percentile(ded,20))
a1.set_ylabel('Frequency (MHz)'); a1.set_title('Dedispersed (DM=56.7)')
a2.plot(t, prof, lw=0.8); a2.set_xlabel('Time (s)')
a2.set_ylabel('Power'); a2.set_title('Pulse profile (freq-summed)')
plt.tight_layout(); plt.savefig(out_png, dpi=130)
print(f'saved {out_png}')
