import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe: works whether or not a display is attached
import matplotlib.pyplot as plt

# Defaults match dmsearch.cpp's default output names, in the current directory.
# Usage: view_dmsearch.py [dm_snr.bin] [bestprofile.bin] [out.png]
curve_path = sys.argv[1] if len(sys.argv) > 1 else 'dm_snr.bin'
prof_path  = sys.argv[2] if len(sys.argv) > 2 else 'crab_bestprofile.bin'
out_png    = sys.argv[3] if len(sys.argv) > 3 else 'dm_search.png'

HOP, FS = 2048, 20e6
c = np.fromfile(curve_path, dtype=np.float32).reshape(-1,2)
dm, snr = c[:,0], c[:,1]
prof = np.fromfile(prof_path, dtype=np.float32)
t = np.arange(prof.size)*HOP/FS
best = dm[np.argmax(snr)]

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
a1.plot(dm, snr, lw=1.2)
a1.axvline(best, color='r', ls='--', label=f'peak DM={best:.2f}')
a1.axvline(56.7, color='g', ls=':', label='Crab literature 56.7')
a1.set_xlabel('DM (pc/cm$^3$)'); a1.set_ylabel('Pulse SNR')
a1.set_title('DM search'); a1.legend()
a2.plot(t, prof, lw=0.8)
a2.set_xlabel('Time (s)'); a2.set_ylabel('Power/chan')
a2.set_title(f'Best profile (DM={best:.2f})')
plt.tight_layout(); plt.savefig(out_png, dpi=130)
print(f'saved {out_png}')
