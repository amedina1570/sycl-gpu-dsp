import numpy as np, matplotlib.pyplot as plt
NFFT, HOP, FS, FC = 8192, 2048, 20e6, 410e6
ded = np.fromfile('/home/user/crab_dedispersed.bin', dtype=np.float32)
nframes = ded.size // NFFT
ded = ded.reshape(nframes, NFFT)
prof = np.fromfile('/home/user/crab_profile.bin', dtype=np.float32)
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
plt.tight_layout(); plt.savefig('/home/user/crab_dedispersed.png', dpi=130)
print('saved crab_dedispersed.png')
