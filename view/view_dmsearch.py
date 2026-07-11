import numpy as np, matplotlib.pyplot as plt
HOP, FS = 2048, 20e6
c = np.fromfile('/home/user/dm_snr.bin', dtype=np.float32).reshape(-1,2)
dm, snr = c[:,0], c[:,1]
prof = np.fromfile('/home/user/crab_bestprofile.bin', dtype=np.float32)
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
plt.tight_layout(); plt.savefig('/home/user/dm_search.png', dpi=130)
print('saved dm_search.png')
