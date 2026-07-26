#!/usr/bin/env python3
# Welch power spectral density estimate for a segment of a SigMF IQ file:
# averages many overlapping periodograms (scipy.signal.welch) for a much
# lower-variance spectral estimate than the single windowed FFT snapshot in
# view_iq_snapshot.py -- the right tool for characterizing a continuous RF
# signal's occupied bandwidth and noise floor (as opposed to a quick visual
# look at one short window). Reports a numeric summary (noise floor, peak,
# occupied bandwidth) alongside the plot, same spirit as radar_pulses'
# width/PRI/PRF report.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from view_iq_snapshot import load_sigmf_meta, load_iq_segment


def occupied_bandwidth(freqs, psd_db, down_db, smooth_hz=0.0):
    """Bandwidth of the contiguous region around the spectrum's peak that
    stays within `down_db` of it -- walking outward from the peak (rather
    than a global threshold crossing) so an unrelated sidelobe elsewhere in
    the spectrum, below the threshold, can't be mistaken for part of the
    main occupied region. Returns (bandwidth_hz, low_edge_hz, high_edge_hz).

    The peak is located on a version of the PSD smoothed over `smooth_hz`
    (0 disables smoothing), not the raw per-bin PSD: a real modulated
    channel is wide, but a single narrow-band spur/tone (LO leakage, a
    calibration signal, ...) can sit tens of dB above a wider, genuinely
    occupied channel and would otherwise hijack the peak-finding -- exactly
    what happens on real captures with a strong CW spur off to one side.
    """
    search_db = psd_db
    if smooth_hz > 0 and len(freqs) > 1:
        bin_hz = abs(freqs[1] - freqs[0])
        smooth_bins = max(1, int(round(smooth_hz / bin_hz)))
        if smooth_bins > 1:
            # Edge-replicate before convolving, not numpy's default
            # zero-padded 'same' mode: this is a dB-scale curve, so zero
            # padding would pull the smoothed value toward 0 dB -- i.e. an
            # enormous linear power -- right at the array boundaries,
            # fabricating a false "peak" near the band edges.
            pad = smooth_bins // 2
            padded = np.pad(psd_db, pad, mode='edge')
            kernel = np.ones(smooth_bins) / smooth_bins
            search_db = np.convolve(padded, kernel, mode='valid')[:len(psd_db)]

    peak_idx = int(np.argmax(search_db))
    thresh = search_db[peak_idx] - down_db
    above = search_db >= thresh

    lo = peak_idx
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak_idx
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    return freqs[hi] - freqs[lo], freqs[lo], freqs[hi]


def main():
    ap = argparse.ArgumentParser(
        description='Welch PSD estimate (occupied bandwidth, noise floor) for a segment of a SigMF IQ file.')
    ap.add_argument('input', help='SigMF IQ file (.sigmf-data or raw ci16_le/cf32_le)')
    ap.add_argument('--offset', type=int, default=0, help='starting sample (complex samples, default 0)')
    ap.add_argument('--duration', type=float, default=None,
                     help='segment length in seconds (default 20ms, overridden by --nsamp)')
    ap.add_argument('--nsamp', type=int, default=None, help='segment length in samples instead of --duration')
    ap.add_argument('--fs', type=float, default=None, help='sample rate in Hz (default: from .sigmf-meta, else 20e6)')
    ap.add_argument('--fc', type=float, default=None, help='center frequency in Hz (default: from .sigmf-meta, else 0)')
    ap.add_argument('--datatype', default=None, help='ci16_le | cf32_le (default: from .sigmf-meta, else ci16_le)')
    ap.add_argument('--nperseg', type=int, default=4096,
                     help='Welch segment length in samples (default 4096) -- sets frequency resolution '
                          '(fs/nperseg) and, via segment count, how much the PSD estimate is averaged')
    ap.add_argument('--overlap', type=float, default=0.5, help='Welch segment overlap fraction, 0-1 (default 0.5)')
    ap.add_argument('--window', default='blackmanharris',
                     help='Welch window function (default blackmanharris, any scipy.signal.get_window name)')
    ap.add_argument('--obw-down-db', type=float, default=10.0,
                     help='dB below peak used for the occupied-bandwidth estimate (default 10)')
    ap.add_argument('--obw-smooth-khz', type=float, default=0.0,
                     help='smoothing applied before locating the peak for the occupied-bandwidth estimate '
                          '(default 0, disabled). Raise this (e.g. 100-500) if the peak keeps landing on a '
                          'narrow spur/CW tone (LO leakage, a cal signal, ...) instead of a wider, genuinely '
                          'occupied channel elsewhere in the band -- but leave it at 0 when the signal of '
                          'interest really is narrowband, where smoothing would dilute its own peak away')
    ap.add_argument('-o', '--out', default=None, help='output PNG path (default: <stem>_welch.png)')
    args = ap.parse_args()

    in_path = Path(args.input)
    meta = load_sigmf_meta(in_path)
    fs = args.fs or meta.get('fs') or 20e6
    fc = args.fc if args.fc is not None else (meta.get('fc') or 0.0)
    datatype = args.datatype or meta.get('datatype') or 'ci16_le'
    out_path = Path(args.out) if args.out else Path(f'{in_path.stem}_welch.png')
    json_path = out_path.with_suffix('.json')

    if args.nsamp:
        nsamp = args.nsamp
    else:
        duration = args.duration if args.duration is not None else 20e-3
        nsamp = int(round(duration * fs))

    iq = load_iq_segment(in_path, args.offset, nsamp, datatype)
    nsamp = iq.size

    nperseg = min(args.nperseg, nsamp)
    noverlap = int(nperseg * args.overlap)
    freqs, psd = welch(iq, fs=fs, window=args.window, nperseg=nperseg,
                        noverlap=noverlap, return_onesided=False, scaling='density')
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-20)
    freqs_mhz = (freqs + fc) / 1e6

    noise_floor_db = float(np.percentile(psd_db, 20))
    raw_peak_db = float(psd_db.max())
    raw_peak_freq_hz = float(freqs[np.argmax(psd_db)] + fc)
    obw_hz, lo_hz, hi_hz = occupied_bandwidth(freqs, psd_db, args.obw_down_db, args.obw_smooth_khz * 1e3)
    obw_center_hz = (lo_hz + hi_hz) / 2 + fc
    n_segments = 1 + (nsamp - nperseg) // (nperseg - noverlap) if nsamp > nperseg else 1

    print(f"segment: {nsamp} samples ({nsamp/fs*1e3:.3f} ms) at fs={fs/1e6:.3f} Msps, fc={fc/1e6:.3f} MHz")
    print(f"Welch: nperseg={nperseg} ({nperseg/fs*1e6:.1f} us/segment, {fs/nperseg/1e3:.2f} kHz resolution), "
          f"~{n_segments} averaged segments")
    print(f"noise floor (20th pct): {noise_floor_db:.2f} dB/Hz")
    print(f"raw peak (unsmoothed, single bin): {raw_peak_db:.2f} dB/Hz at {raw_peak_freq_hz/1e6:.3f} MHz")
    print(f"occupied bandwidth (-{args.obw_down_db:.0f} dB, {args.obw_smooth_khz:.0f} kHz-smoothed peak): "
          f"{obw_hz/1e6:.3f} MHz centered at {obw_center_hz/1e6:.3f} MHz "
          f"({(lo_hz+fc)/1e6:.3f} to {(hi_hz+fc)/1e6:.3f} MHz)")
    if args.obw_smooth_khz > 0 and abs(raw_peak_freq_hz - obw_center_hz) > obw_hz:
        print(f"  note: the raw peak is well outside the occupied-bandwidth region -- likely a narrow "
              f"spur/CW tone (e.g. LO leakage) separate from the main occupied channel")

    with open(json_path, 'w') as jf:
        json.dump({
            'input': str(in_path), 'offset': args.offset, 'nsamp': nsamp,
            'fs': fs, 'fc': fc, 'nperseg': nperseg, 'noverlap': noverlap, 'window': args.window,
            'noise_floor_db': noise_floor_db,
            'raw_peak_db': raw_peak_db, 'raw_peak_freq_hz': raw_peak_freq_hz,
            'obw_down_db': args.obw_down_db, 'obw_smooth_khz': args.obw_smooth_khz,
            'occupied_bandwidth_hz': float(obw_hz), 'occupied_center_hz': float(obw_center_hz),
            'occupied_low_hz': float(lo_hz + fc), 'occupied_high_hz': float(hi_hz + fc),
        }, jf, indent=2)
    print(f'wrote {json_path}')

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_mhz, psd_db, lw=0.9, color='C0')
    plt.axhline(noise_floor_db, color='0.5', ls='--', lw=0.8,
                label=f'noise floor ({noise_floor_db:.1f} dB/Hz)')
    plt.axvspan((lo_hz + fc) / 1e6, (hi_hz + fc) / 1e6, color='C3', alpha=0.15,
                label=f'occupied BW: {obw_hz/1e6:.2f} MHz (-{args.obw_down_db:.0f} dB, smoothed peak)')
    if abs(raw_peak_freq_hz - obw_center_hz) > obw_hz:
        plt.axvline(raw_peak_freq_hz / 1e6, color='C1', ls=':', lw=1.0,
                    label=f'raw peak (likely a spur): {raw_peak_freq_hz/1e6:.3f} MHz')
    plt.xlabel('Frequency (MHz)' if fc else 'Frequency offset (MHz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.title(f'Welch PSD -- {in_path.name}\n(nperseg={nperseg}, overlap={args.overlap:.0%}, window={args.window})')
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f'saved {out_path}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
