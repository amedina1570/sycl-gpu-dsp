#!/usr/bin/env python3
# Quick-look diagnostic for a short segment of a raw SigMF IQ file: FFT
# spectrum, time-domain I/Q traces, and an I/Q (constellation) scatter.
# Reads directly from a .sigmf-data file (not iq2spectrogram's .bin output),
# so it works standalone on any capture.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_key(obj, key):
    """Recursively search a parsed SigMF JSON doc for the first `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def load_sigmf_meta(data_path):
    meta_path = data_path.with_suffix('.sigmf-meta') if data_path.suffix == '.sigmf-data' \
        else Path(str(data_path) + '.sigmf-meta')
    if not meta_path.exists():
        return {}
    with open(meta_path) as fh:
        doc = json.load(fh)
    return {
        'fs': find_key(doc, 'core:sample_rate'),
        'fc': find_key(doc, 'core:frequency'),
        'datatype': find_key(doc, 'core:datatype'),
    }


def load_iq_segment(path, offset, nsamp, datatype):
    if datatype == 'ci16_le':
        dtype, bytes_per_sample, scale = np.int16, 4, 32768.0
    elif datatype == 'cf32_le':
        dtype, bytes_per_sample, scale = np.float32, 8, 1.0
    else:
        raise ValueError(f"unsupported datatype '{datatype}' (supported: ci16_le, cf32_le)")

    with open(path, 'rb') as f:
        f.seek(offset * bytes_per_sample)
        raw = np.frombuffer(f.read(nsamp * bytes_per_sample), dtype=dtype)
    if raw.size < nsamp * 2:
        got = raw.size // 2
        raise ValueError(f'file too short: requested {nsamp} samples at offset {offset}, only {got} available')

    iq = raw.reshape(-1, 2).astype(np.float32)
    return (iq[:, 0] + 1j * iq[:, 1]) / scale


def time_axis(nsamp, fs):
    t = np.arange(nsamp) / fs
    duration = nsamp / fs
    if duration < 1e-3:
        return t * 1e6, 'Time (µs)'
    if duration < 1.0:
        return t * 1e3, 'Time (ms)'
    return t, 'Time (s)'


def main():
    ap = argparse.ArgumentParser(description='FFT + time-domain + I/Q quick-look plot for a SigMF IQ file.')
    ap.add_argument('input', help='SigMF IQ file (.sigmf-data or raw ci16_le/cf32_le)')
    ap.add_argument('--offset', type=int, default=0, help='starting sample (complex samples, default 0)')
    ap.add_argument('--nsamp', type=int, default=4096, help='number of complex samples to plot (default 4096)')
    ap.add_argument('--fs', type=float, default=None, help='sample rate in Hz (default: from .sigmf-meta, else 20e6)')
    ap.add_argument('--fc', type=float, default=None, help='center frequency in Hz (default: from .sigmf-meta, else 0)')
    ap.add_argument('--datatype', default=None, help='ci16_le | cf32_le (default: from .sigmf-meta, else ci16_le)')
    ap.add_argument('-o', '--out', default=None, help='output PNG path (default: <stem>_snapshot.png)')
    args = ap.parse_args()

    in_path = Path(args.input)
    meta = load_sigmf_meta(in_path)
    fs = args.fs or meta.get('fs') or 20e6
    fc = args.fc if args.fc is not None else (meta.get('fc') or 0.0)
    datatype = args.datatype or meta.get('datatype') or 'ci16_le'
    out_path = args.out or f'{in_path.stem}_snapshot.png'

    iq = load_iq_segment(in_path, args.offset, args.nsamp, datatype)
    nsamp = iq.size

    # --- FFT: Hann-windowed, fftshifted, dB magnitude ---
    window = np.hanning(nsamp)
    spec = np.fft.fftshift(np.fft.fft(iq * window))
    mag_db = 20 * np.log10(np.abs(spec) + 1e-6)
    freqs = (np.fft.fftshift(np.fft.fftfreq(nsamp, d=1 / fs)) + fc) / 1e6  # MHz

    # --- Time domain: I and Q vs time ---
    t, t_label = time_axis(nsamp, fs)

    fig, (ax_fft, ax_time, ax_iq) = plt.subplots(1, 3, figsize=(16, 5))

    ax_fft.plot(freqs, mag_db, lw=0.9, color='C0')
    ax_fft.set_xlabel('Frequency (MHz)' if fc else 'Frequency offset (MHz)')
    ax_fft.set_ylabel('Magnitude (dB)')
    ax_fft.set_title('FFT spectrum')

    ax_time.plot(t, iq.real, lw=0.8, color='C0', label='I')
    ax_time.plot(t, iq.imag, lw=0.8, color='C1', label='Q')
    ax_time.set_xlabel(t_label)
    ax_time.set_ylabel('Amplitude')
    ax_time.set_title('Time domain')
    ax_time.legend()

    ax_iq.scatter(iq.real, iq.imag, s=4, alpha=0.3, color='C0', edgecolors='none')
    ax_iq.set_xlabel('I')
    ax_iq.set_ylabel('Q')
    ax_iq.set_title('I/Q')
    ax_iq.set_aspect('equal', adjustable='box')
    ax_iq.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax_iq.axvline(0, color='0.8', lw=0.6, zorder=0)

    fig.suptitle(f'{in_path.name}  (offset={args.offset}, nsamp={nsamp}, fs={fs:.0f}, fc={fc:.0f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f'saved {out_path}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
