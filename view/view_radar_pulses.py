#!/usr/bin/env python3
# Pulse-train analysis for RADAR-like (pulsed) signals. Two modes:
#
#   view_radar_pulses.py capture.sigmf-data --duration 5e-3
#       All-NumPy: reads a segment straight from a raw SigMF IQ file,
#       computes the envelope, detects pulses, and plots -- no GPU/build
#       needed, exactly as this script has always worked.
#
#   view_radar_pulses.py capture_radar.json
#       Precomputed: reads the JSON + envelope .bin sidecar written by
#       build/radar_pulses (the GPU-accelerated port of the same detection
#       logic below), and just plots -- lets a much longer segment be
#       processed on GPU instead of in NumPy. Only a small I/Q slice is
#       re-read from the original file, for the FFT/I-Q panels.
#
# Reuses view_iq_snapshot's loader and FFT computation for the spectral/I-Q
# panels in both modes.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from view_iq_snapshot import load_sigmf_meta, load_iq_segment, time_axis, compute_fft_db


def format_duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return 'n/a'
    if seconds < 1e-6:
        return f'{seconds*1e9:.1f} ns'
    if seconds < 1e-3:
        return f'{seconds*1e6:.2f} µs'
    if seconds < 1.0:
        return f'{seconds*1e3:.2f} ms'
    return f'{seconds:.4f} s'


def detect_pulses(envelope, threshold_frac, min_pulse_samples):
    """Threshold the envelope at floor + threshold_frac*(peak-floor) and pair
    rising/falling edges into complete pulses. Returns (rising_idx, falling_idx, threshold)."""
    floor = np.percentile(envelope, 10)
    peak = np.percentile(envelope, 99.9)
    threshold = floor + threshold_frac * (peak - floor)

    above = envelope > threshold
    edges = np.diff(above.astype(np.int8))
    rising = np.where(edges == 1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    # Drop incomplete pulses at the segment boundaries (cut off, can't
    # reliably measure width/PRI for them).
    if len(falling) and (len(rising) == 0 or falling[0] < rising[0]):
        falling = falling[1:]
    if len(rising) and (len(falling) == 0 or rising[-1] > falling[-1]):
        rising = rising[:-1]
    n = min(len(rising), len(falling))
    rising, falling = rising[:n], falling[:n]

    long_enough = (falling - rising) >= min_pulse_samples
    return rising[long_enough], falling[long_enough], threshold


def pulse_stats(rising, falling, fs):
    widths = (falling - rising) / fs
    stats = {
        'n_pulses': int(len(rising)),
        'pulse_width_mean_s': float(np.mean(widths)) if len(widths) else None,
        'pulse_width_std_s': float(np.std(widths)) if len(widths) else None,
        'pri_mean_s': None,
        'pri_std_s': None,
        'prf_hz': None,
        'duty_cycle': None,
    }
    if len(rising) >= 2:
        pri = np.diff(rising) / fs
        stats['pri_mean_s'] = float(np.mean(pri))
        stats['pri_std_s'] = float(np.std(pri))
        stats['prf_hz'] = float(1.0 / np.mean(pri))
        if stats['pulse_width_mean_s'] is not None:
            stats['duty_cycle'] = float(stats['pulse_width_mean_s'] / stats['pri_mean_s'])
    return stats


def run_precomputed(json_path, out_arg):
    """Precomputed mode: plot from a build/radar_pulses .json + envelope .bin
    sidecar instead of recomputing detection. Only a small I/Q slice is
    re-read from the original file, for the FFT/I-Q panels."""
    with open(json_path) as fh:
        meta = json.load(fh)

    iq_path = Path(meta['iq_input'])
    offset, nsamp = meta['offset'], meta['nsamp']
    fs, fc, datatype = meta['fs'], meta['fc'], meta['datatype']
    threshold = meta['threshold']
    rising = np.array(meta['rising'], dtype=np.int64)
    falling = np.array(meta['falling'], dtype=np.int64)
    stats = {k: meta[k] for k in (
        'n_pulses', 'pulse_width_mean_s', 'pulse_width_std_s',
        'pri_mean_s', 'pri_std_s', 'prf_hz', 'duty_cycle')}

    envelope = np.fromfile(meta['envelope_bin'], dtype=np.float32)
    iq = load_iq_segment(iq_path, offset, nsamp, datatype)

    out_path = Path(out_arg) if out_arg else json_path.with_suffix('.png')
    title_name = f'{iq_path.name} (via {json_path.name})'
    return iq, envelope, rising, falling, threshold, stats, fs, fc, offset, nsamp, out_path, title_name


def run_from_raw(in_path, args):
    """All-NumPy mode: read a segment straight from a raw SigMF IQ file,
    detect pulses, and write the same .json a build/radar_pulses run would
    (minus the envelope_bin/iq_input/rising/falling fields a downstream
    --json rerun would need -- this path recomputes everything itself)."""
    meta = load_sigmf_meta(in_path)
    fs = args.fs or meta.get('fs') or 20e6
    fc = args.fc if args.fc is not None else (meta.get('fc') or 0.0)
    datatype = args.datatype or meta.get('datatype') or 'ci16_le'
    out_path = Path(args.out) if args.out else Path(f'{in_path.stem}_radar.png')
    json_path = out_path.with_suffix('.json')

    nsamp = args.nsamp or int(round(args.duration * fs))
    iq = load_iq_segment(in_path, args.offset, nsamp, datatype)
    nsamp = iq.size

    # --- Envelope: magnitude, lightly smoothed to stabilize thresholding ---
    envelope = np.abs(iq)
    if args.smooth_samples > 1:
        kernel = np.ones(args.smooth_samples) / args.smooth_samples
        envelope = np.convolve(envelope, kernel, mode='same')

    rising, falling, threshold = detect_pulses(envelope, args.threshold_frac, args.min_pulse_samples)
    stats = pulse_stats(rising, falling, fs)

    with open(json_path, 'w') as jf:
        json.dump({
            'input': str(in_path), 'offset': args.offset, 'nsamp': nsamp,
            'fs': fs, 'fc': fc, 'threshold_frac': args.threshold_frac,
            **stats,
        }, jf, indent=2)
    print(f'wrote {json_path}')

    return iq, envelope, rising, falling, threshold, stats, fs, fc, args.offset, nsamp, out_path, in_path.name


def main():
    ap = argparse.ArgumentParser(description='Pulse-train analysis (width/PRI/PRF) for a RADAR-like SigMF IQ capture.')
    ap.add_argument('input', help='SigMF IQ file (.sigmf-data or raw ci16_le/cf32_le), '
                                   'or a .json written by build/radar_pulses to plot precomputed results')
    ap.add_argument('--offset', type=int, default=0, help='starting sample (complex samples, default 0)')
    ap.add_argument('--duration', type=float, default=5e-3,
                     help='segment length in seconds, must span several pulses (default 5ms)')
    ap.add_argument('--nsamp', type=int, default=None,
                     help='segment length in samples instead of --duration')
    ap.add_argument('--fs', type=float, default=None, help='sample rate in Hz (default: from .sigmf-meta, else 20e6)')
    ap.add_argument('--fc', type=float, default=None, help='center frequency in Hz (default: from .sigmf-meta, else 0)')
    ap.add_argument('--datatype', default=None, help='ci16_le | cf32_le (default: from .sigmf-meta, else ci16_le)')
    ap.add_argument('--threshold-frac', type=float, default=0.5,
                     help='detection threshold as a fraction between the noise floor and peak envelope (default 0.5 = half-amplitude)')
    ap.add_argument('--smooth-samples', type=int, default=5,
                     help='envelope smoothing window in samples, reduces false edges from noise (default 5)')
    ap.add_argument('--min-pulse-samples', type=int, default=3,
                     help='discard detections shorter than this many samples (default 3)')
    ap.add_argument('-o', '--out', default=None, help='output PNG path (default: <stem>_radar.png)')
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.suffix == '.json':
        (iq, envelope, rising, falling, threshold, stats, fs, fc,
         offset, nsamp, out_path, title_name) = run_precomputed(in_path, args.out)
    else:
        (iq, envelope, rising, falling, threshold, stats, fs, fc,
         offset, nsamp, out_path, title_name) = run_from_raw(in_path, args)

    print(f"pulses detected: {stats['n_pulses']}")
    if stats['n_pulses']:
        print(f"  pulse width: {format_duration(stats['pulse_width_mean_s'])} "
              f"(+/- {format_duration(stats['pulse_width_std_s'])})")
    if stats['pri_mean_s'] is not None:
        print(f"  PRI: {format_duration(stats['pri_mean_s'])} (+/- {format_duration(stats['pri_std_s'])})")
        print(f"  PRF: {stats['prf_hz']:.1f} Hz")
        print(f"  duty cycle: {stats['duty_cycle']*100:.2f}%")
    else:
        print('  fewer than 2 pulses in this segment -- cannot compute PRI; try a longer --duration')

    # --- Plot: FFT | envelope+pulses | I/Q | extracted-parameters text ---
    freqs, mag_db = compute_fft_db(iq, fs, fc)
    t, t_label = time_axis(nsamp, fs)

    fig, (ax_fft, ax_env, ax_iq, ax_text) = plt.subplots(1, 4, figsize=(20, 5))

    ax_fft.plot(freqs, mag_db, lw=0.9, color='C0')
    ax_fft.set_xlabel('Frequency (MHz)' if fc else 'Frequency offset (MHz)')
    ax_fft.set_ylabel('Magnitude (dB)')
    ax_fft.set_title('FFT spectrum')

    ax_env.plot(t, envelope, lw=0.7, color='C0')
    ax_env.axhline(threshold, color='0.5', lw=0.8, ls='--', label='threshold')
    for r, f in zip(rising, falling):
        ax_env.axvspan(t[r], t[f - 1], color='C3', alpha=0.25, lw=0)
    ax_env.set_xlabel(t_label)
    ax_env.set_ylabel('Amplitude (envelope)')
    ax_env.set_title(f'Pulse train ({stats["n_pulses"]} detected)')
    ax_env.legend(loc='upper right', fontsize=8)

    ax_iq.scatter(iq.real, iq.imag, s=3, alpha=0.2, color='C0', edgecolors='none')
    ax_iq.set_xlabel('I'); ax_iq.set_ylabel('Q'); ax_iq.set_title('I/Q')
    ax_iq.set_aspect('equal', adjustable='box')
    ax_iq.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax_iq.axvline(0, color='0.8', lw=0.6, zorder=0)

    ax_text.axis('off')
    lines = [f"Pulses: {stats['n_pulses']}"]
    if stats['n_pulses']:
        lines.append(f"Width: {format_duration(stats['pulse_width_mean_s'])}")
        lines.append(f"  (+/- {format_duration(stats['pulse_width_std_s'])})")
    if stats['pri_mean_s'] is not None:
        lines.append(f"PRI: {format_duration(stats['pri_mean_s'])}")
        lines.append(f"  (+/- {format_duration(stats['pri_std_s'])})")
        lines.append(f"PRF: {stats['prf_hz']:.1f} Hz")
        lines.append(f"Duty cycle: {stats['duty_cycle']*100:.2f}%")
    else:
        lines.append('(need >=2 pulses')
        lines.append(' for PRI/PRF)')
    ax_text.text(0.02, 0.95, '\n'.join(lines), transform=ax_text.transAxes,
                 va='top', ha='left', fontsize=11, family='monospace')
    ax_text.set_title('Extracted parameters')

    fig.suptitle(f'{title_name}  (offset={offset}, nsamp={nsamp}, fs={fs:.0f}, fc={fc:.0f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f'saved {out_path}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
