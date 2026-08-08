#!/usr/bin/env python3
# Interactive, no-server spectrogram explorer for iq2spectrogram output.
#
# This uses matplotlib's normal GUI window instead of an HTTP server. The
# backing .bin is memory-mapped, and every pan/zoom redraw samples only the
# visible viewport into a bounded image.
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt


CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'gray']
FILTERS = ['none', 'freq_median', 'time_median', 'zscore_freq']
FREQ_REDUCERS = ['sample', 'mean', 'peak']
DEFAULT_OVERVIEW_TIME = 1024
DEFAULT_OVERVIEW_FREQ = 8
DEFAULT_FREQ_REDUCE = 'sample'


class SpectrogramStore:
    def __init__(self, meta_path, overview_time=DEFAULT_OVERVIEW_TIME,
                 overview_freq=DEFAULT_OVERVIEW_FREQ, build_overview=False,
                 overview_reduce=DEFAULT_FREQ_REDUCE):
        self.meta_path = Path(meta_path)
        with open(self.meta_path) as fh:
            self.meta = json.load(fh)
        self.bin_path = Path(self.meta['bin'])
        self.nfft = int(self.meta['nfft'])
        self.hop = int(self.meta['hop'])
        self.fs = float(self.meta['fs'])
        self.fc = float(self.meta['fc'])

        raw = np.memmap(self.bin_path, dtype=np.float32, mode='r')
        self.nframes = int(raw.size // self.nfft)
        if self.nframes <= 0:
            raise ValueError(f'{self.bin_path} does not contain a complete spectrogram frame')
        self.spec = raw[:self.nframes * self.nfft].reshape(self.nframes, self.nfft)

        self.duration_s = (self.nframes - 1) * self.hop / self.fs
        self.freq_min_mhz = (self.fc - self.fs / 2) / 1e6
        self.freq_max_mhz = (self.fc + (self.nfft / 2 - 1) * self.fs / self.nfft) / 1e6
        self.overview_time = overview_time
        self.overview_freq = overview_freq
        self.overview_reduce = overview_reduce
        self.overview_path = self.meta_path.with_name(
            f'{self.meta_path.stem}_overview_t{overview_time}_f{overview_freq}_{overview_reduce}.npz')
        self.overview = None
        self.overview_shape = None
        if build_overview or self.overview_path.exists():
            self.load_or_build_overview()

    def load_or_build_overview(self):
        if not self.overview_path.exists():
            self.build_overview()
        cached = np.load(self.overview_path)
        cached_reduce = 'peak'
        if 'freq_reduce' in cached.files:
            cached_reduce = str(cached['freq_reduce'].item())
        if (int(cached['nfft']) != self.nfft or
                int(cached['nframes']) != self.nframes or
                int(cached['time_factor']) != self.overview_time or
                int(cached['freq_factor']) != self.overview_freq or
                cached_reduce != self.overview_reduce):
            print(f'overview cache metadata changed; rebuilding {self.overview_path}')
            self.build_overview()
            cached = np.load(self.overview_path)
        self.overview = cached['overview']
        self.overview_shape = self.overview.shape
        print(f'overview cache: {self.overview_path} {self.overview_shape[0]} x {self.overview_shape[1]}')

    def build_overview(self):
        rows = int(math.ceil(self.nframes / self.overview_time))
        cols = int(math.ceil(self.nfft / self.overview_freq))
        out = np.empty((rows, cols), dtype=np.float32)
        started = time.time()
        print(f'building overview cache {self.overview_path} ({rows} x {cols})')
        for r in range(rows):
            r0 = r * self.overview_time
            r1 = min(self.nframes, r0 + self.overview_time)
            block = self.spec[r0:r1, :]
            padded_cols = cols * self.overview_freq
            if padded_cols != self.nfft:
                block = np.pad(block, ((0, 0), (0, padded_cols - self.nfft)), mode='edge')
            block = block.reshape(block.shape[0], cols, self.overview_freq)
            time_peak = block.max(axis=0)
            if self.overview_reduce == 'mean':
                out[r] = time_peak.mean(axis=1)
            elif self.overview_reduce == 'sample':
                out[r] = time_peak[:, self.overview_freq // 2]
            else:
                out[r] = time_peak.max(axis=1)
            if rows >= 20 and (r + 1) % max(1, rows // 20) == 0:
                pct = 100.0 * (r + 1) / rows
                elapsed = time.time() - started
                print(f'  {pct:5.1f}%  {elapsed:6.1f}s', flush=True)
        np.savez_compressed(
            self.overview_path,
            overview=out,
            nfft=np.array(self.nfft),
            nframes=np.array(self.nframes),
            time_factor=np.array(self.overview_time),
            freq_factor=np.array(self.overview_freq),
            freq_reduce=np.array(self.overview_reduce),
        )
        print(f'wrote {self.overview_path} in {time.time() - started:.1f}s')

    def frame_for_time(self, seconds):
        frame = int(math.floor(seconds * self.fs / self.hop))
        return max(0, min(frame, self.nframes - 1))

    def bin_for_freq_mhz(self, mhz):
        hz = mhz * 1e6
        k = int(math.floor((hz - (self.fc - self.fs / 2)) * self.nfft / self.fs))
        return max(0, min(k, self.nfft - 1))

    def viewport(self, t0, t1, f0_mhz, f1_mhz, max_time_px, max_freq_px,
                 max_source_cells, freq_reduce='sample'):
        if t1 <= t0:
            t1 = t0 + self.hop / self.fs
        if f1_mhz <= f0_mhz:
            f1_mhz = f0_mhz + self.fs / self.nfft / 1e6

        fr0 = self.frame_for_time(t0)
        fr1 = self.frame_for_time(t1) + 1
        k0 = self.bin_for_freq_mhz(f0_mhz)
        k1 = self.bin_for_freq_mhz(f1_mhz) + 1
        fr1 = max(fr0 + 1, min(fr1, self.nframes))
        k1 = max(k0 + 1, min(k1, self.nfft))

        rows = fr1 - fr0
        cols = k1 - k0
        time_factor = max(1, int(math.ceil(rows / max_time_px)))
        freq_factor = max(1, int(math.ceil(cols / max_freq_px)))
        out_rows = int(math.ceil(rows / time_factor))
        out_cols = int(math.ceil(cols / freq_factor))
        source_cells = rows * cols
        mode = 'max-hold'

        if source_cells > max_source_cells:
            if (self.overview is not None and freq_reduce == self.overview_reduce and
                    time_factor >= self.overview_time and freq_factor >= self.overview_freq):
                mode = 'overview'
                out = self.overview_view(fr0, fr1, k0, k1, out_rows, out_cols)
                freq_reduce_used = freq_reduce
            else:
                # Fast fallback: sample the visible rectangle instead of
                # scanning it. Building a matching overview cache replaces
                # this with time max-hold plus the selected frequency reducer.
                mode = 'sampled'
                row_idx = np.linspace(fr0, fr1 - 1, out_rows).astype(np.int64)
                col_idx = np.linspace(k0, k1 - 1, out_cols).astype(np.int64)
                out = np.asarray(self.spec[np.ix_(row_idx, col_idx)], dtype=np.float32)
                freq_reduce_used = 'sample'
        else:
            out = np.empty((out_rows, out_cols), dtype=np.float32)
            freq_reduce_used = freq_reduce
            for r in range(out_rows):
                r0 = fr0 + r * time_factor
                r1 = min(fr1, r0 + time_factor)
                block = self.spec[r0:r1, k0:k1]
                if freq_factor > 1:
                    padded_cols = out_cols * freq_factor
                    if padded_cols != cols:
                        block = np.pad(block, ((0, 0), (0, padded_cols - cols)), mode='edge')
                    block = block.reshape(block.shape[0], out_cols, freq_factor)
                    time_peak = block.max(axis=0)
                    if freq_reduce == 'mean':
                        out[r] = time_peak.mean(axis=1)
                    elif freq_reduce == 'sample':
                        out[r] = time_peak[:, freq_factor // 2]
                    else:
                        out[r] = time_peak.max(axis=1)
                else:
                    out[r] = block.max(axis=0)

        extent = [
            fr0 * self.hop / self.fs,
            (fr1 - 1) * self.hop / self.fs,
            (self.fc - self.fs / 2 + k0 * self.fs / self.nfft) / 1e6,
            (self.fc - self.fs / 2 + (k1 - 1) * self.fs / self.nfft) / 1e6,
        ]
        return out, extent, {
            'frame0': fr0, 'frame1': fr1 - 1,
            'bin0': k0, 'bin1': k1 - 1,
            'time_factor': time_factor, 'freq_factor': freq_factor,
            'mode': mode,
            'freq_reduce': freq_reduce_used,
            'source_cells': source_cells,
        }

    def overview_view(self, fr0, fr1, k0, k1, out_rows, out_cols):
        or0 = fr0 // self.overview_time
        or1 = max(or0 + 1, int(math.ceil(fr1 / self.overview_time)))
        ok0 = k0 // self.overview_freq
        ok1 = max(ok0 + 1, int(math.ceil(k1 / self.overview_freq)))
        coarse = self.overview[or0:or1, ok0:ok1]
        rows, cols = coarse.shape
        row_factor = max(1, int(math.ceil(rows / out_rows)))
        col_factor = max(1, int(math.ceil(cols / out_cols)))
        rr = int(math.ceil(rows / row_factor))
        cc = int(math.ceil(cols / col_factor))
        out = np.empty((rr, cc), dtype=np.float32)
        for r in range(rr):
            r0 = r * row_factor
            r1 = min(rows, r0 + row_factor)
            block = coarse[r0:r1, :]
            if col_factor > 1:
                padded_cols = cc * col_factor
                if padded_cols != cols:
                    block = np.pad(block, ((0, 0), (0, padded_cols - cols)), mode='edge')
                block = block.reshape(block.shape[0], cc, col_factor)
                time_peak = block.max(axis=0)
                if self.overview_reduce == 'mean':
                    out[r] = time_peak.mean(axis=1)
                elif self.overview_reduce == 'sample':
                    out[r] = time_peak[:, col_factor // 2]
                else:
                    out[r] = time_peak.max(axis=1)
            else:
                out[r] = block.max(axis=0)
        return out


def apply_filter(data, mode):
    out = data.astype(np.float32, copy=True)
    if mode == 'freq_median':
        out -= np.median(out, axis=0, keepdims=True).astype(np.float32)
    elif mode == 'time_median':
        out -= np.median(out, axis=1, keepdims=True).astype(np.float32)
    elif mode == 'zscore_freq':
        med = np.median(out, axis=0, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(out - med), axis=0, keepdims=True).astype(np.float32)
        out = (out - med) / (1.4826 * mad + 1e-6)
    return out


def fmt_time(seconds):
    if abs(seconds) < 1e-3:
        return f'{seconds * 1e6:.3f} us'
    if abs(seconds) < 1.0:
        return f'{seconds * 1e3:.3f} ms'
    return f'{seconds:.6f} s'


def fmt_freq(mhz):
    return f'{mhz:.6f} MHz'


class Viewer:
    def __init__(self, store, max_time_px, max_freq_px, max_source_cells):
        self.store = store
        self.max_time_px = max_time_px
        self.max_freq_px = max_freq_px
        self.max_source_cells = max_source_cells
        self.cmap = 'viridis'
        self.filter = 'none'
        self.freq_reduce = DEFAULT_FREQ_REDUCE
        self.vmin_pct = 20.0
        self.vmax_pct = 99.5
        self.measure_points = []
        self._updating = False

        self.fig = plt.figure(figsize=(13, 7))
        self.ax = self.fig.add_axes([0.08, 0.13, 0.72, 0.80])
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (MHz)' if store.fc else 'Frequency offset (MHz)')
        self.ax.set_title(f'{store.bin_path.name} ({store.nframes} frames x {store.nfft} bins)')

        self.status = self.fig.text(0.08, 0.035, '', family='monospace', fontsize=9)
        self.measure = self.fig.text(0.49, 0.035, 'Click two points to measure', family='monospace', fontsize=9)
        self.help = self.fig.text(
            0.08, 0.005,
            'toolbar/mouse wheel: zoom/pan   c: colormap   f: filter   g: reducer   [ ]: low pct   { }: high pct   r: reset',
            family='monospace', fontsize=8, color='0.45')
        self.panel = self.fig.add_axes([0.83, 0.13, 0.14, 0.80])
        self.panel.set_axis_off()
        self.panel_items = []

        data, extent, info = self.render_data(0.0, store.duration_s, store.freq_min_mhz, store.freq_max_mhz)
        self.image = self.ax.imshow(data.T, extent=extent, aspect='auto', origin='lower',
                                    cmap=self.cmap, interpolation='nearest')
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax, pad=0.01)
        self.set_color_limits(data)
        self.update_status(info)
        self.draw_panel()
        self.connect_events()

    def connect_events(self):
        self.ax.callbacks.connect('xlim_changed', lambda _ax: self.redraw_from_axes())
        self.ax.callbacks.connect('ylim_changed', lambda _ax: self.redraw_from_axes())
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def render_data(self, t0, t1, f0, f1):
        data, extent, info = self.store.viewport(
            t0, t1, f0, f1, self.max_time_px, self.max_freq_px,
            self.max_source_cells, self.freq_reduce)
        return apply_filter(data, self.filter), extent, info

    def set_color_limits(self, data):
        vmin = float(np.percentile(data, self.vmin_pct))
        vmax = float(np.percentile(data, self.vmax_pct))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
        self.image.set_clim(vmin, vmax)

    def update_status(self, info):
        self.status.set_text(
            f'frames {info["frame0"]}..{info["frame1"]}  bins {info["bin0"]}..{info["bin1"]}  '
            f'decim time x{info["time_factor"]}, freq x{info["freq_factor"]}  '
            f'{info["mode"]}  reduce={info["freq_reduce"]}  filter={self.filter}  '
            f'cmap={self.cmap}  pct={self.vmin_pct:g}..{self.vmax_pct:g}'
        )

    def draw_panel(self):
        self.panel.clear()
        self.panel.set_axis_off()
        self.panel_items = []
        y = 0.98

        def heading(text):
            nonlocal y
            self.panel.text(0.0, y, text, transform=self.panel.transAxes,
                            fontsize=10, fontweight='bold', va='top')
            y -= 0.055

        def item(kind, value, label, active):
            nonlocal y
            marker = '>' if active else ' '
            color = 'C0' if active else '0.75'
            artist = self.panel.text(0.02, y, f'{marker} {label}', transform=self.panel.transAxes,
                                     fontsize=9, family='monospace', color=color, va='top',
                                     picker=True)
            self.panel_items.append((artist, kind, value))
            y -= 0.04

        heading('Reducer')
        for name in FREQ_REDUCERS:
            item('reduce', name, name, name == self.freq_reduce)

        y -= 0.04
        heading('Filter')
        for name in FILTERS:
            item('filter', name, name, name == self.filter)

        y -= 0.04
        heading('Colormap')
        for name in CMAPS:
            item('cmap', name, name, name == self.cmap)

        y -= 0.04
        heading('Display')
        self.panel.text(0.02, y, f'low  [{self.vmin_pct:g}%]  [ / ]',
                        transform=self.panel.transAxes, fontsize=9, family='monospace',
                        color='0.75', va='top')
        y -= 0.04
        self.panel.text(0.02, y, f'high [{self.vmax_pct:g}%]  {{ / }}',
                        transform=self.panel.transAxes, fontsize=9, family='monospace',
                        color='0.75', va='top')
        y -= 0.07
        self.panel.text(0.02, y, 'r reset view', transform=self.panel.transAxes,
                        fontsize=9, family='monospace', color='0.75', va='top')

    def redraw_from_axes(self):
        if self._updating:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.redraw(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def redraw(self, t0, t1, f0, f1):
        self._updating = True
        try:
            data, extent, info = self.render_data(t0, t1, f0, f1)
            self.image.set_data(data.T)
            self.image.set_extent(extent)
            self.ax.set_xlim(t0, t1)
            self.ax.set_ylim(f0, f1)
            self.set_color_limits(data)
            self.update_status(info)
            self.draw_panel()
            self.fig.canvas.draw_idle()
        finally:
            self._updating = False

    def set_cmap(self, name):
        self.cmap = name
        self.image.set_cmap(name)
        self.redraw_from_axes()
        self.fig.canvas.draw_idle()

    def set_filter(self, name):
        self.filter = name
        self.redraw_from_axes()

    def set_freq_reduce(self, name):
        self.freq_reduce = name
        self.redraw_from_axes()

    def set_vmin(self, text):
        try:
            self.vmin_pct = max(0.0, min(99.0, float(text)))
        except ValueError:
            return
        self.redraw_from_axes()

    def set_vmax(self, text):
        try:
            self.vmax_pct = max(self.vmin_pct + 0.1, min(100.0, float(text)))
        except ValueError:
            return
        self.redraw_from_axes()

    def reset(self, _event=None):
        self.measure_points.clear()
        self.measure.set_text('Click two points to measure')
        self.redraw(0.0, self.store.duration_s, self.store.freq_min_mhz, self.store.freq_max_mhz)

    def on_click(self, event):
        if getattr(event, 'inaxes', None) == self.panel:
            self.on_panel_click(event)
            return
        if getattr(event, 'inaxes', None) != self.ax or event.xdata is None or event.ydata is None:
            return
        self.measure_points.append((float(event.xdata), float(event.ydata)))
        if len(self.measure_points) > 2:
            self.measure_points = [self.measure_points[-1]]
        self.update_measure()

    def on_motion(self, event):
        if getattr(event, 'inaxes', None) == self.ax and event.xdata is not None and event.ydata is not None:
            self.fig.canvas.manager.set_window_title(
                f'{fmt_time(event.xdata)} | {fmt_freq(event.ydata)}'
            )

    def on_scroll(self, event):
        if getattr(event, 'inaxes', None) != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.75 if event.button == 'up' else 1.35
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (event.xdata - x0) / (x1 - x0)
        cy = (event.ydata - y0) / (y1 - y0)
        dt = (x1 - x0) * scale
        df = (y1 - y0) * scale
        t0 = event.xdata - cx * dt
        t1 = t0 + dt
        f0 = event.ydata - cy * df
        f1 = f0 + df
        self.redraw(t0, t1, f0, f1)

    def on_key(self, event):
        if event.key == 'r':
            self.reset()
        elif event.key == 'f':
            i = (FILTERS.index(self.filter) + 1) % len(FILTERS)
            self.set_filter(FILTERS[i])
        elif event.key == 'g':
            i = (FREQ_REDUCERS.index(self.freq_reduce) + 1) % len(FREQ_REDUCERS)
            self.set_freq_reduce(FREQ_REDUCERS[i])
        elif event.key == 'c':
            i = (CMAPS.index(self.cmap) + 1) % len(CMAPS)
            self.set_cmap(CMAPS[i])
        elif event.key == '[':
            self.vmin_pct = max(0.0, self.vmin_pct - 1.0)
            self.redraw_from_axes()
        elif event.key == ']':
            self.vmin_pct = min(self.vmax_pct - 0.1, self.vmin_pct + 1.0)
            self.redraw_from_axes()
        elif event.key == '{':
            self.vmax_pct = max(self.vmin_pct + 0.1, self.vmax_pct - 0.5)
            self.redraw_from_axes()
        elif event.key == '}':
            self.vmax_pct = min(100.0, self.vmax_pct + 0.5)
            self.redraw_from_axes()

    def on_panel_click(self, event):
        for artist, kind, value in self.panel_items:
            contains, _ = artist.contains(event)
            if not contains:
                continue
            if kind == 'filter':
                self.set_filter(value)
            elif kind == 'cmap':
                self.set_cmap(value)
            elif kind == 'reduce':
                self.set_freq_reduce(value)
            return

    def update_measure(self):
        if len(self.measure_points) < 2:
            self.measure.set_text('Point A set')
            self.fig.canvas.draw_idle()
            return
        (t0, f0), (t1, f1) = self.measure_points
        dt = t1 - t0
        df = f1 - f0
        slope = df / dt if dt != 0 else float('inf')
        self.measure.set_text(
            f'A {fmt_time(t0)}, {fmt_freq(f0)}   '
            f'B {fmt_time(t1)}, {fmt_freq(f1)}   '
            f'dT {fmt_time(dt)}  dF {df:.6f} MHz  slope {slope:.6f} MHz/s'
        )
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description='Interactive no-server viewer for iq2spectrogram output.')
    ap.add_argument('json', help='spectrogram .json sidecar written by iq2spectrogram')
    ap.add_argument('--max-time-px', type=int, default=1400,
                    help='maximum rendered time pixels per viewport (default 1400)')
    ap.add_argument('--max-freq-px', type=int, default=900,
                    help='maximum rendered frequency pixels per viewport (default 900)')
    ap.add_argument('--max-source-mcells', type=float, default=96.0,
                    help='source cells to scan before using sampled overview mode, in millions (default 96)')
    ap.add_argument('--build-overview', action='store_true',
                    help='build/load a coarse max-hold overview cache before opening the viewer')
    ap.add_argument('--overview-time', type=int, default=DEFAULT_OVERVIEW_TIME,
                    help=f'frames per overview row (default {DEFAULT_OVERVIEW_TIME})')
    ap.add_argument('--overview-freq', type=int, default=DEFAULT_OVERVIEW_FREQ,
                    help=f'frequency bins per overview column (default {DEFAULT_OVERVIEW_FREQ})')
    ap.add_argument('--overview-reduce', choices=FREQ_REDUCERS, default=DEFAULT_FREQ_REDUCE,
                    help=f'frequency reducer for overview cache (default {DEFAULT_FREQ_REDUCE})')
    args = ap.parse_args()

    store = SpectrogramStore(args.json, args.overview_time, args.overview_freq,
                             args.build_overview, args.overview_reduce)
    print(f'spectrogram: {store.bin_path} ({store.nframes} frames x {store.nfft} bins)')
    print('Use the matplotlib toolbar or mouse wheel to zoom/pan; click two points to measure.')
    Viewer(store, args.max_time_px, args.max_freq_px, int(args.max_source_mcells * 1e6))
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
