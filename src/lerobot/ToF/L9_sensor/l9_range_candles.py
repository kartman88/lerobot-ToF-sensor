#!/usr/bin/env python3
"""Candle plot of the VL53L9 raw distance on a flat target: one I-shaped candle per recorded distance showing the
spread of every sample of the ROI (all pixels x all frames) around the true distance.

The recordings, the ROI of each recording and the frames used are the same as l9_range_characterization.py (its
functions are reused). Unlike that script, no sample is discarded: spurious returns are part of the candle.

Candle: whiskers with caps from the 5th to the 95th percentile, box from the 25th to the 75th, line at the median,
dots for the samples outside the whiskers (one dot per distinct value). Candles beyond the sensor range are hollow.
  (a) measured vs true distance [m], log-log, with the ideal y = x line
  (b) error (measured - true) [mm], symmetric log scale: linear within ±LINEAR_ERROR_MM, logarithmic beyond

Output: Figures/candles_<material>.pdf/.png
Usage: python3 l9_range_candles.py [--frames N] [--all-frames]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.legend_handler
import matplotlib.pyplot as plt
import numpy as np

import l9_range_characterization as rc

LINEAR_ERROR_MM = 50      # half width of the linear part of the error axis
CANDLE_WIDTH = 0.042      # candle width in decades of the log distance axis (8 m -> 9 m is 0.051 decades)
MAX_OUTLIER_DOTS = 200    # distinct outlier values drawn per candle
X_LIMITS = (0.085, 12.5)  # true distance axis [m]
REFERENCE_CONE_END_MM = 10  # reference uncertainty at the right edge, widening (log-linearly) from rc.REFERENCE_UNCERTAINTY_MM
# same plot rules as the learning curve figures: Times 20 pt, black axes, Dark2 colours, soft grid
COLORS = {"paper": "#1b9e77", "black": "#6a3d9a"}  # teal / purple of the same Dark2 palette
FILL_ALPHA = 0.25
INK, REF = "#000000", "#555555"
SHADE = "#bfbfbf"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

def roi_samples(material, frames):
    """Return [(true distance m, ROI samples mm)] of every recording of material, sorted by distance."""
    folder, _ = rc.MATERIALS[material]
    recordings = sorted((rc.parse_distance(p.stem), p) for p in (rc.RECORDS_DIR / folder).glob("*.csv"))
    h, w = rc.load_recording(recordings[0][1])[0].shape[1:]
    centre = ((h - 1) / 2, (w - 1) / 2)
    result, previous_roi = [], None
    for true_m, path in recordings:
        distance, _, amplitude = rc.load_recording(path)
        if frames:
            distance, amplitude = distance[:frames], amplitude[:frames]
        roi, _, method = rc.find_roi(material, true_m, distance, amplitude, centre, previous_roi)
        previous_roi = roi
        result.append((true_m, distance[:, roi].astype(float).ravel()))
        print(f"{path.name:16s} {method:40s} {roi.sum():2d} px x {distance.shape[0]} frames")
    return result


def draw_candle(ax, x, values, out_of_range, color):
    """I-shaped candle at x (log axis): whiskers 5-95th pct with caps, box 25-75th pct, median line, outlier dots."""
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    half = 10 ** (CANDLE_WIDTH / 2)
    left, right = x / half, x * half
    cap_left, cap_right = x / half ** 0.6, x * half ** 0.6
    face = matplotlib.colors.to_rgba("white" if out_of_range else color, 1 if out_of_range else FILL_ALPHA)
    line = dict(color=color, lw=1.6, zorder=3, solid_capstyle="butt")
    ax.plot([x, x], [p5, p25], **line)
    ax.plot([x, x], [p75, p95], **line)
    ax.plot([cap_left, cap_right], [p5, p5], **line)
    ax.plot([cap_left, cap_right], [p95, p95], **line)
    ax.fill_between([left, right], p25, p75, facecolor=face, edgecolor=color, lw=1.6, zorder=3)
    ax.plot([left, right], [p50, p50], color=INK, lw=2.4, zorder=4, solid_capstyle="butt")
    outliers = np.unique(values[(values < p5) | (values > p95)])
    if len(outliers) > MAX_OUTLIER_DOTS:
        outliers = outliers[np.linspace(0, len(outliers) - 1, MAX_OUTLIER_DOTS).astype(int)]
    ax.plot(np.full(len(outliers), x), outliers, ls="none", marker="o", ms=3.5, mfc="white", mec=color, mew=0.8,
            zorder=2)


class WhiskerHandler(matplotlib.legend_handler.HandlerBase):
    """Legend key of the 5-95th percentile whisker: a horizontal line with caps, i.e. a candle turned on its side."""

    def __init__(self, color):
        super().__init__()
        self.color = color

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y, cap = height / 2 - ydescent, height * 0.5
        x0, x1 = -xdescent, width - xdescent
        style = dict(color=self.color, lw=1.6, transform=trans, solid_capstyle="butt")
        return [matplotlib.lines.Line2D([x0, x1], [y, y], **style),
                matplotlib.lines.Line2D([x0, x0], [y - cap, y + cap], **style),
                matplotlib.lines.Line2D([x1, x1], [y - cap, y + cap], **style)]


def style_axes(ax):
    ax.grid(True, which="major", color="#999999", alpha=0.45, lw=0.6)
    ax.grid(True, which="minor", color=SHADE, alpha=0.3, lw=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="major", length=3.5, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.6)


def plot_candles(material, data):
    plt.rcParams.update(STYLE)
    color = COLORS[material]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 12), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.1))
    beyond_left = np.sqrt(rc.SENSOR_MAX_RANGE_M * 9)  # halfway (log) between the 8 m and 9 m candles
    for ax in axes:
        ax.axvspan(beyond_left, 13, color=SHADE, alpha=0.25, zorder=0, lw=0)
        style_axes(ax)

    # (a) measured vs true
    ax = axes[0]
    ax.plot([0.08, 13], [0.08, 13], color=REF, lw=1.2, ls="--", zorder=1)
    for true_m, samples in data:
        draw_candle(ax, true_m, samples / 1000, true_m > rc.SENSOR_MAX_RANGE_M, color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 14)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Measured distance [m]")

    # (b) error
    ax = axes[1]
    cone_x = np.array(X_LIMITS)
    cone_y = np.array([rc.REFERENCE_UNCERTAINTY_MM, REFERENCE_CONE_END_MM])  # straight edges on the log x axis
    ax.fill_between(cone_x, -cone_y, cone_y, color=REF, alpha=0.3, lw=0, zorder=1)
    ax.axhline(0, color=REF, lw=0.8, zorder=1)
    for limit in (-LINEAR_ERROR_MM, LINEAR_ERROR_MM):  # linear / logarithmic boundary of the symlog axis
        ax.axhline(limit, color=REF, lw=1.0, ls=":", zorder=1)
    for true_m, samples in data:
        draw_candle(ax, true_m, samples - true_m * 1000, true_m > rc.SENSOR_MAX_RANGE_M, color)
    ax.set_yscale("symlog", linthresh=LINEAR_ERROR_MM, linscale=2.5)
    ax.set_yticks([-10000, -1000, -50, -25, 0, 25, 50, 1000, 10000])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-12000, 12000)
    ax.set_ylabel("Error (measured \u2212 true) [mm]")
    ax.set_xlabel("True distance [m]")
    ax.set_xlim(*X_LIMITS)
    ticks = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 8, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(1, 10)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # one legend for both panels, above the figure: at 20 pt it does not fit inside the axes
    whisker = matplotlib.lines.Line2D([], [])  # proxy drawn by WhiskerHandler
    handles = [
        matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba(color, FILL_ALPHA), edgecolor=color, lw=1.6),
        matplotlib.lines.Line2D([], [], color=INK, lw=2.4, solid_capstyle="butt"),
        whisker,
        matplotlib.lines.Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=color, mew=0.8),
        matplotlib.lines.Line2D([], [], color=REF, lw=1.2, ls="--"),
        matplotlib.patches.Patch(facecolor=REF, alpha=0.3, lw=0),
        matplotlib.lines.Line2D([], [], color=REF, lw=1.0, ls=":"),
        matplotlib.patches.Patch(facecolor=SHADE, alpha=0.25, lw=0),
    ]
    labels = [
        "25th\u201375th percentile",
        "Median",
        "5th\u201395th percentile",
        "Samples outside 5th\u201395th",
        "Ideal (measured = true)",
        f"Reference uncertainty (\u00b1{rc.REFERENCE_UNCERTAINTY_MM}\u2192{REFERENCE_CONE_END_MM} mm)",
        f"Linear scale within \u00b1{LINEAR_ERROR_MM} mm",
        f"Beyond sensor range (> {rc.SENSOR_MAX_RANGE_M:g} m)",
    ]
    fig.legend(handles=handles, labels=labels, handler_map={whisker: WhiskerHandler(color)}, loc="lower left",
               bbox_to_anchor=(axes[0].get_position().x0, axes[0].get_position().y1 + 0.005), ncol=2,
               frameon=False, fontsize=16, handlelength=1.6, columnspacing=1.2, borderaxespad=0, borderpad=0)
    for ax, letter in zip(axes, "ab"):
        ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, color=INK, va="top", ha="left")
    fig.align_ylabels(axes)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(rc.FIGURES_DIR / f"candles_{material}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=rc.N_FRAMES, help="frames used for each recording")
    parser.add_argument("--all-frames", action="store_true", help="use every frame of each recording")
    args = parser.parse_args()
    rc.FIGURES_DIR.mkdir(exist_ok=True)
    frames = None if args.all_frames else args.frames
    for material in rc.MATERIALS:
        plot_candles(material, roi_samples(material, frames))
    print(f"figures written to {rc.FIGURES_DIR}")


if __name__ == "__main__":
    main()
