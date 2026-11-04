#!/usr/bin/env python3
"""Ranging characterization of the VL53L9 raw distance on a flat target (white and black paper).

Input: the recordings of l9_visualizer.py --record in Records/paper_tests and Records/black_tests, named
<material>_<distance>.csv where the distance is in meters with "_" as decimal point and a leading 0 for values
below 1 m (paper_05 = 0.5 m, paper_1_5 = 1.5 m).

For every recording the target zone (ROI) is chosen on the median frame:
  - black paper: the expected paper rectangle (size / distance, 1 degree per zone) with the lowest signal with
    respect to a ring on the panel plane around it, within SEARCH_RADIUS zones of the optical centre. Zones that do
    not return the panel distance count as zero signal (beyond 6 m the black paper often returns a spurious peak at
    ~0.1 m). The ROI is the paper core (eroded by one zone against the mixed edge zones) nearest to its centre;
  - white paper: not separable from the wooden panel in amplitude, the ROI is taken on the panel plane nearest to
    the optical centre;
  - below 1 m the paper fills the central part of the FoV and the ROI is centred on the optical centre;
  - with no target visible (10 m) the ROI of the previous distance of the same material is reused.
The ROI has at most ROI_MAX_PIXELS zones, so that the raw radial distance stays within a few degrees of the optical
axis. Every ROI is drawn on Figures/roi_<material>.png for a visual check.

Statistics use the first N_FRAMES frames of each recording (the shortest one), so that every distance has the
same number of samples. A sample detects the target when it is within DETECTION_TOLERANCE of the true distance:
panel (a) shows every sample, the error (b) and the temporal noise (d) only the detected ones, (c) the detection
rate. Output: Figures/range_<material>.pdf/.png, Figures/range_stats.csv.

Usage: python3 l9_range_characterization.py [--frames N] [--all-frames]
"""
import argparse
import collections
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
RECORDS_DIR = TOOLS_DIR / "Records"
CACHE_DIR = TOOLS_DIR / "build" / "range_cache"
FIGURES_DIR = TOOLS_DIR / "Figures"
MATERIALS = {"paper": ("paper_tests", "White paper"), "black": ("black_tests", "Black paper")}
SENSOR_MAX_RANGE_M = 8.0      # declared range of the configuration under test, beyond it the points are out of spec
REFERENCE_UNCERTAINTY_MM = 2  # ground truth measured from the sensor cover glass, optical zero plane unknown
N_FRAMES = 69                 # frames of the shortest recording (black_1)
ROI_MAX_PIXELS = 16
NEAR_RANGE_M = 1.0            # below this distance the paper fills the central FoV, no segmentation
DEG_PER_ZONE = 1.0            # 54 x 42 zones over a 54 x 42 degrees FoV (VL53L9CA data brief DB4528)
BLACK_MIN_CONTRAST = 1.5      # panel / paper amplitude ratio needed to accept a black paper segmentation
DETECTION_TOLERANCE = 0.1     # a sample detects the target if within 10% (at least DETECTION_MIN_MM) of the true distance
DETECTION_MIN_MM = 30
PLANE_TOLERANCE = 0.03        # relative distance tolerance of the panel plane
PAPER_ZONES = (50.0, 25.0)    # (height, width) of the black paper in zones at 1 m, measured on the 1.5-6 m recordings
RING_ON_PLANE = 0.8           # fraction of the ring around the paper that must lie on the panel plane
SEARCH_RADIUS = 5             # zones around the optical centre where the paper centre is searched


def parse_distance(name):
    """paper_05 -> 0.5, paper_1_5 -> 1.5, paper_10 -> 10."""
    value = re.fullmatch(r"[a-z]+_(\d+(?:_\d+)?)", name).group(1)
    if "_" in value:
        return float(value.replace("_", "."))
    if value.startswith("0"):
        return float("0." + value[1:])
    return float(value)


def load_recording(path):
    """Return (distance, flag, amplitude) as (frames, h, w) arrays, cached as npz since the csv files are large."""
    cache = CACHE_DIR / f"{path.stem}.npz"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        data = np.load(cache)
        return data["distance"], data["flag"], data["amplitude"]
    print(f"reading {path.name}...")
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6), dtype=np.int32)
    h, w = values[:, 0].max() + 1, values[:, 1].max() + 1
    n = len(values) // (h * w)
    values = values[:n * h * w]
    distance, flag, amplitude = (values[:, i].reshape(n, h, w) for i in (2, 3, 4))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, distance=distance.astype(np.uint16), flag=flag.astype(np.uint8),
                        amplitude=amplitude.astype(np.uint16))
    return distance, flag, amplitude


def connected(mask, seed):
    """4-connected component of mask containing seed."""
    out = np.zeros_like(mask, dtype=bool)
    if not mask[seed]:
        return out
    queue = collections.deque([seed])
    out[seed] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((ny, nx))
    return out


def erode(mask):
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def box_mean(image, size=3):
    padded = np.pad(image, size // 2, mode="edge")
    h, w = image.shape
    return np.mean([padded[dy:dy + h, dx:dx + w] for dy in range(size) for dx in range(size)], axis=0)


def nearest_pixels(mask, centre, count=ROI_MAX_PIXELS):
    """The count pixels of mask nearest to centre, as a boolean mask."""
    ys, xs = np.nonzero(mask)
    order = np.argsort((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2)[:count]
    roi = np.zeros_like(mask, dtype=bool)
    roi[ys[order], xs[order]] = True
    return roi


def box_sum(image, top, left, bottom, right):
    """Sum and pixel count of image[top:bottom, left:right] clipped to the image."""
    h, w = image.shape
    top, left, bottom, right = max(top, 0), max(left, 0), min(bottom, h), min(right, w)
    if bottom <= top or right <= left:
        return 0.0, 0
    return image[top:bottom, left:right].sum(), (bottom - top) * (right - left)


def find_black_paper(amp, plane, true_m, centre, allowed):
    """Search the expected paper rectangle (PAPER_ZONES / distance) that is darkest with respect to a ring around it.

    Return (seed, paper level, panel level, contrast, search box) or None."""
    h, w = amp.shape
    half_h = max(1.5, PAPER_ZONES[0] / true_m / 2)
    half_w = max(1.0, PAPER_ZONES[1] / true_m / 2)
    inner_h, inner_w = max(1, int(half_h * 0.6)), max(1, int(half_w * 0.6))
    ring = 2
    best = None
    for r in range(int(centre[0]) - SEARCH_RADIUS, int(centre[0]) + SEARCH_RADIUS + 1):
        for c in range(int(centre[1]) - SEARCH_RADIUS, int(centre[1]) + SEARCH_RADIUS + 1):
            if not (0 <= r < h and 0 <= c < w and allowed[r, c]):
                continue
            s_in, n_in = box_sum(amp, r - inner_h, c - inner_w, r + inner_h + 1, c + inner_w + 1)
            top, left = int(r - half_h - ring), int(c - half_w - ring)
            bottom, right = int(r + half_h + ring) + 1, int(c + half_w + ring) + 1
            s_all, n_all = box_sum(amp, top, left, bottom, right)
            s_mid, n_mid = box_sum(amp, int(r - half_h), int(c - half_w), int(r + half_h) + 1, int(c + half_w) + 1)
            if n_in == 0 or n_all - n_mid <= 0:
                continue
            # the paper is attached on the panel: its surrounding ring must lie on the panel plane
            p_all, _ = box_sum(plane.astype(float), top, left, bottom, right)
            p_mid, _ = box_sum(plane.astype(float), int(r - half_h), int(c - half_w), int(r + half_h) + 1,
                               int(c + half_w) + 1)
            if (p_all - p_mid) / (n_all - n_mid) < RING_ON_PLANE:
                continue
            inner, outer = s_in / n_in, (s_all - s_mid) / (n_all - n_mid)
            contrast = outer / max(inner, 1.0)
            if best is None or contrast > best[0]:
                best = (contrast, (r, c), inner, outer)
    if best is None or best[0] < BLACK_MIN_CONTRAST:
        return None
    # the paper cannot extend beyond its expected size (+1 zone of margin), so the threshold does not leak into
    # darker parts of the panel at long range
    box = np.zeros((h, w), dtype=bool)
    r, c = best[1]
    box[max(0, int(r - 2 * half_h - 1)):int(r + 2 * half_h + 2), max(0, int(c - 2 * half_w - 1)):int(c + 2 * half_w + 2)] = True
    return best[1], best[2], best[3], best[0], box


def find_roi(material, true_m, distance, amplitude, centre, previous_roi):
    """Return (roi mask, target mask, method) on the median frame."""
    amp = np.median(amplitude, axis=0).astype(float)
    dist = np.median(distance, axis=0).astype(float)
    h, w = amp.shape
    seed_c = (int(round(centre[0])), int(round(centre[1])))

    if true_m < NEAR_RANGE_M:
        target = np.ones((h, w), dtype=bool)
        return nearest_pixels(target, centre), target, "fills FoV"

    # panel plane: pixels whose axial distance (radial distance projected with 1 degree per zone) matches the one
    # of the image centre, the paper is attached on it
    rows, cols = np.indices((h, w))
    axial = dist / np.sqrt(1 + np.tan(np.radians((rows - centre[0]) * DEG_PER_ZONE)) ** 2
                           + np.tan(np.radians((cols - centre[1]) * DEG_PER_ZONE)) ** 2)
    d_centre = np.median(axial[seed_c[0] - 4:seed_c[0] + 5, seed_c[1] - 4:seed_c[1] + 5])
    on_plane = np.abs(axial - d_centre) < PLANE_TOLERANCE * d_centre
    ys, xs = np.nonzero(on_plane)
    nearest = np.argmin((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2) if len(ys) else None
    plane = connected(on_plane, (ys[nearest], xs[nearest])) if nearest is not None else on_plane
    plane_ok = plane.sum() >= 9 and abs(d_centre / 1000 - true_m) < 0.1 * true_m
    # zones that do not return the panel distance count as zero signal: beyond 6 m the black paper returns a
    # spurious peak at ~0.1 m with a high amplitude
    no_target = dist < 0.5 * d_centre
    signal = np.where(no_target, 0.0, amp)

    if material == "black" and plane_ok:
        found = find_black_paper(signal, plane, true_m, centre, plane | no_target)
        if found is not None:
            seed, paper_level, panel_level, contrast, box = found
            threshold = np.sqrt(paper_level * panel_level)
            target = connected((signal < threshold) & box, seed)
            core = erode(target)
            if core.sum() >= 4:
                ys, xs = np.nonzero(core)
                roi = nearest_pixels(core, (ys.mean(), xs.mean()))
            else:
                # paper a few zones wide: keep its darkest half, the edge zones mix paper and panel
                count = max(2, target.sum() // 2)
                order = np.argsort(np.where(target, signal, np.inf), axis=None)[:count]
                roi = np.zeros((h, w), dtype=bool)
                roi.flat[order] = True
            return roi, target, f"amplitude segmentation (contrast {contrast:.1f})"

    if material == "paper" and plane_ok:
        core = erode(plane)
        base = core if core.sum() >= 4 else plane
        return nearest_pixels(base, centre), plane, "panel plane at centre"

    if previous_roi is not None:
        return previous_roi, np.zeros((h, w), dtype=bool), "ROI of previous distance (no target)"
    target = np.ones((h, w), dtype=bool)
    return nearest_pixels(target, centre), target, "optical centre (no target)"


def roi_stats(true_m, distance, flag, amplitude, roi, centre):
    """Statistics of the ROI samples.

    all_*: every sample, as reported by the sensor. The error statistics use only the samples that detected the
    target (within DETECTION_TOLERANCE of the true distance), the others are counted in detection_rate."""
    samples = distance[:, roi].astype(float)  # (frames, pixels)
    true_mm = true_m * 1000
    detected = np.abs(samples - true_mm) < max(DETECTION_TOLERANCE * true_mm, DETECTION_MIN_MM)
    values = np.where(detected, samples, np.nan)
    ys, xs = np.nonzero(roi)
    angles = np.radians(np.hypot(ys - centre[0], xs - centre[1]) * DEG_PER_ZONE)
    all_p5, all_p50, all_p95 = np.percentile(samples, [5, 50, 95])
    p5, p25, p50, p75, p95 = np.nanpercentile(values, [5, 25, 50, 75, 95]) if detected.any() else [np.nan] * 5
    pixels_ok = detected.sum(axis=0) >= 2
    return dict(
        true_mm=true_mm, n_frames=samples.shape[0], n_pixels=samples.shape[1],
        roi_row=ys.mean(), roi_col=xs.mean(), detection_rate=detected.mean(),
        all_median_mm=all_p50, all_p5_mm=all_p5, all_p95_mm=all_p95,
        median_mm=p50, mean_mm=np.nanmean(values), p5_mm=p5, p25_mm=p25, p75_mm=p75, p95_mm=p95,
        error_median_mm=p50 - true_mm, error_mean_mm=np.nanmean(values) - true_mm,
        temporal_std_mm=np.nanstd(values[:, pixels_ok], axis=0).mean() if pixels_ok.any() else np.nan,
        spatial_std_mm=np.nanmedian(values[:, pixels_ok], axis=0).std() if pixels_ok.any() else np.nan,
        total_std_mm=np.nanstd(values),
        flag_fraction=flag[:, roi].mean(), amplitude_median=np.median(amplitude[:, roi]),
        max_angle_deg=np.degrees(angles.max()), radial_bias_mm=true_mm * np.mean(1 / np.cos(angles) - 1),
    )


ROI_STYLE = {  # same plot rules as the candle figures, font scaled to the wider figure
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 26,
    "svg.fonttype": "none",
}
TARGET_COLOR, ROI_COLOR, CENTRE_COLOR = "#3987e5", "#e5484d", "#f5a524"


def plot_rois(material, items, centre, name="roi"):
    """Draw the target and the ROI of every recording on its image (item["image"], median map shown in log)."""
    with plt.rc_context(ROI_STYLE):
        fig, axes = plt.subplots(3, 5, figsize=(14, 8.6), gridspec_kw=dict(wspace=0.06, hspace=0.25))
        for ax, item in zip(axes.flat, items):
            image = np.log10(np.clip(np.nan_to_num(item["image"]), 0, None) + 1)
            lo, hi = np.percentile(image, [2, 98])
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            if item["target"].any() and not item["target"].all():
                ax.contour(item["target"], levels=[0.5], colors=TARGET_COLOR, linewidths=1.8)
            ax.contour(item["roi"], levels=[0.5], colors=ROI_COLOR, linewidths=2.4)
            ax.plot(centre[1], centre[0], "+", color=CENTRE_COLOR, ms=14, mew=2.4)
            ax.set_title(f"{item['true_m']:g} m", pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        for ax in axes.flat[len(items):]:
            ax.axis("off")
        handles = [matplotlib.lines.Line2D([], [], color=TARGET_COLOR, lw=3, label="Target"),
                   matplotlib.lines.Line2D([], [], color=ROI_COLOR, lw=3, label="ROI"),
                   matplotlib.lines.Line2D([], [], ls="none", marker="+", ms=18, mew=3, color=CENTRE_COLOR,
                                           label="Optical centre")]
        top = axes[0, 0].get_position().y1
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top + 0.06), ncol=3, frameon=False,
                   handlelength=1.6, columnspacing=1.8, borderaxespad=0)
        for ext in ("png", "svg"):
            fig.savefig(FIGURES_DIR / f"{name}_{material}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def shared_limits(rows):
    """y limits of the error (b) and temporal noise (d) panels common to every material, so that the figures can be
    compared side by side."""
    get = lambda key: np.array([r[key] for r in rows], dtype=float)
    low = min(np.nanmin(get("p5_mm") - get("true_mm")), -10)
    high = np.nanmax(get("p95_mm") - get("true_mm"))
    return dict(error=(low - 5, high + 0.35 * (high - low)), sigma=(0, np.nanmax(get("temporal_std_mm")) * 1.15))


def plot_range(material, label, stats, limits):
    """(a) measured vs true on every sample, (b) error of the detected samples with percentile bands,
    (c) detection rate, (d) temporal noise of the detected samples. limits: shared y limits of (b) and (d)."""
    blue_dark, blue_mid, blue_light = "#1f5aa6", "#8fb5e3", "#d4e3f5"
    ink, muted, grid = "#1d1d1f", "#6e6e73", "#e5e5ea"
    true = np.array([s["true_mm"] for s in stats]) / 1000
    order = np.argsort(true)
    stats = [stats[i] for i in order]
    true = true[order]
    get = lambda key: np.array([s[key] for s in stats], dtype=float)
    out = true > SENSOR_MAX_RANGE_M
    hollow = dict(ls="none", marker="o", ms=6, mfc="white", mec=blue_dark, mew=1.5, zorder=4)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.family": "serif"})
    fig, axes = plt.subplots(4, 1, figsize=(5.6, 9.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.2, 1.2, 0.6, 0.7], hspace=0.1))
    for ax in axes:
        ax.axvspan(np.sqrt(SENSOR_MAX_RANGE_M * 9), 13, color="#f2f2f4", zorder=0, lw=0)
        ax.grid(True, which="major", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) measured vs true, every sample as reported by the sensor
    ax = axes[0]
    ax.plot([0.08, 13], [0.08, 13], color=muted, lw=1, ls="--", label="ideal (y = x)")
    ax.fill_between(true, get("all_p5_mm") / 1000, get("all_p95_mm") / 1000, color=blue_light, lw=0,
                    label="5-95th pct")
    ax.plot(true, get("all_median_mm") / 1000, color=blue_dark, lw=1.5, marker="o", ms=4, label="median")
    ax.plot(true[out], get("all_median_mm")[out] / 1000, **hollow)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.06, 14)
    ax.set_ylabel("Measured distance [m]")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.text(9.8, 0.3, "beyond\nspec", color=muted, fontsize=8, ha="center", va="bottom")

    # (b) error of the samples that detected the target
    ax = axes[1]
    ax.axhspan(-REFERENCE_UNCERTAINTY_MM, REFERENCE_UNCERTAINTY_MM, color="#c7c7cc", alpha=0.7, lw=0,
               label=f"reference uncertainty (±{REFERENCE_UNCERTAINTY_MM} mm)")
    ax.axhline(0, color=muted, lw=0.8)
    ok = ~np.isnan(get("median_mm"))
    ax.fill_between(true[ok], (get("p5_mm") - get("true_mm"))[ok], (get("p95_mm") - get("true_mm"))[ok],
                    color=blue_light, lw=0, label="5-95th pct")
    ax.fill_between(true[ok], (get("p25_mm") - get("true_mm"))[ok], (get("p75_mm") - get("true_mm"))[ok],
                    color=blue_mid, lw=0, label="IQR (25-75th pct)")
    ax.plot(true[ok], get("error_median_mm")[ok], color=blue_dark, lw=1.5, marker="o", ms=4, label="median")
    ax.plot(true[ok & out], get("error_median_mm")[ok & out], **hollow)
    ax.set_ylabel("Error of detected\nsamples [mm]")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    ax.set_ylim(*limits["error"])

    # (c) detection rate
    ax = axes[2]
    ax.plot(true, 100 * get("detection_rate"), color=blue_dark, lw=1.5, marker="o", ms=4)
    ax.plot(true[out], 100 * get("detection_rate")[out], **hollow)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel(f"Detected [%]\n(±{DETECTION_TOLERANCE:.0%} of true)")

    # (d) temporal noise of the detected samples
    ax = axes[3]
    ax.plot(true[ok], get("temporal_std_mm")[ok], color=blue_dark, lw=1.5, marker="o", ms=4)
    ax.plot(true[ok & out], get("temporal_std_mm")[ok & out], **hollow)
    ax.set_ylim(*limits["sigma"])
    ax.set_ylabel("Temporal σ [mm]")
    ax.set_xlabel("True distance [m]")
    ax.set_xlim(0.085, 12.5)
    ticks = [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    for ax, letter in zip(axes, "abcd"):
        ax.text(0.985, 0.97, f"({letter})", transform=ax.transAxes, fontsize=9, color=ink, va="top", ha="right")
    axes[0].set_title(f"{label} - raw distance, {stats[0]['n_frames']} frames per point", fontsize=10, color=ink,
                      loc="left")
    fig.align_ylabels(axes)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES_DIR / f"range_{material}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="frames used for each recording")
    parser.add_argument("--all-frames", action="store_true", help="use every frame of each recording")
    args = parser.parse_args()
    FIGURES_DIR.mkdir(exist_ok=True)

    recordings = {material: sorted(((parse_distance(p.stem), p) for p in (RECORDS_DIR / folder).glob("*.csv")))
                  for material, (folder, _) in MATERIALS.items()}
    # optical axis on the centre of the zone array (the fit of the radial distance on a flat target is biased by the
    # tilt of the panel leaning on the wall)
    h, w = load_recording(dict(recordings["paper"])[1.0])[0].shape[1:]
    centre = ((h - 1) / 2, (w - 1) / 2)
    print(f"optical centre: row {centre[0]:.2f}, col {centre[1]:.2f}")

    rows, stats_by_material = [], {}
    for material, (_, label) in MATERIALS.items():
        items, stats, previous_roi = [], [], None
        for true_m, path in recordings[material]:
            distance, flag, amplitude = load_recording(path)
            if not args.all_frames:
                distance, flag, amplitude = distance[:args.frames], flag[:args.frames], amplitude[:args.frames]
            roi, target, method = find_roi(material, true_m, distance, amplitude, centre, previous_roi)
            previous_roi = roi
            s = roi_stats(true_m, distance, flag, amplitude, roi, centre)
            s.update(material=material, file=path.name, roi_method=method)
            stats.append(s)
            items.append(dict(true_m=true_m, image=np.median(amplitude, axis=0), roi=roi, target=target, method=method))
            print(f"{path.name:16s} {method:38s} px={s['n_pixels']:2d} median={s['median_mm']:8.1f} "
                  f"err={s['error_median_mm']:+8.1f} IQR=[{s['p25_mm'] - s['true_mm']:+.0f},"
                  f"{s['p75_mm'] - s['true_mm']:+.0f}] sigma_t={s['temporal_std_mm']:7.1f} "
                  f"det={s['detection_rate']:.2f} all_med={s['all_median_mm']:.0f} flag={s['flag_fraction']:.2f} amp={s['amplitude_median']:.0f} bias={s['radial_bias_mm']:.1f}")
        plot_rois(material, items, centre)
        stats_by_material[material] = stats
        rows += stats
    limits = shared_limits(rows)
    for material, (_, label) in MATERIALS.items():
        plot_range(material, label, stats_by_material[material], limits)

    columns = ["material", "file", "true_mm", "roi_method", "roi_row", "roi_col", "n_frames", "n_pixels",
               "detection_rate", "all_median_mm", "all_p5_mm", "all_p95_mm",
               "median_mm", "mean_mm", "p5_mm", "p25_mm", "p75_mm", "p95_mm", "error_median_mm", "error_mean_mm",
               "temporal_std_mm", "spatial_std_mm", "total_std_mm", "flag_fraction", "amplitude_median",
               "max_angle_deg", "radial_bias_mm"]
    with open(FIGURES_DIR / "range_stats.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"figures and statistics written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
