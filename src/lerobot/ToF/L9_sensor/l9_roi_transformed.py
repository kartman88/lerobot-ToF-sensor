#!/usr/bin/env python3
"""Draw the ROI of l9_range_characterization.py on the transformed outputs of the recordings, to check whether the
paper (the white one in particular, not separable from the wooden panel in raw amplitude) is more visible there.

The ROI is still the one found on the raw data; only the background image changes:
  signal_rate: amplitude processed by the ST pipeline (vl53l9-transform-c), corrected for DSS and exposure
  reflectance: signal rate compensated for the distance, estimate of the target reflectance
Median over the first N_FRAMES frames, like the statistics.

Output: Figures/roi_signal_rate_<material>.png/.svg, Figures/roi_reflectance_<material>.png/.svg
Usage: python3 l9_roi_transformed.py
"""
import numpy as np

import l9_range_characterization as rc

TRANSFORMED = {"signal_rate": 10, "reflectance": 13}  # CSV column of each map


def load_transformed(path, h, w):
    """Median signal rate and reflectance maps of the first N_FRAMES frames, cached as npz."""
    cache = rc.CACHE_DIR / f"{path.stem}_transformed_{rc.N_FRAMES}.npz"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        data = np.load(cache)
        return {key: data[key] for key in TRANSFORMED}
    print(f"reading {path.name}...")
    columns = list(TRANSFORMED.values())
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, max_rows=rc.N_FRAMES * h * w)
    n = len(values) // (h * w)
    maps = {key: np.nanmedian(values[:n * h * w, i].reshape(n, h, w), axis=0) for i, key in enumerate(TRANSFORMED)}
    rc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **maps)
    return maps


def main():
    rc.FIGURES_DIR.mkdir(exist_ok=True)
    for material, (folder, _) in rc.MATERIALS.items():
        recordings = sorted((rc.parse_distance(p.stem), p) for p in (rc.RECORDS_DIR / folder).glob("*.csv"))
        h, w = rc.load_recording(recordings[0][1])[0].shape[1:]
        centre = ((h - 1) / 2, (w - 1) / 2)
        items = {key: [] for key in TRANSFORMED}
        previous_roi = None
        for true_m, path in recordings:
            distance, _, amplitude = rc.load_recording(path)
            distance, amplitude = distance[:rc.N_FRAMES], amplitude[:rc.N_FRAMES]
            roi, target, method = rc.find_roi(material, true_m, distance, amplitude, centre, previous_roi)
            previous_roi = roi
            maps = load_transformed(path, h, w)
            for key in TRANSFORMED:
                items[key].append(dict(true_m=true_m, image=maps[key], roi=roi, target=target, method=method))
        for key in TRANSFORMED:
            rc.plot_rois(material, items[key], centre, name=f"roi_{key}")
    print(f"figures written to {rc.FIGURES_DIR}")


if __name__ == "__main__":
    main()
