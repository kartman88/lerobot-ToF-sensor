#!/usr/bin/env python3
"""Save panels of l9_visualizer.py exactly as the viewer draws them, for one frame of a recording.

The frame is read with the replay reader of the viewer and drawn by its Viewer class (same figure, colour maps,
colour ranges and layout), then every requested panel (image, title and colorbar) is cropped from the figure and
saved at the screen resolution of the viewer. The frame is the temporally central one unless --frame is given.

Output: Records/<recording>/<recording>_viewer_<panel>.png/.pdf/.svg
Usage: python3 l9_viewer_frame.py [recording.csv] [--frame N] [--flip-x] [--flip-y]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.transforms import Bbox

import l9_visualizer as viz

PANELS = {"Raw distance": "depth_raw", "Raw amplitude": "amplitude_raw"}  # viewer panel title: file suffix
MARGIN_IN = 0.05


def read_frame(path, index):
    """(frame, transformed) of frame index, transformed is None if the pipeline skipped it while recording."""
    for i, (_, frame, transformed) in enumerate(viz.read_recording(path)):
        if i == index:
            return frame, transformed
    raise ValueError(f"{path} has only {i + 1} frames")


def count_frames(path):
    with open(path) as csv:
        next(csv)
        return len({line.split(",", 2)[1] for line in csv})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", type=Path, default=viz.RECORDS_DIR / "picture.csv")
    parser.add_argument("--frame", type=int, help="frame index (default: central frame)")
    parser.add_argument("--flip-x", action="store_true", help="mirror the images horizontally, as in the viewer")
    parser.add_argument("--flip-y", action="store_true", help="mirror the images vertically, as in the viewer")
    args = parser.parse_args()
    n_frames = count_frames(args.recording)
    index = n_frames // 2 if args.frame is None else args.frame
    frame, transformed = read_frame(args.recording, index)

    options = argparse.Namespace(range=None, flip_x=args.flip_x, flip_y=args.flip_y, show_invalid=False,
                                 no_transform=False)
    viewer = viz.Viewer(source=None, args=options)
    viewer.draw_frame(frame, transformed)
    # the viewer draws images and colorbars with blitting: disconnect its draw callback (screen backgrounds, fails on a
    # cropped savefig) and make the artists plain, a savefig skips animated ones
    fig = viewer.fig
    for cid in list(fig.canvas.callbacks.callbacks.get("draw_event", {})):
        fig.canvas.mpl_disconnect(cid)
    for image, colorbar, *_ in viewer.images:
        image.set_animated(False)
        colorbar.ax.set_animated(False)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # keep the viewer layout when the other panels are hidden below
    renderer = fig.canvas.get_renderer()

    out_dir = args.recording.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    panels = []
    for image, colorbar, *_ in viewer.images:
        title = image.axes.get_title()
        if title not in PANELS:
            continue
        box = Bbox.union([image.axes.get_tightbbox(renderer), colorbar.ax.get_tightbbox(renderer)])
        panels.append((title, {image.axes, colorbar.ax}, box.transformed(fig.dpi_scale_trans.inverted()).padded(MARGIN_IN)))
    others = [*fig.texts, *fig.legends, *fig.images, *fig.lines, *fig.patches]
    for title, keep, box in panels:
        # bbox_inches only moves the viewBox of a vector file: the other panels stay in it, outside the page, and show
        # up when a tool exports the whole drawing (Inkscape in Overleaf), so hide everything outside the panel
        for ax in fig.axes:
            ax.set_visible(ax in keep)
        for artist in others:
            artist.set_visible(False)
        path = out_dir / f"{args.recording.stem}_viewer_{PANELS[title]}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(path.with_name(f"{path.name}.{ext}"), dpi=fig.dpi, bbox_inches=box)
    print(f"frame {index} of {n_frames} (counter {frame['frame_counter']}) written to {out_dir}")


if __name__ == "__main__":
    main()
