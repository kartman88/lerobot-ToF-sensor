#!/usr/bin/env python3
"""Live viewer of the VL53L9 raw stream (vl53l9_app.c with CONF_OUTPUT_MODE = OUTPUT_MODE_RAW).

Top row: the raw buffer as read from the sensor (distance, amplitude, ambient, DSS index).
Bottom row: the same frame processed on the PC by the ST vl53l9-transform-c pipeline, the one that runs on the
STM32 in OUTPUT_MODE_TRANSFORM (depth in mm, signal rate, ambient rate, confidence). Depth pixels flagged as invalid
by the pipeline are shown in gray.

The pipeline is built from the sources of the package (Middlewares/ST/vl53l9-transform-c) together with
l9_transform_host.c into build/libl9transform.so on the first run (requires gcc). It uses the calibration
received in the L9CB packet, sent once at boot: start the viewer before resetting the board, or pass --calib with a
capture made by l9_raw_receiver.py --out. Without calibration the default one of the library is used.

Packets are decoded with l9_raw_receiver.py. The reception and the pipeline run in a separate process, so that the
matplotlib redraws never delay the serial reads (the ST-LINK VCP has no flow control, a late reader loses bytes).

Usage:
  python3 l9_visualizer.py /dev/ttyACM0 [--baud 12500000]
  python3 l9_visualizer.py --file capture.bin [--fps 30]
  python3 l9_visualizer.py /dev/ttyACM0 --record [NAME]   (Records/NAME.csv, date and time by default)
  python3 l9_visualizer.py --replay Records/NAME.csv [--speed 1.0]

From a serial port the board clock is synced with the host one (see l9_time_sync.py): recordings have an acq_time_s
column, acquisition time of the frame on the host clock (nan until the clock is synced).
Requires: pip install -r requirements.txt (a venv is provided in Tools/.venv)
"""
import argparse
import collections
import ctypes
import datetime
import itertools
import multiprocessing
import os
import queue
import signal
import struct
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.transforms import Bbox

import l9_raw_receiver as rx
import l9_time_sync as ts

TOOLS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TOOLS_DIR.parents[5]  # Tools -> project -> 53L9A1 -> Applications -> NUCLEO-H563ZI -> Projects
TRANSFORM_DIR = PACKAGE_ROOT / "Middlewares/ST/vl53l9-transform-c/vl53l9-transform-c-lib"
MEDIA_DIR = PACKAGE_ROOT / "Middlewares/ST/media-object"
WRAPPER_SOURCE = TOOLS_DIR / "l9_transform_host.c"
LIBRARY_PATH = TOOLS_DIR / "build" / "libl9transform.so"
RECORDS_DIR = TOOLS_DIR / "Records"

# media-c return codes (see media-c/error.h)
MEDIA_ERRORS = {-1: "INVALID_PARAMETER", -2: "INVALID_STATE", -3: "INVALID_CONFIGURATION", -4: "UNALIGNED_MEMORY",
                -5: "STATIC_CTRL", -6: "R_ONLY_CTRL", -7: "W_ONLY_CTRL", -8: "PENDING", -9: "NOT_FOUND",
                -10: "INVALID_PAD_CAPS", -11: "INVALID_PAD_NAME", -12: "INVALID_OPERATION", -13: "UNIMPLEMENTED",
                -14: "UNKNOWN"}
L9T_ERROR_INVALID_METADATA = -100  # see l9_transform_host.c
# float outputs of l9t_process(), in the order of its arguments
FLOAT_OUTPUTS = ["depth", "amplitude", "ambient", "confidence", "reflectance"]
RESULT_QUEUE_SIZE = 2  # frames waiting for the display, older ones are dropped
FILE_CHUNK = 64 * 1024
STREAM_FPS_FRAMES = 30
AUTO_RANGE_PERCENTILES = (2, 98)
AUTO_RANGE_SMOOTHING = 0.2  # weight of the new frame in the displayed color range
TITLE_REFRESH_S = 0.5  # title refresh period when no frame arrives (stream stopped, errors)
COLORBAR_MARGIN_PX = 20  # colorbar backgrounds include the tick labels and the offset text


# --- host transform --------------------------------------------------------------------------------------------------

def build_library():
    """Build the host transform library if it is missing or older than its sources."""
    sources = [WRAPPER_SOURCE, *sorted((TRANSFORM_DIR / "src").rglob("*.c"))]
    includes = [TRANSFORM_DIR / "include/public", TRANSFORM_DIR / "include/private",
                MEDIA_DIR / "media-c/header-lib/include/public", MEDIA_DIR / "transform-c/header-lib/include/public"]
    dependencies = sources + [h for d in includes for h in d.rglob("*.h")]
    if LIBRARY_PATH.exists() and LIBRARY_PATH.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies):
        return
    LIBRARY_PATH.parent.mkdir(exist_ok=True)
    command = [os.environ.get("CC", "gcc"), "-O2", "-shared", "-fPIC", "-std=gnu11",
               *[f"-I{d}" for d in includes], *map(str, sources), "-lm", "-o", str(LIBRARY_PATH)]
    print(f"building {LIBRARY_PATH.relative_to(TOOLS_DIR)} from {TRANSFORM_DIR.relative_to(PACKAGE_ROOT)}")
    subprocess.run(command, check=True)


def load_library():
    lib = ctypes.CDLL(str(LIBRARY_PATH))
    float_p = ctypes.POINTER(ctypes.c_float)
    lib.l9t_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    lib.l9t_create.restype = ctypes.c_int
    lib.l9t_destroy.argtypes = [ctypes.c_void_p]
    lib.l9t_destroy.restype = None
    lib.l9t_get_output_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                        ctypes.POINTER(ctypes.c_uint32)]
    lib.l9t_get_output_size.restype = None
    lib.l9t_process.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                                *[float_p] * len(FLOAT_OUTPUTS), ctypes.POINTER(ctypes.c_uint32)]
    lib.l9t_process.restype = ctypes.c_int
    return lib


def media_error(code):
    return f"{code} ({MEDIA_ERRORS.get(code, '?')})"


class InvalidFrame(Exception):
    """Raw buffer with an inconsistent status line (e.g. corrupted on the uart), skipped by the pipeline."""


class HostTransform:
    """vl53l9-transform-c pipeline running on the PC, frames must be processed in order (temporal filtering)."""

    def __init__(self, calib, raw_size):
        self._lib = load_library()
        self._handle = ctypes.c_void_p()
        ret = self._lib.l9t_create(calib, raw_size, ctypes.byref(self._handle))
        if ret:
            raise RuntimeError(f"pipeline creation failed: {media_error(ret)}")
        width, height = ctypes.c_uint32(), ctypes.c_uint32()
        self._lib.l9t_get_output_size(self._handle, ctypes.byref(width), ctypes.byref(height))
        self.shape = (height.value, width.value)
        self._floats = [np.zeros(self.shape, dtype=np.float32) for _ in FLOAT_OUTPUTS]
        self._status = np.zeros(self.shape, dtype=np.uint32)

    def process(self, payload):
        """Return the output streams of one raw buffer as a dict of arrays (status 0 = valid pixel)."""
        float_p = ctypes.POINTER(ctypes.c_float)
        ret = self._lib.l9t_process(self._handle, payload, len(payload),
                                    *[a.ctypes.data_as(float_p) for a in self._floats],
                                    self._status.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
        if ret == L9T_ERROR_INVALID_METADATA:
            raise InvalidFrame()
        if ret:
            raise RuntimeError(f"processing failed: {media_error(ret)}")
        outputs = {name: a.copy() for name, a in zip(FLOAT_OUTPUTS, self._floats)}
        outputs["status"] = self._status.copy()
        return outputs

    def close(self):
        if self._handle:
            self._lib.l9t_destroy(self._handle)
            self._handle = ctypes.c_void_p()


# --- data source -----------------------------------------------------------------------------------------------------

def load_calib(path):
    """Calibration from a raw calibration dump or from the first L9CB packet of a capture file."""
    data = Path(path).read_bytes()
    if len(data) == rx.CALIB_SIZE:
        return data
    for magic, payload in rx.PacketParser().feed(data):
        if magic == b"L9CB":
            return payload
    raise ValueError(f"no calibration packet in {path}")


def dss_indexes(frame):
    """Per pixel DSS index, packed on 4 bits (2 pixels per byte) as decoded by vl53l9_algo_extract()."""
    h, w = frame["distance"].shape
    pixels = np.arange(w * h)
    shifts = np.where(pixels % 2, 4, 0)
    return ((frame["dss_lut"][pixels // 2] >> shifts) & 0b111).reshape(h, w)


def error_text(payload):
    line, ret = struct.unpack_from("<ii", payload, 0)
    return f"firmware error at vl53l9_app.c:{line} ret={ret} ({rx.DRIVER_ERRORS.get(ret, '?')})"


class Recorder:
    """Keep every frame in memory and write them to a csv file only at the end, so the recording does not slow down
    the reception. One row per pixel: raw values, then the transformed ones (nan if the frame was not transformed),
    then the acquisition time on the host clock (s, nan if unknown)."""

    RAW_COLUMNS = ["distance", "flag", "amplitude", "ambient", "dss_index"]
    TRANSFORMED_COLUMNS = ["depth_mm", "signal_rate", "ambient_rate", "confidence", "reflectance", "transform_status"]
    TRANSFORMED_KEYS = [*FLOAT_OUTPUTS, "status"]

    def __init__(self, path):
        self.path = path
        self.frames = []
        self.start = None

    def add(self, frame, transformed):
        now = time.monotonic()
        if self.start is None:
            self.start = now
        # raw arrays are views on the immutable packet payload, transformed arrays are copies: no copy needed
        self.frames.append((now - self.start, frame, transformed))

    def save(self):
        if not self.frames:
            print(f"[record] no frame received, {self.path} not written")
            return
        print(f"[record] saving {len(self.frames)} frames to {self.path} ...")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(["time_s", "frame_counter", "row", "col", *self.RAW_COLUMNS, *self.TRANSFORMED_COLUMNS,
                           "acq_time_s"])
        with open(self.path, "w") as out:
            out.write(header + "\n")
            for elapsed, frame, transformed in self.frames:
                h, w = frame["distance"].shape
                rows, cols = np.indices((h, w))
                columns = [np.full(h * w, elapsed), np.full(h * w, frame["frame_counter"]), rows.ravel(), cols.ravel(),
                           *[frame[key].ravel() for key in self.RAW_COLUMNS]]
                if transformed is not None and transformed["depth"].size == h * w:
                    columns += [transformed[key].ravel() for key in self.TRANSFORMED_KEYS]
                else:
                    columns += [np.full(h * w, np.nan)] * len(self.TRANSFORMED_KEYS)
                acq = frame.get("acq_time_ns")
                columns.append(np.full(h * w, np.nan if acq is None else acq / 1e9))
                formats = ["%.6f", "%d", "%d", "%d", *["%d"] * len(self.RAW_COLUMNS),
                           *["%.6g"] * len(self.TRANSFORMED_KEYS), "%.6f"]
                np.savetxt(out, np.column_stack(columns), fmt=formats, delimiter=",")
        print(f"[record] saved {self.path}")


def read_recording(path):
    """Yield (elapsed, frame, transformed) from a csv written by Recorder, one frame at a time. transformed is None for
    frames recorded without transformed values."""
    with open(path) as csv:
        header = csv.readline().strip().split(",")
        column = {name: i for i, name in enumerate(header)}
        missing = {"time_s", "frame_counter", "row", "col", *Recorder.RAW_COLUMNS} - column.keys()
        if missing:
            raise ValueError(f"{path} is not a recording, missing columns: {', '.join(sorted(missing))}")
        # rows of one frame share the same time and frame counter, and are written in row-major order
        for _, lines in itertools.groupby(csv, key=lambda line: line.split(",", 2)[:2]):
            values = np.loadtxt(list(lines), delimiter=",", ndmin=2)
            h, w = int(values[:, column["row"]].max()) + 1, int(values[:, column["col"]].max()) + 1
            if len(values) != h * w:
                raise ValueError(f"{path}: truncated frame at t={values[0, column['time_s']]}")
            frame = {key: values[:, column[key]].astype(np.int32).reshape(h, w) for key in Recorder.RAW_COLUMNS}
            frame["frame_counter"] = int(values[0, column["frame_counter"]])
            transformed = None
            if all(name in column for name in Recorder.TRANSFORMED_COLUMNS):
                outputs = [values[:, column[name]].reshape(h, w) for name in Recorder.TRANSFORMED_COLUMNS]
                if np.isfinite(outputs[0]).any():
                    transformed = {key: a.astype(np.float32) for key, a in zip(FLOAT_OUTPUTS, outputs)}
                    transformed["status"] = outputs[-1].astype(np.uint32)
            yield values[0, column["time_s"]], frame, transformed


class Reader:
    """Read packets from the serial port or from a capture file, run the pipeline on every frame (in order, for the
    temporal filtering) and publish the results to the display process."""

    def __init__(self, args, calib, results, stop_event):
        self.args = args
        self.results = results
        self.stop_event = stop_event
        self.parser = rx.PacketParser()
        self.stats = rx.Stats()
        self.clock = ts.StreamClock()  # no port: no sync, frames get only the board times
        self.calib = calib
        self.transform_enabled = not args.no_transform
        self.pipeline = None
        self.last_transformed = None
        self.recorder = Recorder(args.record_path) if args.record_path else None
        self.frame_times = collections.deque(maxlen=STREAM_FPS_FRAMES)
        self.info = dict(calib_source="file" if calib else "default", error=None, transform_error=None,
                         lost_frames=0, invalid_frames=0, stream_fps=0.0)

    def run(self):
        try:
            if self.args.file:
                self._run_file()
            elif self.args.replay:
                self._run_replay()
            else:
                self._run_serial()
        except (OSError, ValueError, serial.SerialException) as e:
            print(f"[error] {e}")
            self.info["error"] = str(e)
            self._publish(None, wait=True)
        finally:
            self._close_pipeline()
            if self.recorder is not None:
                self.recorder.save()

    def _run_serial(self):
        last_frame = None
        with serial.Serial(self.args.port, self.args.baud, timeout=0.1) as port:
            port.reset_input_buffer()
            self.clock = ts.StreamClock(port, self.args.clock, self.args.trigger_latency_us)
            while not self.stop_event.is_set():
                self.clock.poll()
                data = port.read(min(max(port.in_waiting, 1), rx.READ_CHUNK))
                t_rx = self.clock.now_ns()  # reception time of the packets completed by this read (clock sync)
                if data:
                    self.stats.on_bytes(len(data))
                    for magic, payload in self.parser.feed(data):
                        last_frame = self._handle_packet(magic, payload, t_rx) or last_frame
                self.stats.report(time.monotonic(), self.parser, last_frame, self.clock.status())

    def _run_file(self):
        period = 1.0 / self.args.fps
        next_time = time.monotonic()
        with open(self.args.file, "rb") as capture:
            while not self.stop_event.is_set():
                data = capture.read(FILE_CHUNK)
                if not data:
                    print("end of capture file")
                    return
                for magic, payload in self.parser.feed(data):
                    if self._handle_packet(magic, payload) is not None:
                        # replay at the requested frame rate
                        next_time += period
                        time.sleep(max(0.0, next_time - time.monotonic()))
                    if self.stop_event.is_set():
                        return

    def _run_replay(self):
        """Replay a csv recording with its original timing (scaled by --speed), the transformed values are the recorded
        ones: the pipeline is not run again."""
        self.info["calib_source"] = "recorded"
        start = time.monotonic()
        for elapsed, frame, transformed in read_recording(self.args.replay):
            time.sleep(max(0.0, start + elapsed / self.args.speed - time.monotonic()))
            if self.stop_event.is_set():
                return
            self.stats.on_frame(frame)
            self.frame_times.append(time.monotonic())
            if transformed is None and self.transform_enabled:
                # frame skipped by the pipeline while recording: keep showing the previous output, as the live viewer
                if self.last_transformed is not None:
                    self.info["invalid_frames"] += 1
                else:
                    self.info["transform_error"] = "no transformed values in the recording"
            elif self.transform_enabled:
                self.last_transformed = transformed
            self._publish((frame, self.last_transformed))
        print("end of recording")
        self._publish(None, wait=True)

    def _handle_packet(self, magic, payload, t_rx=None):
        """Process one packet, return the decoded frame for L9RW packets."""
        self.clock.on_packet(magic, payload, t_rx)
        if magic == b"L9CB":
            print(f"calibration data received ({len(payload)} bytes)")
            self.calib, self.info["calib_source"] = payload, "sensor"
            self._close_pipeline()  # restart the temporal filtering with the new calibration
        elif magic == b"L9ER":
            rx.print_error(payload)
            self.info["error"] = error_text(payload)
            self._publish(None, wait=True)
        elif magic == b"L9TM":
            rx.print_profile(payload)
        elif magic == b"L9RW":
            frame = self.clock.stamp(rx.parse_frame(payload))
            frame["dss_index"] = dss_indexes(frame)
            self.stats.on_frame(frame)
            self.frame_times.append(time.monotonic())
            invalid_before = self.info["invalid_frames"]
            transformed = self._transform(payload)
            if self.recorder is not None:
                # frames skipped by the pipeline are recorded without transformed values
                self.recorder.add(frame, transformed if self.info["invalid_frames"] == invalid_before else None)
            self._publish((frame, transformed))
            return frame
        return None

    def _transform(self, payload):
        if not self.transform_enabled:
            return None
        try:
            if self.pipeline is None:
                if self.calib is None:
                    print("[warn] no calibration received (reset the board or use --calib), "
                          "using the default calibration of the library")
                self.pipeline = HostTransform(self.calib, len(payload))
            self.last_transformed = self.pipeline.process(payload)
        except InvalidFrame:
            # keep showing the previous output
            self.info["invalid_frames"] += 1
        except (OSError, RuntimeError) as e:
            # keep showing the raw data
            print(f"[error] transform disabled: {e}")
            self.info["transform_error"] = str(e)
            self.transform_enabled = False
            self.last_transformed = None
            self._close_pipeline()
        return self.last_transformed

    def _close_pipeline(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None

    def _publish(self, item, wait=False):
        """Send (frame, transformed) or None (status only) to the display, frames are dropped if it is late."""
        times = self.frame_times
        self.info["stream_fps"] = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
        self.info["lost_frames"] = self.stats.gaps
        try:
            if wait:
                self.results.put((item, dict(self.info)), timeout=1.0)
            else:
                self.results.put_nowait((item, dict(self.info)))
        except queue.Full:
            pass


def reader_main(args, calib, results, stop_event):
    # Ctrl+C is handled by the main process, which stops the reader through stop_event: an interrupt here could
    # abort the saving of the recording
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        Reader(args, calib, results, stop_event).run()
    except KeyboardInterrupt:
        pass


class FrameSource:
    """Display side of the reader process."""

    def __init__(self, args, calib):
        context = multiprocessing.get_context("spawn")
        self.results = context.Queue(maxsize=RESULT_QUEUE_SIZE)
        self.stop_event = context.Event()
        self.process = context.Process(target=reader_main, args=(args, calib, self.results, self.stop_event),
                                       name="l9-reader", daemon=True)
        self.args = args
        self.latest = None
        self.info = {}

    def start(self):
        self.process.start()

    def stop(self):
        self.stop_event.set()
        # the reader writes the recording before exiting, it may take a while
        self.process.join(timeout=None if self.args.record_path else 2.0)
        if self.process.is_alive():
            self.process.terminate()

    def poll(self):
        """Return the last (frame, transformed) received, None before the first frame."""
        while True:
            try:
                item, self.info = self.results.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self.latest = item
        if self.process.exitcode not in (None, 0) and not self.info.get("error"):
            self.info = {**self.info, "error": f"reader process stopped (exit code {self.process.exitcode})"}
        return self.latest


# --- display ---------------------------------------------------------------------------------------------------------

class AutoRange:
    """Color range following the data percentiles, smoothed to avoid flickering, or fixed."""

    def __init__(self, fixed=None):
        self.fixed = fixed
        self.low = self.high = None

    def update(self, values):
        if self.fixed:
            return self.fixed
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.percentile(values, AUTO_RANGE_PERCENTILES)
            high = max(high, low + 1e-6)
            if self.low is None:
                self.low, self.high = low, high
            else:
                self.low += AUTO_RANGE_SMOOTHING * (low - self.low)
                self.high += AUTO_RANGE_SMOOTHING * (high - self.high)
        return (self.low, self.high) if self.low is not None else (0.0, 1.0)


# (row, column, title, source, key, colormap, color range group)
PANELS = [
    (0, 0, "Raw distance", "raw", "distance", "turbo", "distance"),
    (0, 1, "Raw amplitude", "raw", "amplitude", "gray", "raw_amplitude"),
    (0, 2, "Raw ambient", "raw", "ambient", "gray", "raw_ambient"),
    (0, 3, "Raw DSS index", "raw", "dss_index", "viridis", "dss_index"),
    (1, 0, "Depth [mm]", "transformed", "depth", "turbo", "distance"),
    (1, 1, "Signal rate", "transformed", "amplitude", "gray", "amplitude"),
    (1, 2, "Ambient rate", "transformed", "ambient", "gray", "ambient"),
    (1, 3, "Confidence", "transformed", "confidence", "viridis", "confidence"),
]


class Viewer:
    """Figure updated with blitting.

    A full draw (layout, axes titles, ~150 ms) only happens when the window changes. Each frame then redraws the images
    and the title over the backgrounds saved after the full draw, plus one colorbar in turn (tick labels are slow).
    """

    def __init__(self, source, args):
        self.source = source
        self.args = args
        self.shown = None
        self.ranges = {group: AutoRange() for *_, group in PANELS}
        self.ranges["distance"].fixed = tuple(args.range) if args.range else None
        self.ranges["dss_index"].fixed = (0, 7)
        self.backgrounds = None
        self.title_background = None
        self.next_colorbar = 0
        self.last_render = 0.0
        self.render_times = collections.deque(maxlen=STREAM_FPS_FRAMES)

        self.fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), layout="constrained")
        self.fig.canvas.manager.set_window_title("VL53L9 raw viewer")
        self.images = []
        for row, col, title, src, key, cmap, group in PANELS:
            ax = axes[row, col]
            colormap = plt.get_cmap(cmap).copy()
            colormap.set_bad("0.35")  # invalid depth pixels
            image = ax.imshow(np.zeros((1, 1)), cmap=colormap, interpolation="nearest", animated=True)
            colorbar = self.fig.colorbar(image, ax=ax, shrink=0.8)
            colorbar.ax.set_animated(True)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            self.images.append((image, colorbar, src, key, group))
        axes[0, 0].set_ylabel("RAW (sensor)", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("TRANSFORMED (vl53l9-transform-c)", fontsize=12, fontweight="bold")
        self.bottom_text = axes[1, 0].text(0.5, 0.5, "", transform=axes[1, 0].transAxes, ha="center", va="center",
                                           animated=True)
        self.title = self.fig.suptitle("waiting for frames...", animated=True)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event):
        """After a full draw (animated artists excluded), save the backgrounds and draw the animated artists."""
        canvas = self.fig.canvas
        axes_boxes = [image.axes.bbox for image, *_ in self.images]
        colorbar_boxes = [colorbar.ax.bbox for _, colorbar, *_ in self.images]
        # NOTE: with equal aspect the images are shorter than their cells, colorbars can reach the axes titles
        title_bottom = max([image.axes.title.get_window_extent(event.renderer).y1 for image, *_ in self.images] +
                           [b.y1 + COLORBAR_MARGIN_PX for b in colorbar_boxes])
        self.backgrounds = []
        for box, bar in zip(axes_boxes, colorbar_boxes):
            # colorbar area (tick labels included): right of its image up to the next image, vertically shared
            # with the colorbars above and below
            right = min((b.x0 for b in axes_boxes if b.x0 > box.x1 and abs(b.y0 - box.y0) < 1), default=self.fig.bbox.x1)
            column = [b for b in colorbar_boxes if abs(b.x0 - bar.x0) < 1]
            top = min(((bar.y1 + b.y0) / 2 for b in column if b.y0 > bar.y1), default=title_bottom)
            bottom = max(((b.y1 + bar.y0) / 2 for b in column if b.y1 < bar.y0), default=0)
            colorbar_box = Bbox.from_extents(box.x1 + 1, max(bar.y0 - COLORBAR_MARGIN_PX, bottom),
                                             right - 1, min(bar.y1 + COLORBAR_MARGIN_PX, top))
            self.backgrounds.append((canvas.copy_from_bbox(box), canvas.copy_from_bbox(colorbar_box)))
        self.title_background = canvas.copy_from_bbox(
            Bbox.from_extents(0, title_bottom, self.fig.bbox.x1, self.fig.bbox.y1))
        for image, colorbar, *_ in self.images:
            self.fig.draw_artist(image)
            self.fig.draw_artist(colorbar.ax)
        self.fig.draw_artist(self.bottom_text)
        self.fig.draw_artist(self.title)

    def _blit(self):
        canvas = self.fig.canvas
        for (image, *_), (image_background, _) in zip(self.images, self.backgrounds):
            canvas.restore_region(image_background)
            self.fig.draw_artist(image)
        self.fig.draw_artist(self.bottom_text)
        _, colorbar, *_ = self.images[self.next_colorbar]
        canvas.restore_region(self.backgrounds[self.next_colorbar][1])
        self.fig.draw_artist(colorbar.ax)
        self.next_colorbar = (self.next_colorbar + 1) % len(self.images)
        canvas.restore_region(self.title_background)
        self.fig.draw_artist(self.title)
        canvas.blit(self.fig.bbox)

    def orient(self, data):
        if self.args.flip_x:
            data = np.flip(data, axis=1)
        if self.args.flip_y:
            data = np.flip(data, axis=0)
        return data

    def update(self):
        latest = self.source.poll()
        now = time.monotonic()
        is_new_frame = latest is not None and latest is not self.shown
        if is_new_frame:
            self.shown = latest
            self.draw_frame(*latest)
            self.render_times.append(now)
        elif now - self.last_render < TITLE_REFRESH_S:
            return
        self.draw_title()
        if self.backgrounds is None:
            self.fig.canvas.draw_idle()
        else:
            self._blit()
        self.last_render = now

    def draw_frame(self, frame, transformed):
        data = {"raw": frame, "transformed": transformed}
        if transformed is not None and not self.args.show_invalid:
            transformed = dict(transformed)
            transformed["depth"] = np.ma.masked_where(transformed["status"] != 0, transformed["depth"])
            data["transformed"] = transformed

        # raw distance and depth share the same color range to compare them
        distance_values = [frame["distance"].astype(np.float32).ravel()]
        if transformed is not None:
            distance_values = [np.ma.compressed(transformed["depth"]).astype(np.float32)]
        group_values = {"distance": np.concatenate(distance_values)}

        for image, colorbar, src, key, group in self.images:
            image.set_visible(data[src] is not None)
            colorbar.ax.set_visible(data[src] is not None)
            if data[src] is None:
                continue
            values = self.orient(data[src][key])
            h, w = values.shape
            image.set_data(values)
            image.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            finite = group_values.get(group, np.ma.compressed(values).astype(np.float32).ravel())
            image.set_clim(*self.ranges[group].update(finite))
        self.bottom_text.set_text("" if transformed is not None else self.transform_status())

    def transform_status(self):
        if self.args.no_transform:
            return "transform disabled (--no-transform)"
        return f"transform unavailable:\n{self.source.info.get('transform_error') or 'see console'}"

    def display_fps(self):
        times = self.render_times
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def draw_title(self):
        info = self.source.info
        latest = self.shown
        if latest is None:
            text = "waiting for frames..."
        else:
            frame = latest[0]
            h, w = frame["distance"].shape
            calib = info["calib_source"] if latest[1] is not None else "-"
            text = (f"frame {frame['frame_counter']}   {w}x{h}   stream {info['stream_fps']:.1f} fps   "
                    f"display {self.display_fps():.1f} fps   lost frames {info['lost_frames']}   "
                    f"invalid frames {info['invalid_frames']}   calibration: {calib}")
        if info.get("error"):
            text += f"\n{info['error']}"
        self.title.set_text(text)
        self.title.set_color("red" if info.get("error") else "black")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port", nargs="?", help="serial port of the board, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=12500000, help="must match CONF_RAW_BAUDRATE")
    parser.add_argument("--file", help="replay a capture made with l9_raw_receiver.py --out instead of a port")
    parser.add_argument("--fps", type=float, default=30.0, help="replay frame rate with --file")
    parser.add_argument("--replay", metavar="CSV", help="replay a recording made with --record instead of a port")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed factor with --replay")
    parser.add_argument("--calib", help="calibration from a capture file (or a raw 2332 bytes dump)")
    parser.add_argument("--range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="fixed color range of raw distance and depth (automatic by default)")
    parser.add_argument("--show-invalid", action="store_true", help="do not hide depth pixels flagged as invalid")
    parser.add_argument("--no-transform", action="store_true", help="show only the raw data")
    parser.add_argument("--flip-x", action="store_true", help="mirror the images horizontally")
    parser.add_argument("--flip-y", action="store_true", help="mirror the images vertically")
    parser.add_argument("--refresh", type=float, default=60.0, help="display polling rate in Hz")
    parser.add_argument("--clock", choices=ts.CLOCKS, default="realtime",
                        help="host clock of the acquisition times (realtime: librealsense global time)")
    parser.add_argument("--trigger-latency-us", type=float, default=0.0,
                        help="latency between the trigger command and the start of the exposure")
    parser.add_argument("--record", nargs="?", const="", metavar="NAME",
                        help="record every frame to Records/NAME.csv (current date and time by default), "
                             "written when the viewer is closed or stopped with Ctrl+C")
    args = parser.parse_args()
    if sum(map(bool, (args.port, args.file, args.replay))) != 1:
        parser.error("give either a serial port, --file or --replay")
    if args.replay and args.record is not None:
        parser.error("--record cannot be used with --replay")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    args.record_path = None
    if args.record is not None:
        name = args.record.removesuffix(".csv") or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.record_path = RECORDS_DIR / f"{name}.csv"
        print(f"[record] recording to {args.record_path} (written at exit)")

    if not args.no_transform and not args.replay:
        try:
            build_library()
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"[error] cannot build the transform library, showing raw data only: {e}")
            args.no_transform = True

    try:
        calib = load_calib(args.calib) if args.calib else None
    except (OSError, ValueError) as e:
        parser.error(f"--calib: {e}")
    source = FrameSource(args, calib)
    source.start()
    viewer = Viewer(source, args)
    timer = viewer.fig.canvas.new_timer(interval=int(1000.0 / args.refresh))
    timer.add_callback(viewer.update)
    timer.start()
    # Ctrl+C closes the window, so that the reader is stopped cleanly (and the recording saved)
    signal.signal(signal.SIGINT, lambda *_: plt.close("all"))
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        timer.stop()
        source.stop()


if __name__ == "__main__":
    main()
