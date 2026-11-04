"""Live VL53L9 (L9) filtered depth for the lerobot visualization (Rerun / Foxglove), display only.

The sensor is NOT a robot camera: it is not in ``robot.get_observation()``, not in the dataset and not seen by
the policies. :mod:`lerobot.utils.visualization_utils` adds its last depth frame to the dict it sends to the
viewer, nothing else.

The serial reception and the vl53l9-transform-c pipeline run in a separate process, with the ``Reader`` of
``L9_sensor/l9_visualizer.py`` (same calibration handling, temporal filtering and invalid frame handling as the
standalone viewer): the robot control loop never waits for the uart, and the uart reads are never delayed by the
control loop (the ST-LINK VCP has no flow control, a late reader loses bytes).

The displayed image is the transformed depth (``Depth [mm]`` panel of the standalone viewer) as ``uint16`` mm with
the pixels flagged as invalid by the pipeline (status != 0) set to 0.

The sensor is optional at run time: while no depth arrives (board not connected, unplugged, transform unavailable) a
"NO DEPTH" image is shown in its panel and the robot keeps running. The reader process retries to open the port
every RECONNECT_PERIOD_S, so the depth appears as soon as the board is plugged in.
"""

import argparse
import contextlib
import logging
import multiprocessing
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

L9_SENSOR_DIR = Path(__file__).resolve().parent / "L9_sensor"
# key of the depth image in the dict sent to the viewer, logged as "observation.l9_tof_depth"
TOF_DISPLAY_KEY = "l9_tof_depth"
RESULT_QUEUE_SIZE = 2  # frames waiting for the control loop, older ones are dropped by the reader
STOP_TIMEOUT_S = 2.0
RECONNECT_PERIOD_S = 2.0  # delay before reopening the port after an error (board not connected, unplugged)
NO_DEPTH_TIMEOUT_S = 1.0  # without depth for this long the "NO DEPTH" image replaces the last frame
SENSOR_SHAPE = (42, 54)  # (height, width) of the full resolution frames, used for the "NO DEPTH" image
PLACEHOLDER_SCALE = 4  # the "NO DEPTH" image is larger than the sensor so that the text is readable
PLACEHOLDER_VALUE_MM = 1000  # depth value of the text pixels, the background is 0 (no data)


@dataclass
class L9DisplayConfig:
    """Serial link of the L9 board (raw output mode) and orientation of the displayed image."""

    port: str
    baud: int = 12_500_000  # must match CONF_RAW_BAUDRATE of the firmware
    # calibration from a capture of l9_raw_receiver.py --out (or a raw dump), used until the board sends its own
    # one at boot (reset the board after the start to get it)
    calib: str | None = None
    flip_x: bool = False
    flip_y: bool = False


def _reader_main(config: L9DisplayConfig, results, stop_event) -> None:
    """Reader process: l9_visualizer.Reader on the serial port, results published to ``results``."""
    # Ctrl+C is handled by the lerobot process, which stops the reader through stop_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # the Reader prints statistics every second: keep the lerobot console readable, the errors are sent to the
    # lerobot process in the info dict
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    os.environ.setdefault("MPLBACKEND", "Agg")  # l9_visualizer imports pyplot, no window is opened
    sys.path.insert(0, str(L9_SENSOR_DIR))  # the L9 scripts import each other as top-level modules
    try:
        import l9_visualizer as viz
    except ImportError as e:
        results.put(
            (None, {"error": f"cannot import l9_visualizer ({e}), install numpy, pyserial and matplotlib"})
        )
        return

    no_transform = False
    if not viz.LIBRARY_PATH.exists():
        try:
            viz.build_library()
        except Exception as e:  # noqa: BLE001 - missing gcc or transform sources
            results.put((None, {"error": f"cannot build {viz.LIBRARY_PATH.name}: {e}"}))
            no_transform = True
    calib = None
    if config.calib:
        try:
            calib = viz.load_calib(config.calib)
        except (OSError, ValueError) as e:
            results.put((None, {"error": f"calib: {e}"}))
    args = argparse.Namespace(
        port=config.port,
        baud=config.baud,
        file=None,
        replay=None,
        no_transform=no_transform,
        record_path=None,
        clock="realtime",
        trigger_latency_us=0.0,
    )
    # Reader.run() returns on a serial error (port missing, board unplugged) after publishing it: retry until stopped
    while not stop_event.is_set():
        try:
            viz.Reader(args, calib, results, stop_event).run()
        except KeyboardInterrupt:
            pass
        except Exception as e:  # noqa: BLE001 - e.g. termios.error on an unsupported baudrate: retry, never die
            with contextlib.suppress(queue.Full):
                results.put((None, {"error": f"{type(e).__name__}: {e}"}), timeout=1.0)
        stop_event.wait(RECONNECT_PERIOD_S)


def _no_depth_image(shape: tuple[int, int]) -> np.ndarray:
    """``uint16`` (H, W, 1) image with "NO DEPTH" written in it, same aspect ratio as the depth frames."""
    height, width = shape[0] * PLACEHOLDER_SCALE, shape[1] * PLACEHOLDER_SCALE
    text, font, thickness = "NO DEPTH", cv2.FONT_HERSHEY_SIMPLEX, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = 0.8 * width / text_w
    text_w, text_h = int(text_w * scale), int(text_h * scale)
    mask = np.zeros((height, width), dtype=np.uint8)  # putText draws only on 8-bit images in recent opencv
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(mask, text, origin, font, scale, 255, thickness, cv2.LINE_8)
    return np.where(mask > 0, PLACEHOLDER_VALUE_MM, 0).astype(np.uint16)[..., None]


class L9DepthStream:
    """Lerobot side of the reader process: last filtered depth frame of the L9 sensor."""

    def __init__(self, config: L9DisplayConfig):
        """Prepare the reader process, started by :meth:`start`."""
        self.config = config
        context = multiprocessing.get_context("spawn")  # no fork of the lerobot process (threads, cuda)
        self._results = context.Queue(maxsize=RESULT_QUEUE_SIZE)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_reader_main,
            args=(config, self._results, self._stop_event),
            name="l9-tof-reader",
            daemon=True,
        )
        self._last_counter: int | None = None
        self._reported: dict[str, object] = {}
        self._last_depth_time = -float("inf")
        self._last_shape = SENSOR_SHAPE
        self._showing_no_depth = False

    def start(self) -> None:
        """Start the reader process (serial reception and transform pipeline)."""
        self._process.start()
        logger.info("L9 ToF display: reading %s at %d baud", self.config.port, self.config.baud)

    def stop(self) -> None:
        """Stop the reader process, killed if it does not exit within STOP_TIMEOUT_S."""
        self._stop_event.set()
        self._process.join(timeout=STOP_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()

    def poll(self) -> np.ndarray | None:
        """Image to show in the depth panel, None if the panel does not change (the viewer keeps the last image).

        The filtered depth of the last frame received since the previous call, or the "NO DEPTH" image once when
        no depth arrived for NO_DEPTH_TIMEOUT_S (immediately at the first call). ``uint16`` (H, W, 1) in mm, 0 for
        the invalid pixels: the viewers show single channel integer images as depth in mm.
        """
        depth = self._poll_depth()
        now = time.monotonic()
        if depth is not None:
            if self._showing_no_depth:
                logger.info("L9 ToF display: depth received")
            self._showing_no_depth = False
            self._last_depth_time = now
            self._last_shape = depth.shape[:2]
            return depth
        if not self._showing_no_depth and now - self._last_depth_time > NO_DEPTH_TIMEOUT_S:
            self._showing_no_depth = True
            logger.warning("L9 ToF display: no depth from %s, showing NO DEPTH", self.config.port)
            return _no_depth_image(self._last_shape)
        return None

    def _poll_depth(self) -> np.ndarray | None:
        """Filtered depth of the last frame received since the previous call, None if there is no new one."""
        exitcode = self._process.exitcode
        if exitcode not in (None, 0) and self._reported.get("exitcode") != exitcode:
            self._reported["exitcode"] = exitcode
            logger.error("L9 ToF display: reader process stopped (exit code %s)", exitcode)
        latest = None
        while True:
            try:
                item, info = self._results.get_nowait()
            except queue.Empty:
                break
            self._report(info)
            if item is not None:
                latest = item
        if latest is None:
            return None
        frame, transformed = latest
        if transformed is None or frame["frame_counter"] == self._last_counter:
            return None  # transform unavailable (reported) or frame already shown
        self._last_counter = frame["frame_counter"]
        depth = transformed["depth"]
        valid = (transformed["status"] == 0) & np.isfinite(depth) & (depth > 0)
        depth_mm = np.where(valid, np.clip(np.rint(depth), 1, np.iinfo(np.uint16).max), 0).astype(np.uint16)
        if self.config.flip_x:
            depth_mm = depth_mm[:, ::-1]
        if self.config.flip_y:
            depth_mm = depth_mm[::-1, :]
        return np.ascontiguousarray(depth_mm)[..., None]

    def _report(self, info: dict) -> None:
        """Log the errors and the calibration source of the reader once, when they change."""
        for key, level in (
            ("error", logging.ERROR),
            ("transform_error", logging.ERROR),
            ("calib_source", logging.INFO),
        ):
            value = info.get(key)
            if value and value != self._reported.get(key):
                logger.log(level, "L9 ToF display: %s = %s", key, value)
            self._reported[key] = value


def make_tof_display_config(cfg) -> L9DisplayConfig | None:
    """L9 display config from the ``display_tof_*`` fields of a lerobot script config, None if no port is given."""
    if not getattr(cfg, "display_tof_port", None):
        return None
    return L9DisplayConfig(
        port=cfg.display_tof_port,
        baud=cfg.display_tof_baud,
        calib=cfg.display_tof_calib,
        flip_x=cfg.display_tof_flip_x,
        flip_y=cfg.display_tof_flip_y,
    )
