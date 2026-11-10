"""Live FlexiTac tactile heatmap for the lerobot visualization (Rerun / Foxglove), display only.

The sensor is NOT a robot camera: it is not in ``robot.get_observation()``, not in the dataset and not seen by
the policies. :mod:`lerobot.utils.visualization_utils` adds its last heatmap to the dict it sends to the viewer,
nothing else.

The board runs ``FlexiTac/32_16_fast.ino``: frames of ``0xAA 0x55`` + 16 x 32 ``uint8`` values at 2 Mbaud. The
processing is set by ``flexitac_config.yaml`` (see the comments of that file), its defaults are the processing of
``FlexiTac/fast_32_16.py``: median baseline of the first frames, baseline and threshold subtracted, normalization
(by the frame max, or by a fixed scale), temporal filter (applied per received frame) and colormap. The baseline is
taken again at each (re)connection: do not touch the sensor during the first frames.

The serial reception runs in a separate process: the robot control loop never waits for the serial port, and the
serial reads are never delayed by the control loop (no flow control, a late reader loses bytes).

The sensor is optional at run time: while no frame arrives (board not connected, unplugged, baseline in progress)
a "NO TACTILE" image is shown in its panel and the robot keeps running. The reader process retries to open the
port every RECONNECT_PERIOD_S, so the heatmap appears as soon as the board is plugged in.
"""

import contextlib
import logging
import multiprocessing
import queue
import signal
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# key of the heatmap in the dict sent to the viewer, logged as "observation.flexitac_tactile"
TACTILE_DISPLAY_KEY = "flexitac_tactile"

# serial format of 32_16_fast.ino
MAGIC = b"\xaa\x55"
ROWS, COLS = 16, 32
FRAME_BYTES = ROWS * COLS
MAX_BUFFER_BYTES = 50_000  # as fast_32_16.py: older bytes are dropped

# default settings file of the processing, see TactileSettings
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "flexitac_config.yaml"
# official FlexiTac repo (git submodule): firmware of the board and original viewer, not needed by this module
FLEXITAC_REPO_DIR = Path(__file__).resolve().parent / "FlexiTac"
FLEXITAC_FIRMWARE = FLEXITAC_REPO_DIR / "32_16_fast.ino"
FLEXITAC_SETUP_HINT = (
    "if the board has no firmware yet: get the official FlexiTac repo with "
    "`git submodule update --init src/lerobot/tactile_sensor/FlexiTac` (or clone "
    f"https://github.com/FlexiTac/FlexiTac into {FLEXITAC_REPO_DIR}), then flash FlexiTac/32_16_fast.ino with the "
    "Arduino IDE (see lerobot/tactile_sensor/README.md)"
)
NORMALIZATIONS = ("per_frame_max", "fixed")

READ_SIZE = 8192
READ_TIMEOUT_S = 0.01
RESULT_QUEUE_SIZE = 2  # frames waiting for the control loop, older ones are dropped by the reader
STOP_TIMEOUT_S = 2.0
RECONNECT_PERIOD_S = 2.0  # delay before reopening the port after an error (board not connected, unplugged)
NO_DATA_TIMEOUT_S = 1.0  # without frames for this long the "NO TACTILE" image replaces the last heatmap


@dataclass
class TactileSettings:
    """Processing of the heatmap, from flexitac_config.yaml (meaning of each setting in the comments of that file).

    The defaults are the processing of FlexiTac/fast_32_16.py.
    """

    baseline_frames: int = 30
    threshold: float = 15.0
    contact_clip: float = 100.0
    normalization: str = "per_frame_max"
    fixed_scale: float = 100.0
    noise_scale: float = 60.0
    temporal_alpha: float = 0.2
    colormap: str = "viridis"
    display_scale: int = 10

    def __post_init__(self) -> None:
        """Check the values: a wrong setting is reported at start instead of giving a strange image."""
        if self.baseline_frames < 1:
            raise ValueError("baseline_frames must be >= 1")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        for name in ("contact_clip", "fixed_scale", "noise_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {', '.join(NORMALIZATIONS)}, not {self.normalization!r}"
            )
        if not 0 < self.temporal_alpha <= 1:
            raise ValueError("temporal_alpha must be in ]0, 1]")
        if not hasattr(cv2, f"COLORMAP_{self.colormap.upper()}"):
            raise ValueError(f"unknown OpenCV colormap {self.colormap!r}")
        if self.display_scale < 1:
            raise ValueError("display_scale must be >= 1")

    @property
    def colormap_id(self) -> int:
        """OpenCV constant of the colormap."""
        return getattr(cv2, f"COLORMAP_{self.colormap.upper()}")


def load_tactile_settings(path: str | Path) -> TactileSettings:
    """Settings from a yaml file: missing keys keep their default, unknown keys and wrong values raise ValueError."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("the file must be a mapping of settings")
    # type of each setting = type of its default (int, float or str)
    types: dict[str, type] = {field.name: type(field.default) for field in fields(TactileSettings)}
    unknown = sorted(set(data) - set(types))
    if unknown:
        raise ValueError(f"unknown settings {', '.join(unknown)}, expected {', '.join(types)}")
    values: dict[str, Any] = {}
    for name, value in data.items():
        kind = types[name]
        # an int is accepted for a float setting (threshold: 6), a bool is never accepted (yaml true/false)
        if isinstance(value, bool) or not isinstance(value, (int, float) if kind is float else kind):
            raise ValueError(f"{name} must be of type {kind.__name__}, not {value!r}")
        values[name] = kind(value)
    return TactileSettings(**values)


@dataclass
class FlexiTacDisplayConfig:
    """Serial link of the FlexiTac board, processing settings and orientation of the displayed heatmap."""

    port: str
    baud: int = 2_000_000  # must match BAUD_RATE of 32_16_fast.ino
    # processing settings file, None: flexitac_config.yaml next to this module
    config: str | None = None
    flip_x: bool = False
    flip_y: bool = False


class FrameParser:
    """Splits the serial byte stream in ``(ROWS, COLS)`` ``uint8`` frames, as the reader of fast_32_16.py."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[np.ndarray]:
        """Add received bytes, return the complete frames found in the stream."""
        self._buffer.extend(chunk)
        if len(self._buffer) > MAX_BUFFER_BYTES:
            del self._buffer[:-MAX_BUFFER_BYTES]
        frames = []
        while True:
            idx = self._buffer.find(MAGIC)
            if idx < 0:
                del self._buffer[:-1]  # keep the last byte: the marker may be split between two reads
                break
            del self._buffer[:idx]
            if len(self._buffer) < len(MAGIC) + FRAME_BYTES:
                break
            data = bytes(self._buffer[len(MAGIC) : len(MAGIC) + FRAME_BYTES])
            del self._buffer[: len(MAGIC) + FRAME_BYTES]
            frames.append(np.frombuffer(data, dtype=np.uint8).reshape(ROWS, COLS))
        return frames


class TactileProcessor:
    """Baseline, threshold, normalization and temporal filter: raw frames -> contact map in [0, 1]."""

    def __init__(self, settings: TactileSettings | None = None) -> None:
        """Start without baseline: the first ``settings.baseline_frames`` frames are used to take it."""
        self.settings = settings or TactileSettings()
        self._init_frames: list[np.ndarray] = []
        self._baseline: np.ndarray | None = None
        self._filtered = np.zeros((ROWS, COLS), dtype=np.float32)
        # deviation of the frames at rest from the baseline (raw units), to choose the threshold
        self.baseline_noise: tuple[float, float] | None = None  # (99th percentile, max)

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Filtered contact map of ``frame``, None while the baseline is being taken."""
        s = self.settings
        frame = frame.astype(np.float32)
        if self._baseline is None:
            self._init_frames.append(frame)
            if len(self._init_frames) >= s.baseline_frames:
                stack = np.stack(self._init_frames)
                self._baseline = np.median(stack, axis=0)
                deviation = np.abs(stack - self._baseline)
                self.baseline_noise = (float(np.percentile(deviation, 99)), float(deviation.max()))
                self._init_frames = []
            return None
        contact = np.clip(frame - self._baseline - s.threshold, 0, s.contact_clip)
        if s.normalization == "fixed":
            normalized = np.minimum(contact / s.fixed_scale, 1.0)
        else:
            peak = float(contact.max())
            normalized = contact / s.noise_scale if peak < s.threshold else contact / (peak + 1e-6)
        self._filtered = s.temporal_alpha * normalized + (1 - s.temporal_alpha) * self._filtered
        return self._filtered.copy()


def _publish(results, item: tuple | None, info: dict) -> None:
    """Put ``(item, info)`` in ``results``, dropping the oldest entry when the control loop is late."""
    for _ in range(RESULT_QUEUE_SIZE + 1):
        try:
            results.put_nowait((item, info))
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                results.get_nowait()


def _stream(ser, results, stop_event, settings: TactileSettings) -> None:
    """Read frames from the open port until stopped or until a serial error (raised)."""
    parser, processor = FrameParser(), TactileProcessor(settings)
    _publish(results, None, {"state": "taking baseline, do not touch the sensor"})
    while not stop_event.is_set():
        chunk = ser.read(READ_SIZE)
        if not chunk:
            continue
        for frame in parser.feed(chunk):
            received = time.monotonic()  # same clock in all the processes: age of the sample in the dataset
            contact = processor.process(frame)
            # state sent with every frame: a single message could be dropped when the queue is full
            info = {
                "state": "streaming" if contact is not None else "taking baseline, do not touch the sensor"
            }
            if processor.baseline_noise is not None:
                p99, peak = processor.baseline_noise
                info["baseline"] = (
                    f"noise at rest p99 {p99:.1f}, max {peak:.1f} (threshold {settings.threshold:g})"
                )
            # raw frame always (recorded in the dataset, independent of the baseline), contact None during the baseline
            _publish(results, (frame, contact, received), info)


def _reader_main(config: FlexiTacDisplayConfig, settings: TactileSettings, results, stop_event) -> None:
    """Reader process: FlexiTac frames from the serial port, contact maps published to ``results``."""
    # Ctrl+C is handled by the lerobot process, which stops the reader through stop_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import serial
    except ImportError as e:
        _publish(results, None, {"error": f"cannot import pyserial ({e})"})
        return
    while not stop_event.is_set():
        try:
            with serial.Serial(config.port, config.baud, timeout=READ_TIMEOUT_S) as ser:
                ser.reset_input_buffer()
                _stream(ser, results, stop_event, settings)
        except Exception as e:  # noqa: BLE001 - port missing, board unplugged, permission denied: retry, never die
            _publish(results, None, {"error": f"{type(e).__name__}: {e}"})
        stop_event.wait(RECONNECT_PERIOD_S)


def contact_to_image(
    contact: np.ndarray, settings: TactileSettings | None = None, flip_x: bool = False, flip_y: bool = False
) -> np.ndarray:
    """``uint8`` RGB (H, W, 3) heatmap of a contact map in [0, 1], upscaled by ``settings.display_scale``."""
    settings = settings or TactileSettings()
    scaled = np.clip(contact * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(scaled, settings.colormap_id), cv2.COLOR_BGR2RGB)
    if flip_x:
        rgb = rgb[:, ::-1]
    if flip_y:
        rgb = rgb[::-1, :]
    height, width = rgb.shape[:2]
    return cv2.resize(
        np.ascontiguousarray(rgb),
        (width * settings.display_scale, height * settings.display_scale),
        interpolation=cv2.INTER_NEAREST,
    )


def _no_data_image(display_scale: int) -> np.ndarray:
    """``uint8`` RGB image with "NO TACTILE" written in it, same size as the heatmaps."""
    height, width = ROWS * display_scale, COLS * display_scale
    text, font, thickness = "NO TACTILE", cv2.FONT_HERSHEY_SIMPLEX, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = 0.8 * width / text_w
    text_w, text_h = int(text_w * scale), int(text_h * scale)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(image, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


class FlexiTacStream:
    """Lerobot side of the reader process: last tactile heatmap of the FlexiTac sensor."""

    def __init__(self, config: FlexiTacDisplayConfig):
        """Load the processing settings and prepare the reader process, started by :meth:`start`.

        A missing or invalid settings file is reported and the defaults (original FlexiTac processing) are used.
        """
        self.config = config
        config_path = config.config or str(DEFAULT_CONFIG_PATH)
        try:
            self.settings = load_tactile_settings(config_path)
            logger.info("FlexiTac display: settings %s: %s", config_path, self.settings)
        except Exception as e:  # noqa: BLE001 - yaml, OSError, ValueError: the robot must start anyway
            self.settings = TactileSettings()
            logger.error("FlexiTac display: settings %s: %s, using the defaults", config_path, e)
        context = multiprocessing.get_context("spawn")  # no fork of the lerobot process (threads, cuda)
        self._results = context.Queue(maxsize=RESULT_QUEUE_SIZE)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_reader_main,
            args=(config, self.settings, self._results, self._stop_event),
            name="flexitac-reader",
            daemon=True,
        )
        self._reported: dict[str, object] = {}
        self._last_data_time = -float("inf")
        self._showing_no_data = False
        # last raw frame for the dataset (lerobot.utils.sensor_hub): (ROWS, COLS) uint8 as sent by the board (before
        # baseline, threshold and normalization, NOT flipped), and its reception time on the time.monotonic() clock
        self.last_sample: np.ndarray | None = None
        self.last_sample_time: float | None = None

    def start(self) -> None:
        """Start the reader process."""
        self._process.start()
        logger.info("FlexiTac display: reading %s at %d baud", self.config.port, self.config.baud)

    def stop(self) -> None:
        """Stop the reader process, killed if it does not exit within STOP_TIMEOUT_S."""
        self._stop_event.set()
        self._process.join(timeout=STOP_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()

    def poll(self) -> np.ndarray | None:
        """Image to show in the tactile panel, None if the panel does not change (the viewer keeps the last image).

        The heatmap of the last frame received since the previous call, or the "NO TACTILE" image once when no
        frame arrived for NO_DATA_TIMEOUT_S (immediately at the first call).
        """
        contact = self._poll_contact()
        now = time.monotonic()
        if contact is not None:
            if self._showing_no_data:
                logger.info("FlexiTac display: tactile frames received")
            self._showing_no_data = False
            self._last_data_time = now
            return contact_to_image(contact, self.settings, self.config.flip_x, self.config.flip_y)
        if not self._showing_no_data and now - self._last_data_time > NO_DATA_TIMEOUT_S:
            self._showing_no_data = True
            logger.warning("FlexiTac display: no frames from %s, showing NO TACTILE", self.config.port)
            if not FLEXITAC_FIRMWARE.exists() and not self._reported.get("setup_hint"):
                self._reported["setup_hint"] = True
                logger.warning("FlexiTac display: FlexiTac repo not found, %s", FLEXITAC_SETUP_HINT)
            return _no_data_image(self.settings.display_scale)
        return None

    def _poll_contact(self) -> np.ndarray | None:
        """Contact map of the last frame received since the previous call, None if there is no new one."""
        exitcode = self._process.exitcode
        if exitcode not in (None, 0) and self._reported.get("exitcode") != exitcode:
            self._reported["exitcode"] = exitcode
            logger.error("FlexiTac display: reader process stopped (exit code %s)", exitcode)
        latest = None
        while True:
            try:
                item, info = self._results.get_nowait()
            except queue.Empty:
                break
            self._report(info)
            if item is not None:
                raw, contact, received = item
                self.last_sample, self.last_sample_time = raw, received
                if contact is not None:
                    latest = contact
        return latest

    def _report(self, info: dict) -> None:
        """Log the errors and the state of the reader once, when they change."""
        for key, level in (("error", logging.ERROR), ("state", logging.INFO), ("baseline", logging.INFO)):
            value = info.get(key)
            if value and value != self._reported.get(key):
                logger.log(level, "FlexiTac display: %s = %s", key, value)
                self._reported[key] = value
        if info.get("state"):
            self._reported.pop("error", None)  # connected again: report the next error


def make_tactile_display_config(cfg) -> FlexiTacDisplayConfig | None:
    """FlexiTac display config from the ``display_tactile_*`` fields of a lerobot script config, None if no port."""
    if not getattr(cfg, "display_tactile_port", None):
        return None
    return FlexiTacDisplayConfig(
        port=cfg.display_tactile_port,
        baud=cfg.display_tactile_baud,
        config=getattr(cfg, "display_tactile_config", None),
        flip_x=cfg.display_tactile_flip_x,
        flip_y=cfg.display_tactile_flip_y,
    )


if __name__ == "__main__":
    # Standalone test of the FlexiTac in the lerobot viewer, without robot nor cameras:
    #   uv run python -m lerobot.tactile_sensor.flexitac_display --display_tactile_port=/dev/ttyUSB0
    # The package imports are used (not the classes of __main__) to run exactly the lerobot code path.
    import argparse

    from lerobot.tactile_sensor.flexitac_display import make_tactile_display_config as _make_config
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import VISUALIZATION_MODES, display_sensors_only

    parser = argparse.ArgumentParser(
        description="Show only the FlexiTac heatmap in the lerobot viewer (Ctrl+C to stop)."
    )
    parser.add_argument(
        "--display_tactile_port", required=True, help="serial port of the board, e.g. /dev/ttyUSB0"
    )
    parser.add_argument("--display_tactile_baud", type=int, default=2_000_000)
    parser.add_argument(
        "--display_tactile_config",
        default=None,
        help="processing settings yaml (default: flexitac_config.yaml)",
    )
    parser.add_argument("--display_tactile_flip_x", action="store_true")
    parser.add_argument("--display_tactile_flip_y", action="store_true")
    parser.add_argument("--display_mode", default="rerun", choices=VISUALIZATION_MODES)
    parser.add_argument("--display_ip", default=None, help="remote Rerun server / Foxglove bind address")
    parser.add_argument("--display_port", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="rate of the viewer updates")
    cli = parser.parse_args()
    init_logging()
    display_sensors_only(
        cli.display_mode, ip=cli.display_ip, port=cli.display_port, tactile=_make_config(cli), fps=cli.fps
    )
