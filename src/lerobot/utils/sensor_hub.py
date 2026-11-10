"""Extra sensors (VL53L9 ToF, FlexiTac tactile) shared by the viewer and the dataset recording.

The sensors are not robot cameras: they are not in ``robot.get_observation()`` and the policies never see them.
They are shown in the viewer (:mod:`lerobot.utils.visualization_utils`) and, with ``lerobot-record``, saved in the
dataset as raw numeric arrays (parquet, lossless, no video) under the ``sensors.`` prefix:

==========================  =======  =========  ====================================================================
feature                     dtype    shape      content
==========================  =======  =========  ====================================================================
``sensors.tof_depth_mm``    uint16   (42, 54)   filtered depth of the L9 pipeline in mm, 0 = invalid pixel, sensor
                                                orientation (the display flips are not applied)
``sensors.tof_age_s``       float32  (1,)       age of that depth when the row is recorded (centre of the exposure
                                                once the board clock is synced), -1 = no depth
``sensors.tactile_raw``     uint8    (16, 32)   raw FlexiTac frame as sent by the board (before baseline, threshold
                                                and normalization: reprocess it as needed)
``sensors.tactile_age_s``   float32  (1,)       age of that frame when the row is recorded, -1 = no frame
==========================  =======  =========  ====================================================================

The policies only take as inputs the ``observation.*`` features and the image / video ones
(:func:`lerobot.utils.feature_utils.dataset_to_policy_features`): the ``sensors.*`` features are kept in the dataset
for later work and ignored by the training. To use them, convert them to ``observation.*`` features later.

Each row holds the last sample received from each sensor (the sensors run at their own rate). A sensor without a
sample for more than STALE_AFTER_S (not connected, unplugged, baseline of the tactile in progress for the display
only: the raw tactile frames are recorded during it) is recorded as zeros with an age of -1.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lerobot.tactile_sensor.flexitac_display import FlexiTacDisplayConfig
    from lerobot.ToF.l9_display import L9DisplayConfig

logger = logging.getLogger(__name__)

SENSORS_PREFIX = "sensors"
STALE_AFTER_S = 1.0  # a sample older than this is recorded as missing (zeros, age -1)
MISSING_AGE_S = -1.0

TOF_DEPTH_KEY = f"{SENSORS_PREFIX}.tof_depth_mm"
TOF_AGE_KEY = f"{SENSORS_PREFIX}.tof_age_s"
TACTILE_RAW_KEY = f"{SENSORS_PREFIX}.tactile_raw"
TACTILE_AGE_KEY = f"{SENSORS_PREFIX}.tactile_age_s"


class SensorHub:
    """Display-only / recorded extra sensors: one reader process per sensor, polled once per control loop cycle."""

    def __init__(
        self, tof: "L9DisplayConfig | None" = None, tactile: "FlexiTacDisplayConfig | None" = None
    ) -> None:
        """Prepare the streams of the given sensors (None = not used), started by :meth:`start`."""
        # (display key, data key, age key, stream, shape, dtype)
        self._sensors: list[tuple[str, str, str, Any, tuple[int, int], type]] = []
        if tof is not None:
            from lerobot.ToF.l9_display import SENSOR_SHAPE, TOF_DISPLAY_KEY, L9DepthStream

            tof_stream: L9DepthStream = L9DepthStream(tof)
            self._sensors.append(
                (TOF_DISPLAY_KEY, TOF_DEPTH_KEY, TOF_AGE_KEY, tof_stream, SENSOR_SHAPE, np.uint16)
            )
        if tactile is not None:
            from lerobot.tactile_sensor.flexitac_display import (
                COLS,
                ROWS,
                TACTILE_DISPLAY_KEY,
                FlexiTacStream,
            )

            tactile_stream: FlexiTacStream = FlexiTacStream(tactile)
            self._sensors.append(
                (
                    TACTILE_DISPLAY_KEY,
                    TACTILE_RAW_KEY,
                    TACTILE_AGE_KEY,
                    tactile_stream,
                    (ROWS, COLS),
                    np.uint8,
                )
            )
        self._display_images: dict[str, np.ndarray] = {}
        self._missing: dict[str, bool] = {}  # data key -> missing at the last recorded row (logged on change)
        self._started = False

    def __bool__(self) -> bool:
        return bool(self._sensors)

    def start(self) -> None:
        """Start the reader processes."""
        if not self._started:
            for *_, stream, _shape, _dtype in self._sensors:
                stream.start()
            self._started = True

    def stop(self) -> None:
        """Stop the reader processes."""
        if self._started:
            for *_, stream, _shape, _dtype in self._sensors:
                stream.stop()
            self._started = False

    def poll(self) -> None:
        """Read the new samples of all the sensors. Call it once per control loop cycle (cheap, never blocks)."""
        for display_key, _data, _age, stream, _shape, _dtype in self._sensors:
            image = stream.poll()
            if image is not None:
                self._display_images[display_key] = image

    def take_display_images(self) -> dict[str, np.ndarray]:
        """Images changed since the previous call, for the viewer (the viewer keeps the last image of a panel)."""
        images, self._display_images = self._display_images, {}
        return images

    def dataset_features(self) -> dict[str, dict]:
        """Features of the recorded sensors, to add to the dataset features at creation."""
        features: dict[str, dict[str, Any]] = {}
        for _display, data_key, age_key, _stream, shape, dtype in self._sensors:
            features[data_key] = {"dtype": np.dtype(dtype).name, "shape": shape, "names": ["row", "col"]}
            features[age_key] = {"dtype": "float32", "shape": (1,), "names": None}
        return features

    def dataset_frame(self) -> dict[str, np.ndarray]:
        """Values of the recorded sensors for the current row: last sample and its age, or zeros and age -1."""
        now = time.monotonic()
        frame: dict[str, np.ndarray] = {}
        for _display, data_key, age_key, stream, shape, dtype in self._sensors:
            sample, sample_time = stream.last_sample, stream.last_sample_time
            missing = sample is None or sample_time is None or now - sample_time > STALE_AFTER_S
            if not missing and sample.shape != shape:
                if not self._missing.get(f"{data_key}.shape"):
                    logger.error(
                        "%s: sample of shape %s, expected %s: recorded as missing",
                        data_key,
                        sample.shape,
                        shape,
                    )
                self._missing[f"{data_key}.shape"] = True
                missing = True
            if missing != self._missing.get(data_key, False):
                log = logger.warning if missing else logger.info
                log(
                    "%s: %s",
                    data_key,
                    "no sample, recorded as zeros (age -1)" if missing else "recording samples",
                )
            self._missing[data_key] = missing
            if missing:
                frame[data_key] = np.zeros(shape, dtype=dtype)
                frame[age_key] = np.array([MISSING_AGE_S], dtype=np.float32)
            else:
                frame[data_key] = np.ascontiguousarray(sample, dtype=dtype)
                frame[age_key] = np.array([max(0.0, now - sample_time)], dtype=np.float32)
        return frame
