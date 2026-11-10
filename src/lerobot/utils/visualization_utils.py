# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-agnostic visualization dispatch.

Selects a visualization backend at runtime via a display-mode string (e.g. a ``--display_mode`` CLI
flag) so callers never branch on the backend. The concrete implementations live in
:mod:`lerobot.utils.rerun_visualization` and :mod:`lerobot.utils.foxglove_visualization`; importing
this module does not import ``rerun`` or ``foxglove`` (each backend imports its SDK lazily behind a
``require_package`` guard).
"""

import signal
import time
from typing import TYPE_CHECKING

from lerobot.lerobot_types import EnvTransition, RobotAction, RobotObservation, TransitionKey

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

if TYPE_CHECKING:
    from lerobot.tactile_sensor.flexitac_display import FlexiTacDisplayConfig
    from lerobot.ToF.l9_display import L9DisplayConfig

    from .sensor_hub import SensorHub

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
VISUALIZATION_MODES = ("rerun", "foxglove")

# Extra sensors (VL53L9 ToF, FlexiTac tactile) shown in the viewer: their images are added to the observation sent
# to the viewer only, not to the processors or the policies (lerobot-record saves their raw data in the dataset
# through the same hub, see lerobot.utils.sensor_hub). Either created by init_visualization from the tof / tactile
# configs (owned: stopped by shutdown_visualization) or given by the caller (not owned: the caller stops it).
_SENSORS: "SensorHub | None" = None
_SENSORS_OWNED = False


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    tof: "L9DisplayConfig | None" = None,
    tactile: "FlexiTacDisplayConfig | None" = None,
    sensors: "SensorHub | None" = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.

    ``tof`` starts the display-only VL53L9 ToF depth stream (see :mod:`lerobot.ToF.l9_display`).
    ``tactile`` starts the display-only FlexiTac tactile stream (see
    :mod:`lerobot.tactile_sensor.flexitac_display`).
    ``sensors`` is an already started :class:`~lerobot.utils.sensor_hub.SensorHub` to show instead (e.g. the one
    lerobot-record also records): ``tof`` and ``tactile`` are then ignored.
    """
    global _SENSORS, _SENSORS_OWNED

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")

    if _SENSORS is None:
        if sensors is not None:
            _SENSORS, _SENSORS_OWNED = sensors, False
        elif tof is not None or tactile is not None:
            from .sensor_hub import SensorHub

            _SENSORS, _SENSORS_OWNED = SensorHub(tof=tof, tactile=tactile), True
            _SENSORS.start()


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    policy_step: EnvTransition | None = None,
) -> None:
    """Logs observation/action data and the policy's prediction to the backend selected by ``display_mode``.

    ``policy_step`` is the transition a policy returned with its action; its prediction (see
    ``PreTrainedPolicy.select_action``) is shown.
    """

    prediction = policy_step.get(TransitionKey.PREDICTION) if policy_step is not None else None

    if _SENSORS is not None:
        _SENSORS.poll()  # no new sample if the caller already polled the hub in this cycle (lerobot-record)
        images = _SENSORS.take_display_images()
        if images:
            # shallow copy: the caller's observation is left untouched
            observation = {**(observation or {}), **images}

    if display_mode == "rerun":
        log_rerun_data(
            observation=observation,
            action=action,
            compress_images=compress_images,
            prediction=prediction,
        )
    elif display_mode == "foxglove":
        log_foxglove_data(
            observation=observation,
            action=action,
            compress_images=compress_images,
            prediction=prediction,
        )
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""
    global _SENSORS, _SENSORS_OWNED

    if _SENSORS is not None:
        if _SENSORS_OWNED:
            _SENSORS.stop()
        _SENSORS, _SENSORS_OWNED = None, False

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def display_sensors_only(
    display_mode: str,
    *,
    ip: str | None = None,
    port: int | None = None,
    tof: "L9DisplayConfig | None" = None,
    tactile: "FlexiTacDisplayConfig | None" = None,
    fps: float = 30.0,
) -> None:
    """Shows only the display-only sensors (no robot, no cameras) until Ctrl+C, to test them standalone."""
    init_visualization(display_mode, session_name="sensor_test", ip=ip, port=port, tof=tof, tactile=tactile)
    try:
        while True:
            start = time.perf_counter()
            log_visualization_data(display_mode, observation={})
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(
            signal.SIGINT, signal.SIG_IGN
        )  # a second Ctrl+C must not interrupt the readers' shutdown
        shutdown_visualization(display_mode)
