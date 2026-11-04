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

from typing import TYPE_CHECKING

from lerobot.lerobot_types import EnvTransition, RobotAction, RobotObservation, TransitionKey

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

if TYPE_CHECKING:
    from lerobot.ToF.l9_display import L9DepthStream, L9DisplayConfig

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
VISUALIZATION_MODES = ("rerun", "foxglove")

# Display-only VL53L9 ToF depth stream, started by init_visualization when a ToF config is given. Its frames are
# added to the observation sent to the viewer only: not to the dataset, the processors or the policies.
_TOF_STREAM: "L9DepthStream | None" = None


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    tof: "L9DisplayConfig | None" = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.

    ``tof`` starts the display-only VL53L9 ToF depth stream (see :mod:`lerobot.ToF.l9_display`).
    """
    global _TOF_STREAM

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")

    if tof is not None and _TOF_STREAM is None:
        from lerobot.ToF.l9_display import L9DepthStream

        _TOF_STREAM = L9DepthStream(tof)
        _TOF_STREAM.start()


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

    if _TOF_STREAM is not None:
        depth = _TOF_STREAM.poll()
        if depth is not None:
            # shallow copy: the caller's observation is left untouched
            from lerobot.ToF.l9_display import TOF_DISPLAY_KEY

            observation = {**(observation or {}), TOF_DISPLAY_KEY: depth}

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
    global _TOF_STREAM

    if _TOF_STREAM is not None:
        _TOF_STREAM.stop()
        _TOF_STREAM = None

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
