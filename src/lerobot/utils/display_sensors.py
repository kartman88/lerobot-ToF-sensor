"""Standalone test of the display-only sensors (L9 ToF and FlexiTac) together in the lerobot viewer.

No robot and no cameras: only the sensors given on the command line are shown, through the same code path as
``lerobot-teleoperate`` / ``lerobot-record`` (:func:`lerobot.utils.visualization_utils.display_sensors_only`).

Every import is inside the ``__main__`` guard: the reader processes are started with ``spawn``, which re-imports
this module in each of them, so it must stay cheap to import.
"""

if __name__ == "__main__":
    # Standalone test of the display-only sensors together in the lerobot viewer, without robot nor cameras:
    #   uv run python -m lerobot.utils.display_sensors --display_tof_port=... --display_tactile_port=...
    import argparse

    from lerobot.tactile_sensor.flexitac_display import make_tactile_display_config
    from lerobot.ToF.l9_display import make_tof_display_config
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import VISUALIZATION_MODES, display_sensors_only

    parser = argparse.ArgumentParser(
        description="Show only the ToF and tactile sensors in the lerobot viewer."
    )
    parser.add_argument("--display_tof_port", default=None, help="serial port of the L9 board, None: no ToF")
    parser.add_argument("--display_tof_baud", type=int, default=12_500_000)
    parser.add_argument("--display_tof_calib", default=None)
    parser.add_argument(
        "--display_tof_config",
        default=None,
        help="pipeline controls yaml (default: L9_sensor/l9_transform_config.yaml)",
    )
    parser.add_argument("--display_tof_flip_x", action="store_true")
    parser.add_argument("--display_tof_flip_y", action="store_true")
    parser.add_argument(
        "--display_tactile_port", default=None, help="serial port of the FlexiTac, None: no tactile"
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
    if not cli.display_tof_port and not cli.display_tactile_port:
        parser.error("give --display_tof_port and/or --display_tactile_port")
    init_logging()
    display_sensors_only(
        cli.display_mode,
        ip=cli.display_ip,
        port=cli.display_port,
        tof=make_tof_display_config(cli),
        tactile=make_tactile_display_config(cli),
        fps=cli.fps,
    )
