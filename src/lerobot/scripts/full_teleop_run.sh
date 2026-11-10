#!/usr/bin/env bash
# Full teleoperation: SO-101 leader -> follower, with the display-only sensors (VL53L9 ToF and FlexiTac tactile) in
# the viewer next to the cameras. Gives access to all the serial ports first (sudo chmod 666 when needed).
#
#   src/lerobot/scripts/full_teleop_run.sh
#   FOLLOWER_PORT=/dev/ttyACM1 LEADER_PORT=/dev/ttyACM2 src/lerobot/scripts/full_teleop_run.sh --display_mode=foxglove
#
# Every setting below can be overridden by an environment variable of the same name. Extra arguments go to
# lerobot-teleoperate (see `uv run lerobot-teleoperate --help`). Ctrl+C to stop.
#
# The sensors are optional: each one is enabled only if its board is found (stable /dev/serial/by-id path) or its
# port is given (TOF_PORT / TACTILE_PORT); TOF_PORT=none / TACTILE_PORT=none disables it.
set -euo pipefail

# ----------------------------------------------------------------------------------------------------------------
# Settings: fill the arm ports once. Prefer the stable /dev/serial/by-id/... paths (ls -l /dev/serial/by-id/), the
# /dev/ttyACM* numbers change with the plug order. To find which port is which arm: uv run lerobot-find-port
# ----------------------------------------------------------------------------------------------------------------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-}"
FOLLOWER_ID="${FOLLOWER_ID:-my_follower}"      # same id as in lerobot-calibrate
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
LEADER_PORT="${LEADER_PORT:-}"
LEADER_ID="${LEADER_ID:-my_leader}"            # same id as in lerobot-calibrate
# cameras of the robot, empty for none, e.g. "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
CAMERAS="${CAMERAS:-}"
DISPLAY_MODE="${DISPLAY_MODE:-rerun}"          # rerun or foxglove
# sensors: empty = found automatically from their USB id, none = disabled
TOF_PORT="${TOF_PORT:-}"
TOF_BAUD="${TOF_BAUD:-12500000}"
TOF_CONFIG="${TOF_CONFIG:-}"                   # empty: lerobot/ToF/L9_sensor/l9_transform_config.yaml
TACTILE_PORT="${TACTILE_PORT:-}"
TACTILE_BAUD="${TACTILE_BAUD:-2000000}"
TACTILE_CONFIG="${TACTILE_CONFIG:-}"           # empty: lerobot/tactile_sensor/flexitac_config.yaml
# ----------------------------------------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=serial_ports_common.sh
source "$SCRIPT_DIR/serial_ports_common.sh"

setup_serial_ports

echo "== Teleoperation (Ctrl+C to stop). Do not touch the tactile sensor during the first second (baseline)."
run_lerobot uv run lerobot-teleoperate \
    --robot.type="$ROBOT_TYPE" --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
    --teleop.type="$TELEOP_TYPE" --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
    --display_data=true --display_mode="$DISPLAY_MODE" \
    "${SENSOR_ARGS[@]}" "$@"
