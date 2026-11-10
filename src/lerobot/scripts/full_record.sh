#!/usr/bin/env bash
# Full dataset recording: SO-101 leader -> follower with the cameras, and the extra sensors (VL53L9 ToF and FlexiTac
# tactile) shown in the viewer and saved in the dataset as sensors.* (raw data, ignored by the training: see
# lerobot/utils/sensor_hub.py). Gives access to all the serial ports first (sudo chmod 666 when needed).
#
#   src/lerobot/scripts/full_record.sh
#   DRY_RUN=1 src/lerobot/scripts/full_record.sh        # print the lerobot-record command without running it
#
# All the parameters are in the settings below: edit them here. Extra arguments go to lerobot-record (see
# `uv run lerobot-record --help`).
#
# Keys during the recording: right arrow = end the episode / the reset early, left arrow = re-record the episode,
# Esc = stop the recording (the episodes already saved are kept).
set -euo pipefail

# ================================================================================================================
# Settings
# ================================================================================================================
# --- Arms. Prefer the stable /dev/serial/by-id/... paths (ls -l /dev/serial/by-id/): the /dev/ttyACM* numbers change
# with the plug order. To find which port is which arm: uv run lerobot-find-port
ROBOT_TYPE="so101_follower"
FOLLOWER_PORT=""
FOLLOWER_ID="my_follower"            # same id as in lerobot-calibrate
TELEOP_TYPE="so101_leader"
LEADER_PORT=""
LEADER_ID="my_leader"                # same id as in lerobot-calibrate

# --- Cameras of the robot: these are the images used by the training. Find the indexes with:
# uv run lerobot-find-cameras opencv
CAMERAS="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"

# --- Dataset
# '<hf_username>/<name>'. A date-time tag is appended at each new recording (e.g. local/so101_sensors_20261110_1830)
# unless NO_STAMP=true. To add episodes to an existing dataset: RESUME=true with its full name (tag included).
REPO_ID="local/so101_sensors"
NO_STAMP=false
RESUME=false
ROOT=""                              # empty: ~/.cache/huggingface/lerobot/<REPO_ID>
SINGLE_TASK="Pick the object and place it in the box"   # description of the task, saved with every episode
NUM_EPISODES=50
EPISODE_TIME_S=60                    # duration of each episode (right arrow ends it earlier)
RESET_TIME_S=30                      # time to reset the scene between two episodes (not recorded)
FPS=30                               # recording rate, must not exceed the camera fps
VIDEO=true                           # cameras saved as mp4 videos (false: png images)
PUSH_TO_HUB=false                    # upload to the Hugging Face hub at the end (needs huggingface-cli login)
PRIVATE=true                         # with PUSH_TO_HUB=true: private dataset on the hub
PLAY_SOUNDS=true                     # spoken events (start / end of episode, reset)

# --- Extra sensors: empty port = found automatically from the USB id, none = disabled
RECORD_EXTRA_SENSORS=true            # save them in the dataset (false: only shown in the viewer)
TOF_PORT=""
TOF_BAUD=12500000                    # must match CONF_RAW_BAUDRATE of the L9 firmware
TOF_CONFIG=""                        # empty: lerobot/ToF/L9_sensor/l9_transform_config.yaml
TACTILE_PORT=""
TACTILE_BAUD=2000000                 # must match BAUD_RATE of FlexiTac/32_16_fast.ino
TACTILE_CONFIG=""                    # empty: lerobot/tactile_sensor/flexitac_config.yaml (display only: raw data saved)

# --- Viewer
DISPLAY_MODE="rerun"                 # rerun or foxglove
# ================================================================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=serial_ports_common.sh
source "$SCRIPT_DIR/serial_ports_common.sh"

setup_serial_ports
[[ -n "$CAMERAS" ]] || echo "WARNING: no camera (CAMERAS empty): the dataset has no images to train on" >&2
DATASET_ARGS=()
[[ -n "$ROOT" ]] && DATASET_ARGS+=(--dataset.root="$ROOT")

echo "== Recording $NUM_EPISODES episodes of ${EPISODE_TIME_S}s in $REPO_ID (extra sensors recorded: $RECORD_EXTRA_SENSORS)"
cat <<EOF

  Keys during the recording (on Wayland / Hyprland they are read from THIS terminal: keep the focus on it, not on
  the viewer window):
    Right arrow   end the current episode now and save it (during the reset: end the reset now)
    Left arrow    discard the current episode and record it again (after the reset)
    Esc           stop the recording: the episodes already saved are kept, the dataset is finalized
    Ctrl+C        emergency stop: the current episode is lost, the saved ones are kept

  Each episode lasts ${EPISODE_TIME_S}s, then ${RESET_TIME_S}s of reset (not recorded) to put the scene back.
  Do not touch the tactile sensor at the start: its baseline is taken in the first frames.

EOF
run_lerobot uv run lerobot-record \
    --robot.type="$ROBOT_TYPE" --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
    --teleop.type="$TELEOP_TYPE" --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
    --dataset.repo_id="$REPO_ID" --dataset.no_stamp="$NO_STAMP" --resume="$RESUME" \
    --dataset.single_task="$SINGLE_TASK" --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" --dataset.reset_time_s="$RESET_TIME_S" --dataset.fps="$FPS" \
    --dataset.video="$VIDEO" --dataset.push_to_hub="$PUSH_TO_HUB" --dataset.private="$PRIVATE" \
    --play_sounds="$PLAY_SOUNDS" --record_extra_sensors="$RECORD_EXTRA_SENSORS" \
    --display_data=true --display_mode="$DISPLAY_MODE" \
    "${DATASET_ARGS[@]}" "${SENSOR_ARGS[@]}" "$@"
