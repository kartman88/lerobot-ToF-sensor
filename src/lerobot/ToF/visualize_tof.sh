#!/usr/bin/env bash
# Shows only the VL53L9 ToF depth in the lerobot viewer (no arms, no cameras), after giving access to its serial port.
#   ./visualize_tof.sh                       # ST-LINK port found automatically
#   TOF_PORT=/dev/ttyACM1 ./visualize_tof.sh --display_tof_flip_x --display_mode=foxglove
# Extra arguments go to `python -m lerobot.ToF.l9_display` (see --help). Ctrl+C to stop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

first_match() {  # first existing path among the patterns, else the last argument
    local p
    for p in "$@"; do
        [[ -e "$p" ]] && { echo "$p"; return; }
    done
    echo "${@: -1}"
}

grant_access() {  # read/write access to a serial port for the current user (lost when the board is unplugged)
    local port="$1" dev
    if [[ ! -e "$port" ]]; then
        echo "WARNING: $port not found, is the board connected? The viewer shows the placeholder until it appears." >&2
        return
    fi
    dev="$(readlink -f "$port")"
    if [[ -r "$dev" && -w "$dev" ]]; then
        echo "Access to $dev ok"
    else
        echo "Giving access to $dev (sudo chmod 666)"
        sudo chmod 666 "$dev"
    fi
}

TOF_PORT="${TOF_PORT:-$(first_match /dev/serial/by-id/usb-STMicroelectronics_STLINK* /dev/ttyACM0)}"
grant_access "$TOF_PORT"

cd "$REPO_DIR"
exec uv run python -m lerobot.ToF.l9_display --display_tof_port="$TOF_PORT" "$@"
