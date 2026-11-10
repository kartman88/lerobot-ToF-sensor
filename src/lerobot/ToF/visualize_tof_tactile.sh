#!/usr/bin/env bash
# Shows the VL53L9 ToF depth AND the FlexiTac tactile heatmap together in the lerobot viewer (no arms, no cameras),
# after giving access to their serial ports. Same script in lerobot/ToF/ and lerobot/tactile_sensor/.
#   ./visualize_tof_tactile.sh               # ST-LINK and FTDI ports found automatically
#   TOF_PORT=/dev/ttyACM1 TACTILE_PORT=/dev/ttyUSB1 ./visualize_tof_tactile.sh --display_mode=foxglove
# Extra arguments go to `python -m lerobot.utils.display_sensors` (see --help). Ctrl+C to stop.
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
TACTILE_PORT="${TACTILE_PORT:-$(first_match /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_* /dev/ttyUSB0)}"
grant_access "$TOF_PORT"
grant_access "$TACTILE_PORT"

cd "$REPO_DIR"
exec uv run python -m lerobot.utils.display_sensors \
    --display_tof_port="$TOF_PORT" --display_tactile_port="$TACTILE_PORT" "$@"
