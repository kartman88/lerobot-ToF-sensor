# Helpers shared by full_teleop_run.sh and full_record.sh (sourced, not run): checks and access rights of the serial
# ports of the arms and of the extra sensors (VL53L9 ToF, FlexiTac tactile).
#
# The caller sets FOLLOWER_PORT, LEADER_PORT, TOF_PORT, TOF_BAUD, TOF_CONFIG, TACTILE_PORT, TACTILE_BAUD,
# TACTILE_CONFIG and CAMERAS, then calls setup_serial_ports, which fills SENSOR_ARGS with the lerobot arguments of
# the sensors and of the cameras.

# shellcheck shell=bash

TOF_BY_ID_PATTERN="/dev/serial/by-id/usb-STMicroelectronics_STLINK*"
TACTILE_BY_ID_PATTERN="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_*"

list_ports() {
    echo "Serial ports connected:" >&2
    local p found=0
    for p in /dev/serial/by-id/*; do
        [[ -e "$p" ]] && { echo "  $p -> $(readlink -f "$p")" >&2; found=1; }
    done
    [[ "$found" == 1 ]] || echo "  none" >&2
}

grant_access() {  # read/write access to a serial port for the current user (lost when the board is unplugged)
    local dev
    dev="$(readlink -f "$1")"
    if [[ -r "$dev" && -w "$dev" ]]; then
        echo "Access to $1 ok"
    else
        echo "Giving access to $1 ($dev, sudo chmod 666)"
        sudo chmod 666 "$dev"
    fi
}

require_port() {  # $1 = variable name, $2 = port: the arms are mandatory
    if [[ -z "$2" ]]; then
        echo "ERROR: $1 is not set: write it in the settings of $0" >&2
        list_ports
        exit 1
    fi
    if [[ ! -e "$2" ]]; then
        echo "ERROR: $1=$2 not found, is the arm connected and powered?" >&2
        list_ports
        exit 1
    fi
}

find_sensor() {  # $1 = given port ("" = automatic, none = disabled), $2 = by-id pattern: prints the port or nothing
    local p
    if [[ "$1" == "none" ]]; then
        return 0
    elif [[ -n "$1" ]]; then
        echo "$1"
        return 0
    fi
    for p in $2; do  # no fallback on /dev/ttyACM0: it could be the port of an arm
        [[ -e "$p" ]] && { echo "$p"; return 0; }
    done
    return 0  # not found: the sensor is not used
}

setup_serial_ports() {  # checks + sudo chmod of all the ports, SENSOR_ARGS = lerobot arguments of sensors and cameras
    require_port FOLLOWER_PORT "$FOLLOWER_PORT"
    require_port LEADER_PORT "$LEADER_PORT"
    local tof tactile entry name port dev
    tof="$(find_sensor "$TOF_PORT" "$TOF_BY_ID_PATTERN")"
    tactile="$(find_sensor "$TACTILE_PORT" "$TACTILE_BY_ID_PATTERN")"

    # two devices on the same port would fight for it (e.g. a sensor port given that is actually an arm)
    declare -A owner=()
    for entry in "follower:$FOLLOWER_PORT" "leader:$LEADER_PORT" "tof:$tof" "tactile:$tactile"; do
        name="${entry%%:*}" port="${entry#*:}"
        [[ -z "$port" || ! -e "$port" ]] && continue
        dev="$(readlink -f "$port")"
        if [[ -n "${owner[$dev]:-}" ]]; then
            echo "ERROR: $name and ${owner[$dev]} use the same port $dev" >&2
            exit 1
        fi
        owner[$dev]="$name"
    done

    echo "== Serial ports"
    grant_access "$FOLLOWER_PORT"
    grant_access "$LEADER_PORT"
    SENSOR_ARGS=()
    if [[ -n "$tof" ]]; then
        if [[ -e "$tof" ]]; then
            grant_access "$tof"
        else
            echo "WARNING: ToF $tof not found yet, NO DEPTH (and zeros in a dataset) until it appears" >&2
        fi
        SENSOR_ARGS+=(--display_tof_port="$tof" --display_tof_baud="$TOF_BAUD")
        [[ -n "$TOF_CONFIG" ]] && SENSOR_ARGS+=(--display_tof_config="$TOF_CONFIG")
    else
        echo "ToF: not found / disabled, not used"
    fi
    if [[ -n "$tactile" ]]; then
        if [[ -e "$tactile" ]]; then
            grant_access "$tactile"
        else
            echo "WARNING: tactile $tactile not found yet, NO TACTILE (and zeros in a dataset) until it appears" >&2
        fi
        SENSOR_ARGS+=(--display_tactile_port="$tactile" --display_tactile_baud="$TACTILE_BAUD")
        [[ -n "$TACTILE_CONFIG" ]] && SENSOR_ARGS+=(--display_tactile_config="$TACTILE_CONFIG")
    else
        echo "Tactile: not found / disabled, not used"
    fi
    [[ -n "$CAMERAS" ]] && SENSOR_ARGS+=(--robot.cameras="$CAMERAS")
    return 0
}

run_lerobot() {  # runs "$@" from the repo root, or only prints it with DRY_RUN=1
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "$@"
        echo
        exit 0
    fi
    cd "$REPO_DIR"
    exec "$@"
}
