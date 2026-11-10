# FlexiTac tactile sensor

Integration of the [FlexiTac](https://github.com/FlexiTac/FlexiTac) tactile sensor (16 x 32 taxels) in lerobot:
live heatmap in the viewer of `lerobot-teleoperate` / `lerobot-record`, and raw frames saved in the datasets as
`sensors.tactile_*` (ignored by the training, see `lerobot/utils/sensor_hub.py`).

| File | Content |
|---|---|
| `flexitac_display.py` | serial reader (separate process), processing of the heatmap, standalone test (`python -m`) |
| `flexitac_config.yaml` | processing settings of the heatmap (threshold, normalization, filter, colormap), commented |
| `visualize_tactile.sh` | sudo access to the port + tactile sensor alone in the viewer |
| `visualize_tof_tactile.sh` | same with the ToF and the tactile sensors together (identical copy in `lerobot/ToF/`) |
| `FlexiTac/` | official FlexiTac repo, **git submodule**: Arduino firmware and original viewer |

## 1. Get the official FlexiTac repo

`FlexiTac/` is a git submodule: a plain `git clone` of lerobot leaves it empty. Get it with:

```bash
git clone --recurse-submodules <lerobot repo url>     # new clone
git submodule update --init src/lerobot/tactile_sensor/FlexiTac   # existing clone
```

To always fetch the submodules on `git pull` / `git checkout`: `git config submodule.recurse true`.

The lerobot code does not import anything from this repo: it is needed to flash the firmware and to run the original
viewer. Without it, the "no frames" warning of the lerobot viewer prints these instructions.

FlexiTac is licensed under CC BY-NC 4.0 (non commercial), see `FlexiTac/readme.md`.

## 2. Flash the firmware

Open `FlexiTac/32_16_fast.ino` in the Arduino IDE, select the board (ATmega328P: Uno / Nano) and its port, upload.
The board then streams frames of `0xAA 0x55` + 512 bytes at 2 Mbaud.

## 3. Serial port access

The port belongs to the `uucp` group on Arch based systems. Once (then log out and in again):

```bash
sudo usermod -aG uucp $USER
```

or each time the board is plugged in: `sudo chmod 666 /dev/ttyUSB0` (the `.sh` scripts do it when needed).

## 4. Test and use

```bash
src/lerobot/tactile_sensor/visualize_tactile.sh          # tactile sensor alone in the viewer
src/lerobot/tactile_sensor/visualize_tof_tactile.sh      # ToF + tactile
src/lerobot/scripts/full_teleop_run.sh                   # teleoperation with the sensors in the viewer
src/lerobot/scripts/full_record.sh                       # dataset recording with the sensors
```

Do not touch the sensor during the first frames: the baseline is taken at each (re)connection. To test with the
original FlexiTac viewer: `pip install numpy pyserial opencv-python scipy` then `python3 FlexiTac/fast_32_16.py`.
