#!/usr/bin/env python3
"""Receiver for the raw stream produced by vl53l9_app.c with CONF_OUTPUT_MODE = OUTPUT_MODE_RAW.

Packet: magic (4 bytes) | payload_size (uint32 LE) | payload
  "L9CB" calibration data, sent once at startup
  "L9RW" raw frame: distance | amplitude | ambient (uint16 LE, res px each) | dss lut (res/2) | status (100 bytes)
  "L9ER" firmware stopped on an error (repeated every second): source line | return code | fsm state | error flags
  "L9TM" loop timing statistics (CONF_RAW_PROFILE): nb frames | avg us, max us for each loop phase
  "L9FT" frame timing, before every L9RW: frame counter | trigger time | INTR time | exposure (board clock, us)
  "L9PO" answer to the "L9PI" clock sync pings sent by the host (see l9_time_sync.py)

Every frame gets its acquisition time on the host clock (centre of the exposure, see l9_time_sync.py) once the board
clock is synced, i.e. after MIN_SAMPLES pings (~1 s).

The --baud value must match CONF_RAW_BAUDRATE: with the ST-LINK VCP the host sets the baudrate
used by the ST-LINK on the uart towards the STM32.

Usage: python3 l9_raw_receiver.py /dev/ttyACM0 [--baud 4000000] [--out capture.bin] [--every-frame]
                                 [--clock realtime|monotonic] [--trigger-latency-us US]
Requires: pip install -r requirements.txt (a venv is provided in Tools/.venv)
"""
import argparse
import struct
import time

import numpy as np
import serial

import l9_time_sync as ts

MAGIC_SIZE = 4
HEADER_SIZE = 8
NO_DATA_WARN_S = 3.0
STATS_PERIOD_S = 1.0
READ_CHUNK = 64 * 1024
MAX_COUNTER_GAP = 10000  # larger frame counter jumps are not counted as lost frames (corrupted status)
# vl53l9 driver return codes (see vl53l9.h)
DRIVER_ERRORS = {0: "NONE", -1: "PLATFORM", -2: "INVALID_PARAM", -3: "INVALID_STATE",
                 -4: "INVALID_OPERATION", -5: "TIMEOUT", -6: "INTERNAL"}
STATUS_SIZE = 100
CALIB_SIZE = 2332
ERROR_SIZE = 10
# loop phases in the order of raw_phase_t (vl53l9_app.c)
PROFILE_PHASES = ["delay", "trigger", "intr", "read_start", "read", "ack", "send", "loop"]
PROFILE_SIZE = 4 + 8 * len(PROFILE_PHASES)
# binning -> (width, height) of the raw frame (see _get_raw_resolution() in vl53l9_utils.c)
RAW_RESOLUTIONS = [(54, 42), (24, 24), (18, 14), (12, 10), (8, 8), (4, 4)]
# frame payload size -> resolution: 3 uint16 frames + dss lut (res / 2) + status
FRAME_SIZES = {6 * w * h + (w * h) // 2 + STATUS_SIZE: (w, h) for w, h in RAW_RESOLUTIONS}
MAGICS = {b"L9CB": {CALIB_SIZE}, b"L9RW": set(FRAME_SIZES), b"L9ER": {ERROR_SIZE}, b"L9TM": {PROFILE_SIZE},
          ts.FRAME_TIME_MAGIC: {ts.FRAME_TIME_SIZE}, ts.PONG_MAGIC: {ts.PONG_SIZE}}


class PacketParser:
    """Rebuild packets from a byte stream received in arbitrary chunks.

    Bytes before a valid header (startup text at another baudrate, uart corruption) are discarded
    and counted in skipped_bytes. A header is valid only if its size matches the magic, so a corrupted
    size cannot make the parser swallow the following packets.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.skipped_bytes = 0

    def feed(self, data):
        """Append received bytes and return the list of complete packets as (magic, payload)."""
        self.buffer += data
        packets = []
        while True:
            start = self._find_magic()
            if start < 0:
                # keep a possible partial magic at the end of the buffer
                keep = MAGIC_SIZE - 1
                if len(self.buffer) > keep:
                    self.skipped_bytes += len(self.buffer) - keep
                    del self.buffer[:-keep]
                break
            if start:
                self.skipped_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < HEADER_SIZE:
                break
            magic = bytes(self.buffer[:MAGIC_SIZE])
            size = struct.unpack_from("<I", self.buffer, MAGIC_SIZE)[0]
            if size not in MAGICS[magic]:
                # false magic or corrupted header: resync from the next byte
                self.skipped_bytes += 1
                del self.buffer[:1]
                continue
            if len(self.buffer) < HEADER_SIZE + size:
                break
            nested = self._find_header(HEADER_SIZE, HEADER_SIZE + size)
            if nested >= 0:
                # this packet was truncated (bytes lost on the uart) and the next one started inside it
                self.skipped_bytes += nested
                del self.buffer[:nested]
                continue
            packets.append((magic, bytes(self.buffer[HEADER_SIZE:HEADER_SIZE + size])))
            del self.buffer[:HEADER_SIZE + size]
        return packets

    def _find_magic(self, start=0, end=None):
        positions = [p for p in (self.buffer.find(m, start, end) for m in MAGICS) if p >= 0]
        return min(positions) if positions else -1

    def _find_header(self, start, end):
        """Position of the first magic followed by a size valid for it in buffer[start:end], -1 if none."""
        while True:
            pos = self._find_magic(start, end)
            if pos < 0 or pos + HEADER_SIZE > len(self.buffer):
                return -1
            size = struct.unpack_from("<I", self.buffer, pos + MAGIC_SIZE)[0]
            if size in MAGICS[bytes(self.buffer[pos:pos + MAGIC_SIZE])]:
                return pos
            start = pos + 1


def print_error(payload):
    line, ret, fsm, flags = struct.unpack_from("<iiBB", payload, 0)
    name = DRIVER_ERRORS.get(ret, "?")
    print(f"[error] firmware stopped at vl53l9_app.c:{line} ret={ret} ({name}) fsm={fsm} error_flags=0x{flags:02x}")


def print_profile(payload):
    nb_frames = struct.unpack_from("<I", payload, 0)[0]
    values = struct.unpack_from(f"<{2 * len(PROFILE_PHASES)}I", payload, 4)
    phases = "  ".join(f"{name} {values[2 * i] / 1000:.1f}/{values[2 * i + 1] / 1000:.1f}"
                       for i, name in enumerate(PROFILE_PHASES))
    print(f"[timing ms avg/max, {nb_frames} frames] {phases}")


def parse_frame(payload):
    w, h = FRAME_SIZES[len(payload)]
    res = w * h
    words = np.frombuffer(payload, dtype="<u2", count=3 * res)
    distance = (words[:res] & 0x7FFF).reshape(h, w)
    flag = (words[:res] >> 15).reshape(h, w)
    amplitude = words[res:2 * res].reshape(h, w)
    ambient = words[2 * res:3 * res].reshape(h, w)
    dss_lut = np.frombuffer(payload, dtype=np.uint8, count=res // 2, offset=6 * res)
    status = payload[-STATUS_SIZE:]
    frame_counter = struct.unpack_from("<I", status, 0)[0]
    return dict(distance=distance, flag=flag, amplitude=amplitude, ambient=ambient,
                dss_lut=dss_lut, status=status, frame_counter=frame_counter)


class Stats:
    """Frame rate, throughput and stream health over the last STATS_PERIOD_S."""

    def __init__(self):
        self.last_counter = None
        self.gaps = 0
        self.skipped_reported = 0
        self._reset(time.monotonic())

    def _reset(self, now):
        self.start = now
        self.frames = 0
        self.rx_bytes = 0

    def on_bytes(self, count):
        self.rx_bytes += count

    def on_frame(self, frame):
        counter = frame["frame_counter"]
        # missing counters: frames lost or corrupted on the uart (resync), a lower counter means a firmware restart
        if self.last_counter is not None and self.last_counter + 1 < counter <= self.last_counter + MAX_COUNTER_GAP:
            self.gaps += counter - self.last_counter - 1
        self.last_counter = counter
        self.frames += 1

    def report(self, now, parser, frame, sync_status=None):
        elapsed = now - self.start
        if elapsed < STATS_PERIOD_S:
            return
        skipped = parser.skipped_bytes - self.skipped_reported
        self.skipped_reported = parser.skipped_bytes
        line = (f"{self.frames / elapsed:5.1f} fps  {self.rx_bytes / elapsed / 1000:6.1f} kB/s  "
                f"lost frames {self.gaps}  skipped bytes {skipped}")
        if frame is not None:
            h, w = frame["distance"].shape
            line += f"  | frame {frame['frame_counter']} {w}x{h} center distance {frame['distance'][h // 2, w // 2]}"
        if sync_status:
            line += f"  | {sync_status}"
        print(line)
        self._reset(now)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--baud", type=int, default=12500000, help="must match CONF_RAW_BAUDRATE")
    parser.add_argument("--out", help="append received packets (header included) to this file")
    parser.add_argument("--every-frame", action="store_true", help="print one line per frame")
    parser.add_argument("--clock", choices=ts.CLOCKS, default="realtime",
                        help="host clock of the acquisition times (realtime: librealsense global time)")
    parser.add_argument("--trigger-latency-us", type=float, default=0.0,
                        help="latency between the trigger command and the start of the exposure")
    args = parser.parse_args()

    out = open(args.out, "ab") if args.out else None
    packet_parser = PacketParser()
    stats = Stats()
    last_frame = None
    last_data = time.monotonic()
    # short timeout: read() returns as soon as some bytes are available (see below)
    with serial.Serial(args.port, args.baud, timeout=0.1) as port:
        port.reset_input_buffer()
        clock = ts.StreamClock(port, args.clock, args.trigger_latency_us)
        while True:
            clock.poll()
            # read everything already buffered by the OS, or wait for at least one byte
            data = port.read(min(max(port.in_waiting, 1), READ_CHUNK))
            t_rx = clock.now_ns()  # reception time of the packets completed by this read (clock sync)
            now = time.monotonic()
            if not data:
                if now - last_data > NO_DATA_WARN_S:
                    print(f"[warn] no data from the board for {NO_DATA_WARN_S:.0f} s")
                    last_data = now
                continue
            last_data = now
            stats.on_bytes(len(data))

            for magic, payload in packet_parser.feed(data):
                clock.on_packet(magic, payload, t_rx)
                if magic == b"L9TM":
                    # diagnostic only, not stored in the capture file
                    print_profile(payload)
                    continue
                if magic == ts.PONG_MAGIC:
                    continue  # clock sync only
                if out:
                    out.write(magic + struct.pack("<I", len(payload)) + payload)
                if magic == b"L9CB":
                    print(f"calibration data received ({len(payload)} bytes)")
                elif magic == b"L9ER":
                    print_error(payload)
                elif magic == b"L9RW":
                    last_frame = clock.stamp(parse_frame(payload))
                    stats.on_frame(last_frame)
                    if args.every_frame:
                        h, w = last_frame["distance"].shape
                        acq = last_frame["acq_time_ns"]
                        acquired = (f"acquired {acq / 1e9:.6f} s, received {(t_rx - acq) / 1e6:.1f} ms later"
                                    if acq is not None else f"acquired ? ({clock.status()})")
                        print(f"frame {last_frame['frame_counter']} {w}x{h} "
                              f"center distance {last_frame['distance'][h // 2, w // 2]}  {acquired}")
            stats.report(now, packet_parser, last_frame, clock.status())


if __name__ == "__main__":
    main()
