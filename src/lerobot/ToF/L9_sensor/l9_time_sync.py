#!/usr/bin/env python3
"""Map the board timebase on the host clock and stamp the frames of the raw stream with their acquisition time.

The board (vl53l9_app.c, raw mode) reads its times on TIM2, a 32-bit counter at 1 MHz:
  "L9FT" sent before every L9RW: frame counter | trigger time | INTR time | exposure (us)
  "L9PO" answer to a host "L9PI" ping: sequence number | ping reception time | answer transmission time

Clock sync (NTP-like): the host sends a ping at T1 (host clock), the board receives it at T2 and answers at T3 (board
clock), the host receives the answer at T4. The board time (T2 + T3) / 2 corresponds to the host time (T1 + T4) / 2
within +/- delay / 2, where delay = (T4 - T1) - (T3 - T2): the time the answer waits for the uart (a frame transfer
takes ~25 ms) is excluded. Only the exchanges with the smallest delays of the last SYNC_WINDOW_S are kept (no USB or
OS scheduling latency added) and a line is fitted on them: offset and drift of the board crystal.

Acquisition time of a frame = host time of (trigger + trigger latency + exposure / 2), the centre of the exposure. The
latency between the trigger command and the start of the exposure depends on the power mode: measure it once and pass
it with --trigger-latency-us (0 by default).

Host clock: time.time_ns() (CLOCK_REALTIME) by default, the clock of the global time domain of librealsense
(RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME), so that the frames can be fused with the RealSense ones. The global time domain
must be enabled on the RealSense (RS2_OPTION_GLOBAL_TIME_ENABLED, default on recent librealsense versions).
"""
import collections
import struct
import time

import numpy as np

PING_MAGIC = b"L9PI"
PONG_MAGIC = b"L9PO"
FRAME_TIME_MAGIC = b"L9FT"
CALIB_MAGIC = b"L9CB"
PING_SIZE = 4
PONG_SIZE = 12
FRAME_TIME_SIZE = 16

CLOCKS = {"realtime": time.time_ns, "monotonic": time.monotonic_ns}
PING_PERIOD_S = 0.1
MAX_PENDING_PINGS = 100       # pings without answer are forgotten (lost on the uart or before the board was listening)
SYNC_WINDOW_S = 60.0          # samples older than this are dropped: follows the crystal drift with the temperature
DELAY_QUANTILE = 0.2          # fraction of the samples (smallest delays) used for the fit
MIN_SAMPLES = 10              # samples in the window before the mapping is used
MIN_FIT_SPAN_S = 5.0          # below this span the drift is not estimated (nominal 1 MHz)
MAX_DRIFT_PPM = 500.0         # a larger fitted drift is not a crystal: the nominal rate is used instead
OUTLIER_NS = 20_000_000       # a sample this far from the mapping (beyond its delay) is an outlier
MAX_OUTLIERS = 3              # consecutive outliers that restart the sync (board reset, host clock step)
NS_PER_US = 1000


class McuClock:
    """Unwrap the 32-bit microsecond counter of the board (wraps every ~71.6 min). The values may arrive slightly out
    of order (pong reception time and frame times), the unwrapping is relative to the last value."""

    def __init__(self):
        self.last = None

    def reset(self):
        self.last = None

    def unwrap(self, value):
        if self.last is None:
            self.last = value
            return value
        delta = (value - self.last) & 0xFFFFFFFF
        if delta >= 1 << 31:
            delta -= 1 << 32
        self.last += delta
        return self.last


class ClockSync:
    """Linear mapping host_ns = host_ref + slope * (mcu_us - mcu_ref) fitted on the ping exchanges."""

    def __init__(self, window_s=SYNC_WINDOW_S):
        self.window_ns = int(window_s * 1e9)
        self.samples = collections.deque()  # (mcu_us, host_ns, delay_ns)
        self.model = None                   # (mcu_ref_us, host_ref_ns, slope_ns_per_us)
        self.outliers = 0
        self.residual_ns = 0.0
        self.min_delay_ns = 0

    def reset(self):
        self.samples.clear()
        self.model = None
        self.outliers = 0

    @property
    def synced(self):
        return self.model is not None

    def add(self, t1_ns, t2_us, t3_us, t4_ns):
        """Add one exchange: t1, t4 host clock (ns), t2, t3 unwrapped board clock (us)."""
        delay = (t4_ns - t1_ns) - (t3_us - t2_us) * NS_PER_US
        if delay < 0 or t3_us < t2_us:
            return  # corrupted answer
        mcu = (t2_us + t3_us) / 2
        host = (t1_ns + t4_ns) // 2
        if self.model is not None and abs(self.to_host_ns(mcu) - host) > OUTLIER_NS + delay:
            self.outliers += 1
            if self.outliers < MAX_OUTLIERS:
                return
            self.reset()
        self.outliers = 0
        self.samples.append((mcu, host, delay))
        while host - self.samples[0][1] > self.window_ns:
            self.samples.popleft()
        self._fit()

    def _fit(self):
        if len(self.samples) < MIN_SAMPLES:
            return
        mcu, host, delay = (np.array(v) for v in zip(*self.samples))
        best = delay <= np.quantile(delay, DELAY_QUANTILE)
        mcu, host, delay = mcu[best], host[best], delay[best]
        # small values for the float fit: host times since the epoch need 61 bits
        mcu_ref, host_ref = mcu[-1], int(host[-1])
        x = mcu - mcu_ref
        y = (host - host_ref).astype(np.float64)
        slope = float(NS_PER_US)
        if x[-1] - x[0] >= MIN_FIT_SPAN_S * 1e6:
            fitted, _ = np.polyfit(x, y, 1)
            if abs(fitted / NS_PER_US - 1) * 1e6 <= MAX_DRIFT_PPM:
                slope = fitted
        intercept = float(np.median(y - slope * x))
        self.model = (mcu_ref, host_ref + round(intercept), slope)
        self.residual_ns = float(np.sqrt(np.mean((y - slope * x - intercept) ** 2)))
        self.min_delay_ns = int(delay.min())

    def to_host_ns(self, mcu_us):
        """Host time (ns) of an unwrapped board time (us), None until synced."""
        if self.model is None:
            return None
        mcu_ref, host_ref, slope = self.model
        return host_ref + round(slope * (mcu_us - mcu_ref))

    def status(self):
        if self.model is None:
            return f"sync {len(self.samples)}/{MIN_SAMPLES} samples"
        drift_ppm = (self.model[2] / NS_PER_US - 1) * 1e6
        return (f"sync drift {drift_ppm:+.1f} ppm  min delay {self.min_delay_ns / 1e3:.0f} us  "
                f"residual {self.residual_ns / 1e3:.0f} us")


class StreamClock:
    """Ping the board, keep the clock mapping and stamp the frames of the stream.

    Usage, with the packets of l9_raw_receiver.PacketParser:
        clock = StreamClock(port)
        clock.poll()                       # in the reception loop, sends the pings
        data = port.read(...)
        t_rx = clock.now_ns()              # right after the read
        for magic, payload in parser.feed(data):
            clock.on_packet(magic, payload, t_rx)
            if magic == b"L9RW":
                frame = parse_frame(payload)
                clock.stamp(frame)         # frame["acq_time_ns"], host clock
    """

    def __init__(self, port=None, clock="realtime", trigger_latency_us=0.0, ping_period_s=PING_PERIOD_S):
        self.port = port
        self.now_ns = CLOCKS[clock]
        self.trigger_latency_us = trigger_latency_us
        self.ping_period_s = ping_period_s
        self.mcu = McuClock()
        self.sync = ClockSync()
        self.pending = collections.OrderedDict()  # sequence number -> T1
        self.seq = 0
        self.next_ping = 0.0
        self.frame_time = None

    def poll(self):
        """Send a ping when due (no-op without a port, e.g. reading a capture file)."""
        if self.port is None or time.monotonic() < self.next_ping:
            return
        self.next_ping = time.monotonic() + self.ping_period_s
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        packet = PING_MAGIC + struct.pack("<II", PING_SIZE, self.seq)
        self.pending[self.seq] = self.now_ns()
        self.port.write(packet)
        while len(self.pending) > MAX_PENDING_PINGS:
            self.pending.popitem(last=False)

    def on_packet(self, magic, payload, t_rx_ns):
        """Handle the timing packets, t_rx_ns: host time at which the packet was read."""
        if magic == CALIB_MAGIC:
            # sent once at boot: new board timebase
            self.mcu.reset()
            self.sync.reset()
            self.frame_time = None
        elif magic == PONG_MAGIC:
            seq, t2, t3 = struct.unpack_from("<III", payload, 0)
            t1 = self.pending.pop(seq, None)
            if t1 is not None:
                self.sync.add(t1, self.mcu.unwrap(t2), self.mcu.unwrap(t3), t_rx_ns)
        elif magic == FRAME_TIME_MAGIC:
            counter, trigger, intr, exposure = struct.unpack_from("<IIII", payload, 0)
            self.frame_time = dict(frame_counter=counter, trigger_us=self.mcu.unwrap(trigger),
                                   intr_us=self.mcu.unwrap(intr), exposure_us=exposure)

    def stamp(self, frame):
        """Add the timing of a frame decoded by l9_raw_receiver.parse_frame(), None when unknown:
        trigger_us, intr_us, exposure_us  board clock (unwrapped, us)
        acq_time_ns                       host clock, centre of the exposure (None until the clock is synced)
        """
        ft = self.frame_time
        if ft is None or ft["frame_counter"] != frame["frame_counter"]:
            ft = dict(trigger_us=None, intr_us=None, exposure_us=None)
        frame.update(trigger_us=ft["trigger_us"], intr_us=ft["intr_us"], exposure_us=ft["exposure_us"])
        frame["acq_time_ns"] = None
        if ft["trigger_us"] is not None:
            frame["acq_time_ns"] = self.sync.to_host_ns(
                ft["trigger_us"] + self.trigger_latency_us + ft["exposure_us"] / 2)
        self.frame_time = None
        return frame

    def status(self):
        return self.sync.status()
