"""Probe robot for several robots in lockstep (tests/test_multi.py): one loop, TimedRobot's, at PROBE_PERIOD_MS,
reporting its count and the FPGA time it ran at over a SimDevice. With PROBE_HOLD_AT_MS, the first loop at or after
that FPGA time sleeps PROBE_HOLD_S wall seconds, a breakpoint's stand-in. With PROBE_NAME, it puts that name in its
NetworkTables as /SmartDashboard/robot."""

import os
import time

import hal
import wpilib

_D = hal.SimDevice.Direction
PERIOD_MS = int(os.environ.get("PROBE_PERIOD_MS", "20"))
HOLD_AT_US = int(float(os.environ.get("PROBE_HOLD_AT_MS", "-1")) * 1000)
HOLD_S = float(os.environ.get("PROBE_HOLD_S", "0"))


class MyRobot(wpilib.TimedRobot):
    def __init__(self):
        super().__init__(PERIOD_MS / 1000)
        self.dev = hal.SimDevice("Probe")
        # time first: when the server sees the count change, the loop's time is already there
        self.t = self.dev.createLong("t", _D.kOutput, 0)
        self.n = self.dev.createLong("n", _D.kOutput, 0)
        self.held = False
        if os.environ.get("PROBE_NAME"):
            wpilib.SmartDashboard.putString("robot", os.environ["PROBE_NAME"])

    def robotPeriodic(self):
        now = wpilib.RobotController.getFPGATime()
        if HOLD_AT_US >= 0 and not self.held and now >= HOLD_AT_US:
            self.held = True
            time.sleep(HOLD_S)
        self.t.set(now)
        self.n.set(self.n.get() + 1)
