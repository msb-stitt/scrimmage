"""Probe robot for stepping to a time: loops of 20 ms, 21 ms, 25 ms and 40 ms, each reporting its count, the FPGA time
it ran at and the sensor value it read, over a SimDevice."""

import hal
import wpilib

_D = hal.SimDevice.Direction
PERIODS_MS = (21, 25, 40)


class MyRobot(wpilib.TimedRobot):
    def robotInit(self):
        self.dev = hal.SimDevice("TimeProbe")
        self.sensor = self.dev.createDouble("sensor", _D.kInput, 0.0)
        self.out = {}
        for ms in (20, *PERIODS_MS):
            # time first: when scrimmage sees a loop's count change, its time and reading are already there
            self.out[ms] = (self.dev.createLong(f"t{ms}", _D.kOutput, 0), self.dev.createDouble(f"s{ms}", _D.kOutput, 0.0),
                            self.dev.createLong(f"n{ms}", _D.kOutput, 0))
        for ms in PERIODS_MS:
            self.addPeriodic(lambda ms=ms: self._loop(ms), ms / 1000)

    def _loop(self, ms):
        t, s, n = self.out[ms]
        t.set(wpilib.RobotController.getFPGATime())
        s.set(self.sensor.get())
        n.set(n.get() + 1)

    def robotPeriodic(self):
        self._loop(20)
