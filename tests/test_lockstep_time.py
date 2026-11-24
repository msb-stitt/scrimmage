"""The lockstep extension stepped to a time (step_us = 0): each step_seq is the FPGA time, us, to step to, and the
extension answers with next_us, the robot's next notifier alarm, before done_seq. This test plays the server: it syncs
the robot only at its next alarm rounded up to the millisecond, setting a sensor to that millisecond first.

Needs `pip install ./lockstep` (plus robotpy[sim] and robotpy-halsim-ws).
"""

import asyncio
import os
import subprocess
import sys

import pytest
import websockets

pytest.importorskip("wpilib")
pytest.importorskip("halsim_ws")
pytest.importorskip("scrimmage_lockstep")

from test_lockstep_ext import DEV, FakeSim  # noqa: E402

from scrimmage.protocol import frame  # noqa: E402

PORT = 3319
PROBE = ("SimDevice", "TimeProbe")
PERIODS_MS = (20, 21, 25, 40)
RUN_US = 1_000_000


def ceil_ms(us):
    return -(-us // 1000) * 1000


async def run_probe(body):
    sim = FakeSim()
    env = dict(os.environ, HALSIMWS_PORT=str(PORT), HALSIMWS_HOST="localhost")
    async with websockets.serve(sim.handler, "localhost", PORT):
        robot = subprocess.Popen(
            [sys.executable, "-m", "robotpy", "sim", "--nogui", "--ws-client", "--scrimmage-lockstep"],
            cwd=os.path.join(os.path.dirname(__file__), "lockstep_time_probe"), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            await asyncio.wait_for(sim.connected.wait(), 30)
            await sim.until(lambda: DEV in sim.hal.devices and sim.get(PROBE, "<n20") is not None, 30)
            await body(sim)
        finally:
            robot.kill()
            out = robot.communicate()[0]
    return out


async def sync(sim, t_us):
    """One sync at t_us: the sensor set to t_us, then the step; returns the robot's next alarm."""
    await sim.ws.send(frame(*PROBE, {">sensor": float(t_us)}))
    await sim.ws.send(frame(*DEV, {">step_seq": t_us}))
    await sim.until(lambda: (sim.get(DEV, "<done_seq") or 0) >= t_us, 5)
    assert sim.get(DEV, "<done_seq") == t_us   # each sync answered once, with its own time
    return sim.get(DEV, "<next_us")


def test_steps_to_each_alarm_rounded_up_to_the_millisecond():
    async def body(sim):
        await sim.engage(step_us=0)
        nxt = await sync(sim, 1000)
        syncs, seen = 0, {ms: 0 for ms in PERIODS_MS}
        last = 1000
        while last < RUN_US:
            assert nxt > last                                   # the next alarm is after the time stepped to
            t = ceil_ms(nxt)
            counts = {ms: sim.get(PROBE, f"<n{ms}") for ms in PERIODS_MS}
            nxt = await sync(sim, t)
            syncs += 1
            for ms in PERIODS_MS:
                n = sim.get(PROBE, f"<n{ms}")
                if n != counts[ms]:
                    assert n == counts[ms] + 1, (ms, t)          # one run of a loop for each of its alarms
                    ran = sim.get(PROBE, f"<t{ms}")
                    assert t - 1000 < ran <= t, (ms, t, ran)     # at its alarm, within the millisecond synced
                    assert sim.get(PROBE, f"<s{ms}") == t        # reading that millisecond's sensor
                    seen[ms] += 1
            last = t
        for ms in PERIODS_MS:
            assert seen[ms] == pytest.approx(RUN_US / 1000 / ms, abs=1), (ms, seen)
        assert syncs < RUN_US / 1000 / 20 * 4                    # synced only when a loop is due, not each ms
        frozen = {ms: sim.get(PROBE, f"<n{ms}") for ms in PERIODS_MS}
        await asyncio.sleep(0.5)                                 # no step, no robot time
        assert {ms: sim.get(PROBE, f"<n{ms}") for ms in PERIODS_MS} == frozen
    asyncio.run(run_probe(body))
