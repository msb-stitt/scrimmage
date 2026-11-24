"""The lockstep extension stepped to a time (step_us = 0): each step_seq is the FPGA time, us, to jump to, and the
extension answers with next_us, the robot's next notifier alarm, before done_seq. This test plays the server: it syncs
the robot only at its next alarm rounded up to the millisecond, setting a sensor to that millisecond first.

Needs `pip install ./lockstep` (plus robotpy[sim] and robotpy-halsim-ws). The Java robot, a copy of the RobotPy one,
also needs a JDK 17 and runs with the java marker.
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

from scrimmage import java_tools, robot_process  # noqa: E402
from scrimmage.protocol import frame  # noqa: E402

PORT = 3319
TESTS = os.path.dirname(os.path.abspath(__file__))
PROBE = ("SimDevice", "TimeProbe")
PERIODS_MS = (20, 21, 25, 40)
RUN_US = 1_000_000


def ceil_ms(us):
    return -(-us // 1000) * 1000


def robot_command(lang):
    """The probe robot's command, folder and extra environment, by its language."""
    if lang == "python":
        return ([sys.executable, "-m", "robotpy", "sim", "--nogui", "--ws-client", "--scrimmage-lockstep"],
                os.path.join(TESTS, "lockstep_time_probe"), {})
    gradlew = "gradlew.bat" if sys.platform == "win32" else "./gradlew"
    return ([gradlew, "-I", str(java_tools.init_script_path()), "simulateJava"],
            os.path.join(TESTS, "lockstep_time_java_probe"), robot_process._scrimmage_paths())


async def run_probe(body, lang="python", connect_s=30):
    sim = FakeSim()
    command, cwd, extra = robot_command(lang)
    env = dict(os.environ, HALSIMWS_PORT=str(PORT), HALSIMWS_HOST="localhost", **extra)
    async with websockets.serve(sim.handler, "localhost", PORT):
        robot = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=sys.platform != "win32")
        try:
            await asyncio.wait_for(sim.connected.wait(), connect_s)
            await sim.until(lambda: DEV in sim.hal.devices and sim.get(PROBE, "<n20") is not None, 30)
            await body(sim)
        finally:
            if sys.platform != "win32":
                os.killpg(robot.pid, 9)   # Gradle's robot JVM is a child of the wrapper
            else:
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


@pytest.fixture(scope="module")
def java_probe_built():
    if "SCRIMMAGE_LOCKSTEP_SO" not in robot_process._scrimmage_paths():
        pytest.skip("no lockstep extension built for this computer")
    java_tools.build_agent()
    command, cwd, extra = robot_command("java")
    r = subprocess.run([command[0], "build"], cwd=cwd, env=dict(os.environ, **extra), capture_output=True, text=True)
    assert r.returncode == 0, "gradlew build failed:\n" + (r.stdout + r.stderr)[-3000:]


@pytest.mark.parametrize("lang", ["python", pytest.param("java", marks=pytest.mark.java)])
def test_steps_to_each_alarm_rounded_up_to_the_millisecond(lang, request):
    if lang == "java":
        request.getfixturevalue("java_probe_built")

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
                    assert ran == t, (ms, t, ran)                # at the time synced, its alarm within that ms
                    assert sim.get(PROBE, f"<s{ms}") == t        # reading that millisecond's sensor
                    seen[ms] += 1
            last = t
        for ms in PERIODS_MS:
            assert seen[ms] == pytest.approx(RUN_US / 1000 / ms, abs=1), (ms, seen)
        assert syncs < RUN_US / 1000 / 20 * 4                    # synced only when a loop is due, not each ms
        frozen = {ms: sim.get(PROBE, f"<n{ms}") for ms in PERIODS_MS}
        await asyncio.sleep(0.5)                                 # no step, no robot time
        assert {ms: sim.get(PROBE, f"<n{ms}") for ms in PERIODS_MS} == frozen
        # a sync 1 s on, as after a stall: each loop runs once then, skipping the ones it missed as TimedRobot does
        # after an overrun, and its next alarm is on its own grid again
        t = ceil_ms(last + 1_000_000)
        nxt = await sync(sim, t)
        for ms in PERIODS_MS:
            assert sim.get(PROBE, f"<n{ms}") == frozen[ms] + 1, (ms, frozen[ms], sim.get(PROBE, f"<n{ms}"))
            assert sim.get(PROBE, f"<t{ms}") == t
        assert t < nxt <= t + 20_000
    asyncio.run(run_probe(body, lang, connect_s=30 if lang == "python" else 180))
