"""Several robots in one world (scrimmage/multi.py): two kitbots, each its own robot program at its own path."""

import asyncio
import math
import tomllib
from pathlib import Path

import mujoco
import pytest
import websockets

from fake_robot import next_port
from scrimmage.multi import PATH_PREFIX, MultiServer, MultiWorld
from scrimmage.protocol import HalState, frame
from scrimmage.world import World

TESTS = Path(__file__).resolve().parent
KITBOT = TESTS / "models" / "kitbot.xml"
ROBOTS = {"a": 2.0, "b": 6.0}   # each robot's y (m); both start at x = 2 m, facing +x


def two_kitbots(robots: dict[str, float] = ROBOTS) -> mujoco.MjModel:
    """tests/models/kitbot.xml's field with its kitbot once for each robot, at its y, its names prefixed <name>_."""
    spec = mujoco.MjSpec.from_file(str(KITBOT))
    spec.delete(spec.body("chassis"))
    for e in list(spec.equalities):
        spec.delete(e)
    for name, y in robots.items():
        child = mujoco.MjSpec.from_file(str(KITBOT))
        spec.worldbody.add_frame(pos=[0, y - 4.035, 0]).attach_body(child.body("chassis"), f"{name}_", "")
    return spec.compile()


def kitbot_cfg(p: str) -> dict:
    """fake_robot.toml's drive, and an encoder on the left side, on the joints prefixed p."""
    drive = {"motor": "CIM", "count": 2, "gear_ratio": 10.71}
    return {"robot": {"body": f"{p}chassis"},
            "motor": [{"name": "left", "pwm": 0, "joint": f"{p}left_center", **drive},
                      {"name": "right", "pwm": 1, "joint": f"{p}right_center", "invert": True, **drive}],
            "encoder": [{"joint": f"{p}left_center", "channel_a": 0, "counts_per_rev": 2048}]}


def world() -> MultiWorld:
    return MultiWorld(two_kitbots(), {n: (kitbot_cfg(f"{n}_"), KITBOT.parent) for n in ROBOTS}, timestep=0.001)


def x(w: MultiWorld, name: str) -> float:
    return w.robots[name].robot_pose()[0]


def outputs(speed: float) -> list[str]:
    """Both sides forward at `speed`: the right motor is inverted, so its PWM takes the negative."""
    return [frame("PWM", "0", {"<init": True, "<speed": speed}), frame("PWM", "1", {"<init": True, "<speed": -speed})]


def drive(hal: HalState, speed: float) -> None:
    for f in outputs(speed):
        hal.apply(f)


def test_each_robot_is_driven_by_its_own_outputs():
    w = world()
    hals = {"a": HalState(), "b": HalState()}
    drive(hals["a"], 0.5)
    drive(hals["b"], -0.5)
    for _ in range(1000):
        w.step(hals)
    assert x(w, "a") > 2.3 and x(w, "b") < 1.7
    assert w.robots["a"].robot_pose()[1] == pytest.approx(2.0, abs=0.05)
    assert w.robots["b"].robot_pose()[1] == pytest.approx(6.0, abs=0.05)


def test_a_robot_sharing_the_model_moves_as_it_does_alone():
    w = world()
    alone = World(tomllib.loads((TESTS / "fake_robot.toml").read_text()), TESTS)
    hals, hal = {"a": HalState()}, HalState()
    drive(hals["a"], 0.5)
    drive(hal, 0.5)
    for _ in range(1000):
        w.step(hals)
        alone.step(hal)
    xa, ya, ha = w.robots["a"].robot_pose()
    x1, y1, h1 = alone.robot_pose()
    assert (xa, ya - 2.0, ha) == pytest.approx((x1, y1 - 4.035, h1), abs=1e-6)


def test_a_robot_with_no_program_stays_put():
    w = world()
    hals = {"a": HalState()}
    drive(hals["a"], 0.5)
    for _ in range(500):
        w.step(hals)
    assert x(w, "a") > 2.1 and x(w, "b") == pytest.approx(2.0, abs=0.01)


def test_reset_puts_both_robots_back():
    w = world()
    hals = {"a": HalState(), "b": HalState()}
    drive(hals["a"], 0.5)
    drive(hals["b"], -0.5)
    for _ in range(300):
        w.step(hals)
    w.reset()
    assert w.time == 0.0 and x(w, "a") == pytest.approx(2.0, abs=1e-6) and x(w, "b") == pytest.approx(2.0, abs=1e-6)


def test_a_robot_sharing_a_model_takes_only_its_bindings():
    cfg = {**kitbot_cfg("a_"), "sensor": []}
    with pytest.raises(ValueError, match=r"\[sensor\]"):
        MultiWorld(two_kitbots(), {"a": (cfg, KITBOT.parent)})


class Client:
    """A robot program's side: its outputs sent, the server's inputs merged."""

    def __init__(self, ws):
        self.ws, self.hal = ws, HalState()
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for raw in self.ws:
                self.hal.apply(raw)
        except websockets.ConnectionClosed:
            pass

    def count(self) -> int | None:
        return self.hal.get("Encoder", "0", ">count")


async def connect(port: int, name: str, speed: float) -> Client:
    ws = await websockets.connect(f"ws://localhost:{port}{PATH_PREFIX}{name}", proxy=None)
    c = Client(ws)
    await ws.send(frame("Encoder", "0", {"<init": True, "<channel_a": 0}))
    for f in outputs(speed):
        await ws.send(f)
    return c


def test_two_robot_programs_driven_apart_each_read_only_their_own_sensors():
    async def body():
        w = world()
        server = MultiServer(w, port=next_port())
        task = asyncio.create_task(server.run())
        await asyncio.wait_for(server.listening.wait(), 5)
        a = await connect(server.bound_port, "a", 0.5)
        b = await connect(server.bound_port, "b", -0.5)
        try:
            await asyncio.sleep(1.5)
            assert x(w, "a") > 2.3 and x(w, "b") < 1.7
            ca, cb = a.count(), b.count()
            assert ca is not None and cb is not None and ca > 2048 and cb < -2048
            # each count is its own robot's left wheel, not the other's
            for name, c in (("a", ca), ("b", cb)):
                m, d = w.model, w.data
                q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_left_center")]]
                assert c == pytest.approx(q / (2 * math.pi) * 2048, rel=0.05)
            assert server.links["a"].hal.get("PWM", "0", "<speed") == 0.5
            assert server.links["b"].hal.get("PWM", "0", "<speed") == -0.5
        finally:
            await a.ws.close()
            await b.ws.close()
            server.done.set()
            await asyncio.wait_for(task, 5)
    asyncio.run(body())


def test_an_unknown_path_and_a_second_program_for_a_robot_are_refused():
    async def body():
        server = MultiServer(world(), port=next_port())
        task = asyncio.create_task(server.run())
        await asyncio.wait_for(server.listening.wait(), 5)
        port = server.bound_port
        try:
            first = await websockets.connect(f"ws://localhost:{port}{PATH_PREFIX}a", proxy=None)
            await asyncio.sleep(0.1)
            for path in (f"{PATH_PREFIX}a", f"{PATH_PREFIX}c", "/wpilibws"):
                ws = await websockets.connect(f"ws://localhost:{port}{path}", proxy=None)
                with pytest.raises(websockets.ConnectionClosed) as e:
                    await asyncio.wait_for(ws.recv(), 2)
                assert e.value.rcvd.code == 1008, path
            await first.close()
        finally:
            server.done.set()
            await asyncio.wait_for(task, 5)
    asyncio.run(body())


def test_two_robotpy_programs_each_reach_their_own_path():
    """WPILib's halsim_ws client, given HALSIMWS_URI, connects at its robot's path; each robot's devices land in its own
    HalState."""
    pytest.importorskip("wpilib")
    pytest.importorskip("halsim_ws")
    import os
    import subprocess
    import sys

    async def body():
        server = MultiServer(world(), port=next_port())
        task = asyncio.create_task(server.run())
        await asyncio.wait_for(server.listening.wait(), 5)
        procs = []
        try:
            for name in ROBOTS:
                env = dict(os.environ, HALSIMWS_HOST="localhost", HALSIMWS_PORT=str(server.bound_port),
                           HALSIMWS_URI=f"{PATH_PREFIX}{name}")
                procs.append(subprocess.Popen([sys.executable, "-m", "robotpy", "sim", "--nogui", "--ws-client"],
                                              cwd=TESTS.parent / "examples" / "robotpy_kitbot", env=env,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            for _ in range(300):
                if all(link.hal.get("PWM", "0", "<init") for link in server.links.values()):
                    break
                await asyncio.sleep(0.1)
            for name, link in server.links.items():
                assert link.ws is not None, name
                assert link.hal.get("PWM", "0", "<init") is True, name
        finally:
            for p in procs:
                p.kill()
                p.wait()
            server.done.set()
            await asyncio.wait_for(task, 5)
    asyncio.run(body())


# ---------------------------------------------------------------- lockstep: each robot synced at its own times

PERIODS = {"p21": 21, "p25": 25, "p40": 40}
LOCKSTEP_RUN_S = 2.0


def lockstep_world() -> MultiWorld:
    ys = {name: 1.5 + 2.0 * i for i, name in enumerate(PERIODS)}
    return MultiWorld(two_kitbots(ys), {n: (kitbot_cfg(f"{n}_"), KITBOT.parent) for n in PERIODS}, timestep=0.001)


def run_probes(server: MultiServer, hold: dict[str, tuple[float, float]] | None = None, run_s: float = LOCKSTEP_RUN_S,
               min_wall_s: float = 0.0):
    """Run tests/multi_probe once for each robot (each at its period; `hold` maps a robot to (FPGA ms, wall s) of its
    breakpoint stand-in) against `server` until simulated `run_s` and at least `min_wall_s` of wall time. Returns each robot's loops, (sync T, loop's FPGA
    time) for each sync that saw a new loop, (wall s, simulated s) samples, and the simulated time reached, s."""
    pytest.importorskip("wpilib")
    pytest.importorskip("halsim_ws")
    pytest.importorskip("scrimmage_lockstep")
    import os
    import subprocess
    import sys
    import time

    loops: dict[str, list[tuple[int, int]]] = {n: [] for n in server.links}
    counts = {n: 0 for n in server.links}
    server.runs = {n: [] for n in server.links}   # (sync T, loops it ran) for each sync that ran any

    def answered(name: str, at: int) -> None:
        n = server.links[name].hal.get("SimDevice", "Probe", "<n")
        if isinstance(n, int) and n != counts[name]:
            server.runs[name].append((at, n - counts[name]))
            counts[name] = n
            loops[name].append((at, server.links[name].hal.get("SimDevice", "Probe", "<t")))
    server.on_answer = answered
    samples: list[tuple[float, float]] = []

    async def body():
        task = asyncio.create_task(server.run())
        await asyncio.wait_for(server.listening.wait(), 5)
        procs = []
        try:
            for name, ms in PERIODS.items():
                env = dict(os.environ, HALSIMWS_HOST="localhost", HALSIMWS_PORT=str(server.bound_port),
                           HALSIMWS_URI=f"{PATH_PREFIX}{name}", PROBE_PERIOD_MS=str(ms))
                if hold and name in hold:
                    env.update(PROBE_HOLD_AT_MS=str(hold[name][0]), PROBE_HOLD_S=str(hold[name][1]))
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "robotpy", "sim", "--nogui", "--ws-client", "--scrimmage-lockstep"],
                    cwd=TESTS / "multi_probe", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            deadline = time.monotonic() + 60
            while not all(lk.engaged for lk in server.links.values()):
                assert time.monotonic() < deadline, {n: lk.engaged for n, lk in server.links.items()}
                await asyncio.sleep(0.05)
            start, wall_start = server.world.time, time.monotonic()
            while server.world.time - start < run_s or time.monotonic() - wall_start < min_wall_s:
                assert time.monotonic() < deadline + 60
                samples.append((time.monotonic(), server.world.time - start))
                await asyncio.sleep(0.01)
        finally:
            server.done.set()
            await asyncio.wait_for(task, 10)
            for p in procs:
                p.kill()
                p.wait()
    asyncio.run(body())
    return loops, samples, server.world.time


def longest_stall(samples: list[tuple[float, float]]) -> float:
    """The longest wall time over which simulated time stood still."""
    longest, since = 0.0, None
    for (w0, s0), (w1, s1) in zip(samples, samples[1:]):
        since = (since if s1 == s0 and since is not None else w0) if s1 == s0 else None
        if since is not None:
            longest = max(longest, w1 - since)
    return longest


def assert_on_time(name: str, loops: list[tuple[int, int]], runs: list[tuple[int, int]]) -> None:
    """Each sync ran one loop, at its alarm within the millisecond synced, the alarms one period apart."""
    period = PERIODS[name] * 1000
    assert all(n == 1 for _, n in runs), (name, [r for r in runs if r[1] != 1])
    for at, ran in loops:
        assert at - 1000 < ran <= at, (name, at, ran)
    gaps = [b[1] - a[1] for a, b in zip(loops, loops[1:])]
    assert set(gaps) == {period}, (name, sorted(set(gaps)))


def test_robots_of_21_25_and_40_ms_are_each_synced_at_their_own_times():
    server = MultiServer(lockstep_world(), port=next_port(), speed=None)
    loops, _, end = run_probes(server)
    for name, ms in PERIODS.items():
        assert_on_time(name, loops[name], server.runs[name])
        assert len(loops[name]) == pytest.approx(end * 1000 / ms, abs=2), name
        assert server.links[name].syncs <= len(loops[name]) * 2 + 5, name   # synced when due, not each millisecond
    # their times differ: each robot's loops fall on its own grid
    times = {name: {at for at, _ in loops[name]} for name in PERIODS}
    assert times["p21"] != times["p25"] != times["p40"]


def test_a_breakpoint_holds_every_robot_when_the_wait_has_no_limit():
    server = MultiServer(lockstep_world(), port=next_port(), speed=None, sync_timeout=None)
    loops, samples, _ = run_probes(server, hold={"p25": (500, 2.0)})
    assert longest_stall(samples) > 1.5                     # simulated time stood still while p25 was held
    for name in PERIODS:
        assert_on_time(name, loops[name], server.runs[name])   # and no robot missed a loop
    assert all(lk.timeouts == 0 for lk in server.links.values())


def test_a_robot_held_past_its_sync_timeout_is_left_behind_and_rejoins():
    server = MultiServer(lockstep_world(), port=next_port(), speed=None, sync_timeout=0.3)
    loops, samples, _ = run_probes(server, hold={"p25": (500, 1.0)}, min_wall_s=2.0)   # past the hold
    assert longest_stall(samples) < 1.0                     # the others went on
    assert_on_time("p21", loops["p21"], server.runs["p21"])
    assert_on_time("p40", loops["p40"], server.runs["p40"])
    assert server.links["p25"].timeouts >= 1
    # p25 rejoined at the server's time: the extension jumps the clock, so TimedRobot ran one loop then and skipped
    # the ones it missed, as a late roboRIO does, then went on along its own 25 ms grid
    assert all(n == 1 for _, n in server.runs["p25"]), server.runs["p25"]
    gaps = [b[1] - a[1] for a, b in zip(loops["p25"], loops["p25"][1:])]
    held = [i for i, g in enumerate(gaps) if g > 25_000]
    assert len(held) == 1 and gaps[held[0]] > 500_000, gaps         # one gap, the hold, and no loops run to fill it
    rejoin = loops["p25"][held[0] + 1]
    assert rejoin[0] > 600_000 and rejoin[0] - 1000 < rejoin[1] <= rejoin[0]
    after = loops["p25"][held[0] + 2:]
    assert after and all(at - 1000 < ran <= at for at, ran in after)     # then on time again
    assert all(ran % 25_000 == loops["p25"][0][1] % 25_000 for _, ran in after)
