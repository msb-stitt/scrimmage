"""Each robot program's NetworkTables on its own 5810, by tools/robot-netns.sh: two robots in network namespaces of
their own, each read from the host at its own address. Linux, as root (the robot programs run as SCRIMMAGE_NETNS_USER).
"""

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux") or os.geteuid() != 0
                                or not os.environ.get("SCRIMMAGE_NETNS_USER") or not shutil.which("ip"),
                                reason="needs Linux, root, ip and SCRIMMAGE_NETNS_USER")

from fake_robot import next_port  # noqa: E402
from test_multi import KITBOT, kitbot_cfg, two_kitbots  # noqa: E402

from scrimmage.multi import PATH_PREFIX, MultiServer, MultiWorld  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = {"a": 1, "b": 2}   # name: the namespace's index, its address 10.231.<index>.2


def read_robot_name(address: str, timeout: float = 30.0) -> str | None:
    """/SmartDashboard/robot as NetworkTables at address:5810 serves it."""
    import ntcore
    inst = ntcore.NetworkTableInstance.create()
    try:
        inst.setServer(address, 5810)
        inst.startClient4("scrimmage-test")
        sub = inst.getStringTopic("/SmartDashboard/robot").subscribe("")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sub.get():
                return sub.get()
            time.sleep(0.1)
        return None
    finally:
        ntcore.NetworkTableInstance.destroy(inst)


def test_each_robot_serves_its_own_networktables_on_5810():
    pytest.importorskip("wpilib")
    pytest.importorskip("halsim_ws")

    async def body():
        world = MultiWorld(two_kitbots({n: 2.0 + 2.0 * i for i, n in enumerate(ROBOTS)}),
                           {n: (kitbot_cfg(f"{n}_"), KITBOT.parent) for n in ROBOTS}, timestep=0.001)
        server = MultiServer(world, host="0.0.0.0", port=next_port())
        task = asyncio.create_task(server.run())
        await asyncio.wait_for(server.listening.wait(), 5)
        procs = []
        try:
            for name, index in ROBOTS.items():
                env = dict(os.environ, RUN_AS=os.environ["SCRIMMAGE_NETNS_USER"], HALSIMWS_PORT=str(server.bound_port),
                           HALSIMWS_URI=f"{PATH_PREFIX}{name}", PROBE_NAME=name)
                procs.append(subprocess.Popen(
                    [str(ROOT / "tools" / "robot-netns.sh"), name, str(index), "--",
                     sys.executable, "-m", "robotpy", "sim", "--nogui", "--ws-client"],
                    cwd=ROOT / "tests" / "multi_probe", env=env))
            deadline = time.monotonic() + 60
            while not all(lk.ws is not None for lk in server.links.values()):
                assert time.monotonic() < deadline, "the robots did not connect from their namespaces"
                await asyncio.sleep(0.1)
            for name, index in ROBOTS.items():
                got = await asyncio.to_thread(read_robot_name, f"10.231.{index}.2")
                assert got == name, (name, got)
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.wait(timeout=15)
            server.done.set()
            await asyncio.wait_for(task, 10)
    asyncio.run(body())
