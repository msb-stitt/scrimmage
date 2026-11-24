"""Several robots in one MuJoCo world, each robot program its own halsim_ws client (the fork's multi-robot branch).

A MultiWorld holds one model, the field and every robot in it with their joints and bodies named apart (by prefix, as
MjSpec's attach or mujocobiobuzz's field.with_robots makes them), and a World for each robot over that shared model:
its bindings, battery and pose its own. One physics step drives every robot's motors, then steps MuJoCo once.

A MultiServer serves them on one port, each robot program at its own path, /wpilibws/<name>, which it reaches by
HALSIMWS_URI. Each connection keeps its own HalState, so a robot reads only its own sensors. It runs in real time;
lockstep for several robots is the plan's scr.barrier.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import websockets

from .protocol import HalState, OutboundCache
from .world import World

log = logging.getLogger("scrimmage")

PATH_PREFIX = "/wpilibws/"
SENSOR_PERIOD = 0.005   # simulated seconds between sensor pushes to each robot, as the Bridge's
PHYSICS_TICK = 0.001    # wall seconds to sleep when physics has caught up with real time
MAX_LAG = 0.25          # behind real time by more than this, physics skips ahead rather than racing to catch up


class MultiWorld:
    """One model holding the field and several robots; `robots` maps each robot's name to its config and the folder
    the config's relative paths start from."""

    def __init__(self, model: mujoco.MjModel, robots: dict[str, tuple[dict, Path]], timestep: float | None = None):
        self.model = model
        if timestep is not None:
            model.opt.timestep = float(timestep)
        self.data = mujoco.MjData(model)
        self.robots = {name: World(cfg, cfg_dir, shared=(model, self.data)) for name, (cfg, cfg_dir) in robots.items()}
        self._base_damping = model.dof_damping.copy()
        self.reset()

    @classmethod
    def from_mjcf(cls, mjcf: str | Path, robots: dict[str, tuple[dict, Path]], timestep: float | None = None):
        return cls(mujoco.MjModel.from_xml_path(str(mjcf)), robots, timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for w in self.robots.values():
            w._apply_start_pose()
        mujoco.mj_forward(self.model, self.data)
        for w in self.robots.values():
            for g in w.gyros:
                g.reset()
            w.vbatt = w.battery.nominal_voltage

    def step(self, hals: dict[str, HalState]) -> None:
        """One physics timestep, each robot driven by its own robot program's latest outputs (an empty HalState for
        a robot with none)."""
        self.data.qfrc_applied[:] = 0.0
        self.model.dof_damping[:] = self._base_damping
        for name, w in self.robots.items():
            w.drive(hals.get(name) or HalState(), clear=False)
        mujoco.mj_step(self.model, self.data)
        for w in self.robots.values():
            w.after_physics()


@dataclass
class _Link:
    """One robot's connection: its HAL state and what has been sent to it."""
    hal: HalState = field(default_factory=HalState)
    cache: OutboundCache = field(default_factory=OutboundCache)
    ws: Any = None


class MultiServer:
    """Serves a MultiWorld's robots over halsim_ws in real time, each at PATH_PREFIX + its name."""

    def __init__(self, world: MultiWorld, host: str = "localhost", port: int = 3300):
        self.world = world
        self.host, self.port = host, port
        self.links = {name: _Link() for name in world.robots}
        self.done = asyncio.Event()
        self.listening = asyncio.Event()
        self.bound_port: int | None = None
        self.physics_skipped = 0.0

    def hals(self) -> dict[str, HalState]:
        return {name: link.hal for name, link in self.links.items()}

    async def _handler(self, ws) -> None:
        path = getattr(getattr(ws, "request", None), "path", "")
        name = path[len(PATH_PREFIX):] if path.startswith(PATH_PREFIX) else None
        link = self.links.get(name) if name else None
        if link is None:
            await ws.close(1008, f"no robot at {path}; robots: " + ", ".join(PATH_PREFIX + n for n in self.links))
            return
        if link.ws is not None:
            await ws.close(1008, f"robot {name} is already connected")
            return
        log.info("robot program %s connected", name)
        link.ws, link.hal, link.cache = ws, HalState(), OutboundCache()
        try:
            async for raw in ws:
                got = link.hal.apply(raw)
                if got is not None and got[1].get("<init") is True:
                    link.cache.forget(*got[0])   # initializing a device resets its sim values: resend ours
        except websockets.ConnectionClosed:
            pass
        finally:
            log.info("robot program %s disconnected", name)
            link.ws = None

    async def _push_sensors(self) -> None:
        for name, link in self.links.items():
            ws = link.ws
            if ws is None:
                continue
            msgs = link.cache.frames(self.world.robots[name].sensor_updates(link.hal))
            if not msgs:
                continue
            try:
                for m in msgs:
                    await ws.send(m)
            except websockets.ConnectionClosed:
                pass

    async def _physics(self) -> None:
        """Step physics to keep pace with the wall clock, in passes of at most SENSOR_PERIOD, each robot's sensors
        pushed after each."""
        w = self.world
        wall0, sim0 = time.monotonic(), w.time
        last_sensor = -math.inf
        while not self.done.is_set():
            target = time.monotonic() - wall0
            lag = target - (w.time - sim0)
            if lag > MAX_LAG:
                log.warning("physics fell %.0f ms behind real time; skipping ahead", 1000 * lag)
                self.physics_skipped += lag
                sim0 = w.time - target
            stop = min(target, w.time - sim0 + SENSOR_PERIOD)
            hals = self.hals()
            while (w.time - sim0) < stop:
                w.step(hals)
            if w.time - last_sensor >= SENSOR_PERIOD - 1e-9:
                last_sensor = w.time
                await self._push_sensors()
            await asyncio.sleep(PHYSICS_TICK if w.time - sim0 >= target else 0)

    async def run(self) -> None:
        # halsim_ws_client never answers a close frame, so don't wait the default 10 s for it
        async with websockets.serve(self._handler, self.host, self.port, max_size=None, close_timeout=1) as server:
            self.bound_port = server.sockets[0].getsockname()[1]
            self.listening.set()
            log.info("HALSim WebSocket server listening on ws://%s:%d%s<robot>", self.host, self.bound_port,
                     PATH_PREFIX)
            physics = asyncio.create_task(self._physics())
            await self.done.wait()
            physics.cancel()
            try:
                await physics
            except asyncio.CancelledError:
                pass
            for link in self.links.values():
                if link.ws is not None:
                    await link.ws.close()
