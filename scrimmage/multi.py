"""Several robots in one MuJoCo world, each robot program its own halsim_ws client (the fork's multi-robot branch).

A MultiWorld holds one model, the field and every robot in it with their joints and bodies named apart (by prefix, as
MjSpec's attach or mujocobiobuzz's field.with_robots makes them), and a World for each robot over that shared model:
its bindings, battery and pose its own. One physics step drives every robot's motors, then steps MuJoCo once.

A MultiServer serves them on one port, each robot program at its own path, /wpilibws/<name>, which it reaches by
HALSIMWS_URI. Each connection keeps its own HalState, so a robot reads only its own sensors.

The server steps physics 1 ms at a time. A robot with the lockstep extension is engaged with step_us = 0 and synced
at its own milliseconds: at T, if its next time is T, it is sent its sensors and step_seq = T, runs what is due, and
answers with done_seq and next_us, its next alarm, which rounded up to the millisecond is its next time. Robots due
at the same T run in parallel. A robot without the extension runs on its own clock and is sent its sensors every
SENSOR_PERIOD.

With `speed` 1.0 the run is real time: physics passes T no sooner than the wall clock, and the server does not wait
for a late robot past then; it steps with the outputs it last had, and the robot, once it answers, is synced next at
the server's time, so its clock jumps and TimedRobot skips the loops it missed. With `speed` None, physics holds until
every robot has connected and engaged, then the run is as fast as the robots allow and waits for each robot due: without limit (`sync_timeout` None), so a robot held at a
breakpoint holds every robot; or for `sync_timeout` wall seconds, after which the run goes on with that robot's last
outputs, and the robot rejoins at its next time once it answers. A connected robot not engaged makes the run real
time.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import websockets

from .lockstep import LOCKSTEP_DEVICE
from .protocol import HalState, OutboundCache, frame
from .world import World

log = logging.getLogger("scrimmage")

PATH_PREFIX = "/wpilibws/"
SENSOR_PERIOD = 0.005   # simulated seconds between sensor pushes to each robot, as the Bridge's
PHYSICS_TICK = 0.001    # wall seconds to sleep when physics has caught up with real time
MAX_LAG = 0.25          # behind real time by more than this, physics skips ahead rather than racing to catch up
MS = 1000               # us in the physics step, the grid robots are synced on
MAX_GAP_US = 100_000    # a robot with no alarm due is synced at least this often, us
KEEPALIVE_PERIOD = 0.5  # wall seconds between keepalives to engaged robots while the server waits
DETECT_WINDOW = 5.0     # wall seconds a connected robot has to show the lockstep extension before it is taken as without


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
            w.plugins.reset()

    def step(self, hals: dict[str, HalState]) -> None:
        """One physics timestep, each robot driven by its own robot program's latest outputs (an empty HalState for
        a robot with none)."""
        self.data.qfrc_applied[:] = 0.0
        self.model.dof_damping[:] = self._base_damping
        hals = {name: hals.get(name) or HalState() for name in self.robots}
        for name, w in self.robots.items():
            w.drive(hals[name], clear=False)
        mujoco.mj_step(self.model, self.data)
        for name, w in self.robots.items():
            w.after_physics(hals[name])


def ceil_ms(us: int) -> int:
    return -(-us // MS) * MS


@dataclass
class _Link:
    """One robot's connection: its HAL state and what has been sent to it, and its lockstep: engaged, the time it
    is next synced at, and the time of the sync it has not answered yet (None when none is out)."""
    hal: HalState = field(default_factory=HalState)
    cache: OutboundCache = field(default_factory=OutboundCache)
    ws: Any = None
    connected_at: float = 0.0
    engage_sent: bool = False
    engaged: bool = False
    next_us: int = 0
    pending: int | None = None
    pending_since: float = 0.0
    syncs: int = 0
    timeouts: int = 0


class MultiServer:
    """Serves a MultiWorld's robots over halsim_ws in real time, each at PATH_PREFIX + its name."""

    def __init__(self, world: MultiWorld, host: str = "localhost", port: int = 3300, *, speed: float | None = 1.0,
                 sync_timeout: float | None = None):
        if speed is not None and speed != 1.0:
            raise ValueError("speed is 1.0 (real time) or None (as fast as the robots allow)")
        self.world = world
        self.host, self.port = host, port
        self.speed, self.sync_timeout = speed, sync_timeout
        self.links = {name: _Link() for name in world.robots}
        self.done = asyncio.Event()
        self.listening = asyncio.Event()
        self.changed = asyncio.Event()   # a robot answered a sync or engaged
        self.bound_port: int | None = None
        self.physics_skipped = 0.0
        self.steps_per_ms = round(MS / 1e6 / world.model.opt.timestep)
        if self.steps_per_ms < 1 or abs(self.steps_per_ms * world.model.opt.timestep - MS / 1e6) > 1e-9:
            raise ValueError(f"the physics timestep {world.model.opt.timestep:g} s must divide 1 ms")
        # called with (robot name, T in us) when a robot answers its sync at T, its outputs then in its HalState
        self.on_answer: Callable[[str, int], None] | None = None

    @property
    def now_us(self) -> int:
        return round(self.world.time * 1e6)

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
        link.ws, link.hal, link.cache, link.connected_at = ws, HalState(), OutboundCache(), time.monotonic()
        link.engage_sent = link.engaged = False
        link.pending = None
        try:
            async for raw in ws:
                got = link.hal.apply(raw)
                if got is None:
                    continue
                if got[1].get("<init") is True:
                    link.cache.forget(*got[0])   # initializing a device resets its sim values: resend ours
                if got[0] == LOCKSTEP_DEVICE:
                    await self._lockstep_frame(name, link, got[1])
        except websockets.ConnectionClosed:
            pass
        finally:
            log.info("robot program %s disconnected", name)
            link.ws = None
            link.engaged = False
            link.pending = None
            self.changed.set()

    async def _lockstep_frame(self, name: str, link: _Link, delta: dict) -> None:
        if not link.engage_sent:
            link.engage_sent = True   # step_us goes with engage, so the extension never reads a stale value
            await self._send(link, [frame(*LOCKSTEP_DEVICE, {">step_us": 0, ">engage": True})])
        if delta.get("<engaged") is True and not link.engaged:
            # synced at the server's next millisecond, and not at 0 s: the extension answers a step_seq above 0
            link.engaged, link.next_us, link.pending = True, MS, None
            log.info("robot program %s in lockstep", name)
        elif delta.get("<engaged") is False and link.engaged:
            link.engaged, link.pending = False, None
            if not self.done.is_set():   # not the server's own release as it stops
                log.warning("robot program %s left lockstep; it runs on its own clock", name)
        done = delta.get("<done_seq")
        if link.pending is not None and isinstance(done, int) and done >= link.pending:
            at, link.pending = link.pending, None
            nxt = link.hal.get(*LOCKSTEP_DEVICE, "<next_us")
            nxt = at + MS if not isinstance(nxt, int) else ceil_ms(nxt)
            link.next_us = max(at + MS, min(nxt, at + MAX_GAP_US))
            if self.on_answer is not None:
                self.on_answer(name, at)
        self.changed.set()

    async def _send(self, link: _Link, msgs: list[str]) -> None:
        ws = link.ws
        if ws is None or not msgs:
            return
        try:
            for m in msgs:
                await ws.send(m)
        except websockets.ConnectionClosed:
            pass

    async def _push_sensors(self, name: str) -> None:
        link = self.links[name]
        await self._send(link, link.cache.frames(self.world.robots[name].sensor_updates(link.hal)))

    def starting(self) -> bool:
        """Fast: a robot has yet to connect, or to engage within DETECT_WINDOW of connecting."""
        wall = time.monotonic()
        return self.speed is None and any(
            lk.ws is None or (not lk.engaged and LOCKSTEP_DEVICE not in lk.hal.devices
                              and wall - lk.connected_at < DETECT_WINDOW)
            or (not lk.engaged and LOCKSTEP_DEVICE in lk.hal.devices) for lk in self.links.values())

    async def _keepalive(self) -> None:
        wall = time.monotonic()
        for lk in self.links.values():
            if lk.engaged and lk.pending is None:
                await self._send(lk, [frame(*LOCKSTEP_DEVICE, {">keepalive": int(wall * 1000)})])

    def realtime(self) -> bool:
        """Real time by the run's speed, or because a connected robot is not in lockstep."""
        return self.speed is not None or any(lk.ws is not None and not lk.engaged for lk in self.links.values())

    async def _sync_due(self, now: int) -> list[str]:
        """Sync each engaged robot whose next time is `now`: its sensors at `now`, then step_seq = now."""
        due = [n for n, lk in self.links.items()
               if lk.ws is not None and lk.engaged and lk.pending is None and lk.next_us <= now]
        for name in due:
            link = self.links[name]
            await self._push_sensors(name)
            link.pending, link.pending_since = now, time.monotonic()
            link.syncs += 1
            await self._send(link, [frame(*LOCKSTEP_DEVICE, {">step_seq": now})])
        return due

    async def _wait_answers(self, now: int, deadline: float | None) -> None:
        """Wait until each robot synced at `now` has answered, or the wall clock reaches `deadline`, or (fast, with a
        sync_timeout) a robot's sync times out. Keeps the robots not being waited on engaged meanwhile."""
        next_keepalive = time.monotonic() + KEEPALIVE_PERIOD
        while not self.done.is_set():
            waiting = [lk for lk in self.links.values() if lk.ws is not None and lk.pending == now]
            if not waiting:
                return
            wall = time.monotonic()
            if deadline is not None and wall >= deadline:
                return
            if self.sync_timeout is not None and deadline is None:
                late = [lk for lk in waiting if wall - lk.pending_since >= self.sync_timeout]
                if len(late) == len(waiting):
                    for lk in late:
                        lk.timeouts += 1
                    return
            if wall >= next_keepalive:
                next_keepalive = wall + KEEPALIVE_PERIOD
                await self._keepalive()
            wake = [next_keepalive]
            if deadline is not None:
                wake.append(deadline)
            if self.sync_timeout is not None and deadline is None:
                wake += [lk.pending_since + self.sync_timeout for lk in waiting]
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), max(0.0, min(wake) - time.monotonic()))
            except TimeoutError:
                pass

    async def _physics(self) -> None:
        """Step physics 1 ms at a time: sync the robots due, wait for them as the mode says, step. In real time keep
        pace with the wall clock; robots not in lockstep get their sensors every SENSOR_PERIOD."""
        w = self.world
        wall0, sim0 = time.monotonic(), w.time
        last_sensor = -math.inf
        next_keepalive = 0.0
        while not self.done.is_set():
            if self.starting():
                # fast: the run waits for every robot to engage, so none starts behind the others
                if time.monotonic() >= next_keepalive:
                    next_keepalive = time.monotonic() + KEEPALIVE_PERIOD
                    await self._keepalive()
                await asyncio.sleep(0.01)
                wall0, sim0 = time.monotonic(), w.time
                continue
            now = self.now_us
            realtime = self.realtime()
            if realtime:
                lag = (time.monotonic() - wall0) - (w.time - sim0)
                if lag > MAX_LAG:
                    log.warning("physics fell %.0f ms behind real time; skipping ahead", 1000 * lag)
                    self.physics_skipped += lag
                    wall0 = time.monotonic() - (w.time - sim0)
            if await self._sync_due(now):
                await self._wait_answers(now, wall0 + (w.time - sim0) + MS / 1e6 if realtime else None)
            if w.time - last_sensor >= SENSOR_PERIOD - 1e-9:
                last_sensor = w.time
                for name, link in self.links.items():
                    if link.ws is not None and not link.engaged:
                        await self._push_sensors(name)
            # on the event loop: a worker thread for each millisecond's step ran a fast run 3.5 times slower, for
            # no less lateness on the loop (scr.step_thread, 2026-11-24)
            hals = self.hals()
            for _ in range(self.steps_per_ms):
                w.step(hals)
            if realtime:
                ahead = (w.time - sim0) - (time.monotonic() - wall0)
                await asyncio.sleep(ahead if ahead > 0 else 0)
            else:
                wall0, sim0 = time.monotonic(), w.time   # so a change to real time starts pacing from here
                await asyncio.sleep(0)

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
                    if link.engaged:
                        await self._send(link, [frame(*LOCKSTEP_DEVICE, {">engage": False})])  # free its clock
                    await link.ws.close()
