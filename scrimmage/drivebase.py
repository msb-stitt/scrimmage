"""A plugin that reports a drivebase's motion from MuJoCo's truth, and places it where the robot program sets its pose.

For a robot program that reads its odometry from the simulation rather than from simulated encoders, such as an FTC
robot whose Pinpoint and localizer report where the robot is. Named in a robot's config:

    [[plugin]]
    class = "scrimmage.drivebase:Drivebase"
    device = "drive"
    wheels = ["fl_joint", "fr_joint", "bl_joint", "br_joint"]
    wheel_radius = 0.052

Each step it sends ScrimmageDrivebase <device>, in the field's frame, metres, radians and seconds:

- >x, >y, >heading: the robot body's pose;
- >vx, >vy, >omega: its velocity along itself and to its left, and its turn;
- >turned: its heading unwrapped since the start or the last placing;
- >rolled<i>: how far wheel i has rolled at `wheel_radius`, forward positive, since the start;
- >travel_x, >travel_y: its travel along itself and to its left, summed over the steps;
- >place_seq: the last placing taken.

A robot program places the robot by sending <place_x, <place_y, <place_heading and a <place_seq above the last one
taken: the body is put there at rest before the next step, its height and the wheels' turns as they were.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco

from .bindings import Update
from .i18n import t
from .protocol import HalState

TYPE = "ScrimmageDrivebase"


class Drivebase:
    def __init__(self, world: Any, device: str, wheels: list[str], wheel_radius: float):
        self.m, self.d, self.device = world.model, world.data, str(device)
        self.body = world.robot_body
        if world.free_qadr is None:
            raise ValueError(t("drivebase {device}: the robot body has no free joint", device=device))
        self.qadr, self.dadr = world.free_qadr, world.free_dadr
        self.radius = float(wheel_radius)
        self.wheels: list[tuple[int, float]] = []
        mujoco.mj_forward(self.m, self.d)
        left = self.d.xmat[self.body].reshape(3, 3)[:, 1]
        for name in wheels:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(t("drivebase {device}: no joint {joint}", device=device, joint=repr(name)))
            # a wheel rolling forward turns about the robot's left
            self.wheels.append((int(self.m.jnt_qposadr[j]), 1.0 if float(self.d.xaxis[j] @ left) > 0 else -1.0))
        self.reset()

    def reset(self) -> None:
        self.rolled0 = [s * float(self.d.qpos[q]) for q, s in self.wheels]
        self.travel = [0.0, 0.0]
        self.turned = 0.0
        self.heading = self._pose()[2]
        self.seq = 0

    def _pose(self) -> tuple[float, float, float]:
        r = self.d.xmat[self.body].reshape(3, 3)
        return float(self.d.xpos[self.body][0]), float(self.d.xpos[self.body][1]), math.atan2(r[1, 0], r[0, 0])

    def _twist(self) -> tuple[float, float, float]:
        v = self.d.xmat[self.body].reshape(3, 3).T @ self.d.qvel[self.dadr:self.dadr + 3]
        return float(v[0]), float(v[1]), float(self.d.qvel[self.dadr + 5])

    def before_step(self, hal: HalState) -> None:
        dev = hal.devices.get((TYPE, self.device), {})
        seq = dev.get("<place_seq")
        if isinstance(seq, (int, float)) and int(seq) > self.seq:
            x, y, h = (float(dev.get(k, 0.0)) for k in ("<place_x", "<place_y", "<place_heading"))
            self.d.qpos[self.qadr:self.qadr + 2] = (x, y)
            self.d.qpos[self.qadr + 3:self.qadr + 7] = (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))
            self.d.qvel[self.dadr:self.dadr + 6] = 0.0
            mujoco.mj_forward(self.m, self.d)
            self.seq, self.heading = int(seq), h

    def after_step(self, hal: HalState) -> None:
        vx, vy, _ = self._twist()
        dt = self.m.opt.timestep
        self.travel[0] += vx * dt
        self.travel[1] += vy * dt
        h = self._pose()[2]
        self.turned += math.remainder(h - self.heading, 2 * math.pi)
        self.heading = h

    def sensor_updates(self, hal: HalState) -> list[Update]:
        x, y, h = self._pose()
        vx, vy, w = self._twist()
        out: dict[str, Any] = {">x": x, ">y": y, ">heading": h, ">vx": vx, ">vy": vy, ">omega": w,
                               ">turned": self.turned, ">travel_x": self.travel[0], ">travel_y": self.travel[1],
                               ">place_seq": self.seq}
        for i, ((q, s), r0) in enumerate(zip(self.wheels, self.rolled0, strict=True)):
            out[f">rolled{i}"] = (s * float(self.d.qpos[q]) - r0) * self.radius
        return [(TYPE, self.device, out)]
