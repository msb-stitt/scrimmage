"""Plugins for tests/test_plugins.py."""

import mujoco


class Push:
    """Pushes a body along +x with a constant force (N) before each step."""

    def __init__(self, world, body: str, force: float):
        self.d = world.data
        self.body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.force = float(force)

    def before_step(self, hal):
        self.d.xfrc_applied[self.body, 0] = self.force


class Recorder:
    """Records its hooks' calls, and reports how many steps it has seen as SimDevice <device> >steps."""

    def __init__(self, world, device: str):
        self.device, self.calls, self.steps = device, [], 0

    def before_step(self, hal):
        self.calls.append("before")

    def after_step(self, hal):
        self.calls.append("after")
        self.steps += 1

    def reset(self):
        self.calls.append("reset")
        self.steps = 0

    def sensor_updates(self, hal):
        return [("SimDevice", self.device, {">steps": self.steps})]
