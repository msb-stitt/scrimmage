"""Plugins: Python a robot's config names, run with its physics (the fork's multi-robot branch).

A robot's config names each plugin in a [[plugin]] entry: `class = "package.module:Class"`, and any other keys as the
class's keyword arguments. The class is made as Class(world, **settings), `world` the robot's World, which holds its
model, data and bindings. Each hook is optional:

- before_step(hal): before each physics timestep, after the robot's motors are driven; it may add to the data's
  qfrc_applied, xfrc_applied and ctrl.
- after_step(hal): after each physics timestep.
- reset(): when the world is reset.
- sensor_updates(hal) -> list of (type, device, data): readings for the robot program, sent with the bindings'.

Each plugin runs on every physics step, so each one's time is kept (PluginHost.seconds) to show what it costs.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from .bindings import Update
from .i18n import t
from .protocol import HalState


class PluginError(ValueError):
    pass


def load_class(path: str) -> type:
    module, _, name = path.partition(":")
    if not module or not name:
        raise PluginError(t("{plugin_table} needs {class_setting}, not {path}", plugin_table="[[plugin]]",
                              class_setting='class = "package.module:Class"', path=repr(path)))
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise PluginError(t("{plugin_table} class {path} can't be loaded: {error}",
                              plugin_table="[[plugin]]", path=repr(path), error=e)) from None


class PluginHost:
    """A robot's plugins, in config order, and the time each has taken (s)."""

    def __init__(self, entries: list[dict[str, Any]], world: Any):
        self.plugins: list[Any] = []
        for entry in entries:
            settings = dict(entry)
            path = settings.pop("class", None)
            if not isinstance(path, str):
                raise PluginError(t("each {plugin_table} needs {class_setting}", plugin_table="[[plugin]]",
                                    class_setting='class = "package.module:Class"'))
            cls = load_class(path)
            try:
                self.plugins.append(cls(world, **settings))
            except TypeError as e:
                raise PluginError(f"[[plugin]] {path}: {e}") from None
        self.seconds = [0.0] * len(self.plugins)
        self._hooks = {hook: [(i, getattr(p, hook)) for i, p in enumerate(self.plugins) if hasattr(p, hook)]
                       for hook in ("before_step", "after_step", "reset", "sensor_updates")}

    def _run(self, hook: str, *args: Any) -> list[Any]:
        out = []
        for i, fn in self._hooks[hook]:
            t0 = time.perf_counter()
            out.append(fn(*args))
            self.seconds[i] += time.perf_counter() - t0
        return out

    def before_step(self, hal: HalState) -> None:
        self._run("before_step", hal)

    def after_step(self, hal: HalState) -> None:
        self._run("after_step", hal)

    def reset(self) -> None:
        self._run("reset")

    def sensor_updates(self, hal: HalState) -> list[Update]:
        return [u for ups in self._run("sensor_updates", hal) for u in ups]
