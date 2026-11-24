# Several robots on one field (the fork's multi-robot branch)

**Status: controlling.**

The `multi-robot` branch of [msb-stitt/scrimmage](https://github.com/msb-stitt/scrimmage) runs
several robot programs on one MuJoCo field, each its own halsim_ws client, stepped together. It
sits beside the `Bridge`, which still runs one robot with its Driver Station, viewer and checks.

## What runs where

- **The field and its robots.** `scrimmage.multi.MultiWorld` holds one MuJoCo model and a `World`
  for each robot over it, each robot's bindings, battery and pose its own. The model is built
  outside Scrimmage, with each robot's joints and bodies named apart by a prefix.
- **The server.** `scrimmage.multi.MultiServer` serves each robot program at
  `/wpilibws/<name>`, which the program reaches by `HALSIMWS_URI`. It syncs each robot at its own
  milliseconds, in real time or as fast as the robots allow, and either waits for a robot held
  at a breakpoint or goes on without it. The module's docstring gives the rules.
- **The robot's clock.** The lockstep extension, with `step_us = 0`, moves the robot's clock to
  the time each sync gives and answers with the robot's next alarm. It builds on Linux, Windows
  x64 and macOS on Apple silicon; `lockstep/` holds it.
- **Plugins.** A robot's configuration names Python classes in `[[plugin]]` entries, run with its
  physics each step and timed. `scrimmage/plugins.py` gives the hooks, and
  `scrimmage/drivebase.py` is one: it reports a drivebase's motion from MuJoCo's truth, for a
  robot program that reads its odometry from the simulation, such as an FTC robot.
- **NetworkTables.** Each robot program serves its own on 5810. On Linux,
  `tools/robot-netns.sh` runs each in a network namespace of its own, so two programs on one
  machine do not collide there and a dashboard reads each at its own address.

## The tests

`tests/test_multi.py`, `tests/test_lockstep_time.py`, `tests/test_plugins.py`,
`tests/test_drivebase.py` and `tests/test_multi_netns.py`, which runs as root on Linux.
