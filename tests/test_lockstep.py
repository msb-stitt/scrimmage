import pytest

from scrimmage.lockstep import LOCKSTEP_DEVICE, LockstepDriver, Pacer, parse_speed, physics_steps_per_step
from scrimmage.protocol import HalState, frame


def test_parse_speed():
    assert parse_speed("max") is None
    assert parse_speed(" MAX ") is None
    assert parse_speed("1") == 1.0
    assert parse_speed("2.5") == 2.5
    for bad in ["0", "-1", "fast", "nan", "inf", ""]:
        with pytest.raises(ValueError):
            parse_speed(bad)


def test_physics_steps_per_step():
    assert physics_steps_per_step(0.005, 0.001) == 5
    assert physics_steps_per_step(0.02, 0.002) == 10
    assert physics_steps_per_step(0.001, 0.001) == 1
    assert physics_steps_per_step(0.0, 0.001) == 1   # 0: synced at the robot's alarms, physics by the millisecond
    for step in [0.0025, 0.0005]:
        with pytest.raises(ValueError, match="whole multiple"):
            physics_steps_per_step(step, 0.001)


def test_pacer_holds_sim_time_to_wall_time_times_speed():
    wall = [100.0]
    p = Pacer(2.0, clock=lambda: wall[0])
    p.start(10.0)
    assert p.delay(10.0) == 0.0
    assert p.delay(11.0) == pytest.approx(0.5)     # 1 s of sim at 2x = 0.5 s of wall
    wall[0] = 100.5
    assert p.delay(11.0) == 0.0
    wall[0] = 101.0
    assert p.delay(11.0) == 0.0                    # behind: never negative


def test_pacer_max_never_waits():
    p = Pacer(None, clock=lambda: 0.0)
    p.start(0.0)
    assert p.delay(1000.0) == 0.0


def test_pacer_lag_is_how_far_wall_time_is_behind():
    wall = [100.0]
    p = Pacer(2.0, clock=lambda: wall[0])
    p.start(10.0)
    assert p.lag(11.0) == pytest.approx(-0.5)      # ahead: 0.5 s of wall still to go
    wall[0] = 100.5
    assert p.lag(11.0) == pytest.approx(0.0)
    wall[0] = 102.0
    assert p.lag(11.0) == pytest.approx(1.5)       # behind


class _AlarmBridge:
    def __init__(self, next_us):
        self.hal = HalState()
        if next_us is not None:
            self.hal.apply(frame(*LOCKSTEP_DEVICE, {"<next_us": next_us}))


@pytest.mark.parametrize("last, next_us, want", [
    (0, None, 1000),             # no alarm heard yet: one millisecond on
    (1000, 20001, 21000),        # an alarm rounds up to its millisecond
    (20000, 21000, 21000),       # an alarm on a millisecond is synced at it
    (21000, 21500, 22000),
    (40000, 2**63 - 1, 140000),  # no alarm due: synced again ALARM_MAX_GAP_US on
])
def test_a_step_of_zero_syncs_at_the_next_alarm_rounded_up_to_the_millisecond(last, next_us, want):
    d = LockstepDriver(_AlarmBridge(next_us), conn=0, max_lag=1.0)
    d.seq = last
    assert d._next_sync_us() == want
