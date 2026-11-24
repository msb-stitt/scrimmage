// Probe robot for stepping to a time, as tests/lockstep_time_probe/robot.py: loops of 20 ms, 21 ms, 25 ms and 40 ms,
// each reporting its count, the FPGA time it ran at and the sensor value it read, over a SimDevice.
package frc.robot;

import edu.wpi.first.hal.SimDevice;
import edu.wpi.first.hal.SimDevice.Direction;
import edu.wpi.first.hal.SimDouble;
import edu.wpi.first.hal.SimLong;
import edu.wpi.first.wpilibj.RobotController;
import edu.wpi.first.wpilibj.TimedRobot;

public class Robot extends TimedRobot {
  private static final int[] PERIODS_MS = {21, 25, 40};
  private final SimDevice dev = SimDevice.create("TimeProbe");
  private final SimDouble sensor = dev.createDouble("sensor", Direction.kInput, 0.0);
  private final java.util.Map<Integer, SimLong[]> times = new java.util.HashMap<>();
  private final java.util.Map<Integer, SimDouble> readings = new java.util.HashMap<>();

  public Robot() {
    for (int ms : new int[] {20, 21, 25, 40}) {
      // time first: when scrimmage sees a loop's count change, its time and reading are already there
      SimLong t = dev.createLong("t" + ms, Direction.kOutput, 0);
      readings.put(ms, dev.createDouble("s" + ms, Direction.kOutput, 0.0));
      times.put(ms, new SimLong[] {t, dev.createLong("n" + ms, Direction.kOutput, 0)});
    }
    for (int ms : PERIODS_MS) {
      addPeriodic(() -> loop(ms), ms / 1000.0);
    }
  }

  private void loop(int ms) {
    SimLong[] tn = times.get(ms);
    tn[0].set(RobotController.getFPGATime());
    readings.get(ms).set(sensor.get());
    tn[1].set(tn[1].get() + 1);
  }

  @Override
  public void robotPeriodic() {
    loop(20);
  }
}
