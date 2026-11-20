"""Every arm number: gripper geometry, speeds, and the robot model."""

from pathlib import Path

ROBOT = (Path(__file__).parent / "iiwa_with_gripper.yaml").read_text()  # model directives

# Gripper geometry, from the WSG model: the fingers slide along G's x axis and
# reach out along G's +y axis; the 20 mm pads are centered 108 mm out.
GRIPPER_REACH = 0.108
PAD_HALF_LENGTH = 0.010
FINGER_THICKNESS = 0.015    # finger + tip, along the closing direction
FINGER_WIDTH = 0.020        # finger extent across the closing direction
FINGER_OPEN_MARGIN = 0.02   # opening beyond the brick width on the way in
FINGER_SQUEEZE = 0.02       # commanded closing past the brick width
BODY = (0.073, 0.025, 0.09) # gripper body: half length along x, half across, height

PATH_STEP = 0.005   # IK sample spacing along the gripper path (m)
SPEED = 0.05        # gripper speed (m/s)
TURN_SPEED = 0.5    # gripper turning speed (rad/s)
HOLD_SECONDS = 1.0  # time given to the fingers to open or close
