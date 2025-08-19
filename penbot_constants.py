# penbot_constants.py

import math

# --- Debugging Flags ---
DEBUG_SERIAL = True
DEBUG_MOVEMENT = True

# --- Motor Configuration ---
STEPS_PER_REVOLUTION_RAW = 200  # Steps per revolution for the motor itself (e.g., 1.8 deg/step)
MICROSTEPPING = 16  # Microsteps per full step (e.g., 1, 2, 4, 8, 16, 32)
GEAR_RATIO = 3.0  # Gearbox ratio (e.g., 3:1 means 3 motor turns for 1 output turn)

# --- Canvas Size and offset in mm ---
CANVAS_WIDTH=130 # 6inx4in note cards
CANVAS_HEIGHT=100
CANVAS_OFFSET_X=-38
CANVAS_OFFSET_Y=-80

# --- Canvas Home Position in mm ---
# These are the canvas coordinates where the pen is physically located when the script starts
# AND before any homing or explicit moves.
CANVAS_HOME_X=38.8
CANVAS_HOME_Y=-17.4

# Total effective steps for one full revolution of the output shaft
# Ensure this is an integer as it represents a discrete number of steps.
STEPS_PER_REVOLUTION = int(STEPS_PER_REVOLUTION_RAW * MICROSTEPPING * GEAR_RATIO)

# STEPS_PER_DEGREE can remain float for precision in angle calculations
STEPS_PER_DEGREE = STEPS_PER_REVOLUTION / 360.0

# --- Inverse Kinematics Preferred Solution ---
# These were found to give a good starting configuration in tests.
# From penbot_math.py, refers to the order/sign of solutions.
PREFERRED_PE_ID = 1
PREFERRED_ELBOW_B_SIGN = -1 # Elbow "down" for motor B's linkage

# --- Movement Parameters ---
# Delay for each DDA iteration in _execute_dda_steps.
# This is critical for smoothness and speed.
# Smaller values = faster, but can overload CPU or miss steps if too small.
# Typical values for microstepping might be 0.00005 to 0.0002 (50µs to 200µs)
DDA_STEP_DELAY = 0.0005 # (0.1 ms) - Tune this carefully!

DEFAULT_SEGMENT_LENGTH_MM = 1.0  # Length of small line segments for path interpolation (mm)

# --- Lift Mechanism ---
LIFT_STEPS = 450  # Number of steps to lift/lower pen (min 200 as requested)
LIFT_STEP_DELAY = 0.0001  # Delay between steps for the lift motor (3 ms)

# --- Robot Zero Angle to Motor Step Offsets ---
# These define the motor's absolute step count that corresponds to a calculated angle of 0 degrees
# from the IK solver. This is crucial for calibration.
# Example: If motor A's physical 0 degree (e.g., arm pointing right) should correspond to
# a calculated IK angle of 0, and your homing sets steps to 0 at this physical point,
# then OFFSET_A_STEPS would be 0.
# If, however, your physical home sets steps to 0 when Motor A is at +90 degrees physically,
# but your IK's 0 degree angle is defined differently, you adjust this.
# The previous code used:
# offset_A_steps = STEPS_PER_REVOLUTION // 4  (implies 0 deg angle = +90 deg physical position for motor A)
# offset_B_steps = -STEPS_PER_REVOLUTION // 2 (implies 0 deg angle = -180 deg physical position for motor B)
# These need to be verified against your physical setup and homing procedure.
# Let's assume for now that 0 steps corresponds to 0 degrees angle from IK.
# If your physical "home" (where step counters are zeroed) is NOT where the arms
# are at their mathematical zero-angle positions, these offsets bridge that gap.
# To match previous logic:
OFFSET_A_STEPS = STEPS_PER_REVOLUTION // 4
OFFSET_B_STEPS = -STEPS_PER_REVOLUTION // 2
# Note on step direction vs angle:
# The movement module will implement:
# target_steps_A = OFFSET_A_STEPS - round(theta_A_deg * STEPS_PER_DEGREE)
# target_steps_B = OFFSET_B_STEPS + round(theta_B_deg * STEPS_PER_DEGREE)
# This implies:
# For Motor A: Increasing angle (CCW) DECREASES step count from its offset.
# For Motor B: Increasing angle (CCW) INCREASES step count from its offset.
# This must match your motor wiring and direction settings in execute_dda_steps.

# --- Canvas to Robot Coordinate Transformation (from penbot_math.py, duplicated for clarity if needed) ---
# These are based on the origin of the robot (motors' center)
# and the position of the canvas's top-left corner relative to that robot origin.
# ROBOT_X_FOR_CANVAS_ORIGIN = -110.0 # Robot X for canvas x_c = 0
# ROBOT_Y_FOR_CANVAS_ORIGIN_TOP_EDGE = -90.0 # Robot Y for the top edge of canvas (canvas y_c=0)
# These are actually defined and used within penbot_math.py, so no need to duplicate here
# if penbot_math is self-contained for its coordinate transformations.

# --- Logging Function (optional, can be centralized) ---
def debug_log_constants(message, source="CONSTANTS"):
    """Basic debug logger."""
    if DEBUG_MOVEMENT or DEBUG_SERIAL: # Or a specific DEBUG_CONSTANTS flag
        print(f"[{source}][{__import__('time').strftime('%H:%M:%S')}] {message}")

if __name__ == '__main__':
    debug_log_constants(f"STEPS_PER_REVOLUTION: {STEPS_PER_REVOLUTION}")
    debug_log_constants(f"STEPS_PER_DEGREE: {STEPS_PER_DEGREE}")
    debug_log_constants(f"DDA_STEP_DELAY: {DDA_STEP_DELAY} s")
    debug_log_constants(f"LIFT_STEPS: {LIFT_STEPS}")
