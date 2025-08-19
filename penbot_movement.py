# penbot_movement.py

import time
import math
from adafruit_motor import stepper # type: ignore

# Project-specific imports
import penbot_constants as const
import penbot_math

# --- Module-level state for robot position and angles ---
# These are updated by functions within this module.
current_steps1_internal = 0  # Tracks Motor A's absolute step count
current_steps2_internal = 0  # Tracks Motor B's absolute step count
current_canvas_x_internal = 0.0 # Tracks current pen X on canvas
current_canvas_y_internal = 0.0 # Tracks current pen Y on canvas
prev_theta_A_deg_internal = None # Last calculated/used angle for Motor A
prev_theta_B_deg_internal = None # Last calculated/used angle for Motor B
# Note: motor_enabled status is managed by the main driver and checked before calling move functions.

def _debug_log_movement(message):
    """Helper for debug logging specific to this module."""
    if const.DEBUG_MOVEMENT:
        print(f"[MOVE][{time.strftime('%H:%M:%S')}] {message}")

def _execute_dda_steps(kit_obj, delta_steps_A, delta_steps_B):
    """
    Moves motors by a specific number of delta steps, coordinating them using DDA.
    This is the core function for simultaneous motor stepping.
    kit_obj: The MotorKit instance for the X-Y motors.
    delta_steps_A: Number of steps for motor 1 (Motor A).
    delta_steps_B: Number of steps for motor 2 (Motor B).
    """
    # Determine direction based on sign of delta_steps.
    # Adafruit library: BACKWARD for positive steps in some contexts, FORWARD for negative.
    # This needs to align with how STEPS_PER_DEGREE and offsets are calculated.
    # Assuming:
    # - Positive delta_steps_A means motor A should turn in the direction that DECREASES its angle (CW if angle is CCW).
    # - Positive delta_steps_B means motor B should turn in the direction that INCREASES its angle (CCW).
    # This matches the step calculation:
    # target_A = OFFSET_A - angle_A * STEPS_DEGREE  => d(target_A)/d(angle_A) is negative
    # target_B = OFFSET_B + angle_B * STEPS_DEGREE  => d(target_B)/d(angle_B) is positive

    # If delta_A > 0, it means target_motor_stepsA > current_steps1.
    # If target_motor_stepsA = OFFSET_A - angle_A_deg * SPD,
    # then an increase in steps means a decrease in angle_A_deg.
    # So, positive delta_A should correspond to stepper.FORWARD if FORWARD moves towards smaller angles.
    # Let's assume stepper.BACKWARD increases step count, stepper.FORWARD decreases.
    # The original code used: dir = stepper.BACKWARD if delta_steps > 0
    # This means positive delta_steps (increase in step count) used BACKWARD.

    dir1 = stepper.BACKWARD if delta_steps_A > 0 else stepper.FORWARD
    dir2 = stepper.BACKWARD if delta_steps_B > 0 else stepper.FORWARD

    num_steps_A = abs(delta_steps_A)
    num_steps_B = abs(delta_steps_B)
    max_segment_steps = max(num_steps_A, num_steps_B)

    if max_segment_steps == 0:
        return

    dda_taken_A, dda_taken_B = 0, 0
    # The DDA loop:
    # In each iteration, it decides if motor A, motor B, or both should take a microstep.
    # The time.sleep(const.DDA_STEP_DELAY) is the delay *per DDA iteration*,
    # not per individual motor step if only one motor moves in that iteration.
    # This loop should execute very quickly.
    for i_dda in range(max_segment_steps):
        step_A_now = False
        step_B_now = False

        if num_steps_A > 0 and ((i_dda * num_steps_A) / max_segment_steps > dda_taken_A):
            step_A_now = True
            dda_taken_A += 1

        if num_steps_B > 0 and ((i_dda * num_steps_B) / max_segment_steps > dda_taken_B):
            step_B_now = True
            dda_taken_B += 1

        # Issue steps (almost) simultaneously if both are due
        if step_A_now:
            kit_obj.stepper1.onestep(direction=dir1, style=stepper.MICROSTEP)
        if step_B_now:
            kit_obj.stepper2.onestep(direction=dir2, style=stepper.MICROSTEP)

        # Only sleep if at least one motor was supposed to step or if a delay is always wanted
        if (step_A_now or step_B_now) and const.DDA_STEP_DELAY > 0:
            time.sleep(const.DDA_STEP_DELAY)
        elif const.DDA_STEP_DELAY > 0 and max_segment_steps > 0 : # Ensure delay even if one motor has 0 steps for segment
             time.sleep(const.DDA_STEP_DELAY)


def initialize_robot_state_movement(initial_canvas_x, initial_canvas_y):
    """
    Sets the initial motor step counts and angles based on starting canvas coordinates.
    Updates internal module state.
    """
    global current_steps1_internal, current_steps2_internal
    global current_canvas_x_internal, current_canvas_y_internal
    global prev_theta_A_deg_internal, prev_theta_B_deg_internal

    current_canvas_x_internal = initial_canvas_x
    current_canvas_y_internal = initial_canvas_y

    robot_Px, robot_Py = penbot_math.canvas_to_robot(initial_canvas_x, initial_canvas_y)
    _debug_log_movement(f"Initializing state. Canvas Home: ({initial_canvas_x:.2f}, {initial_canvas_y:.2f}) -> Robot Coords: ({robot_Px:.2f}, {robot_Py:.2f})")

    ik_solutions = penbot_math.calculate_inverse_kinematics(robot_Px, robot_Py)

    if not ik_solutions:
        _debug_log_movement(f"CRITICAL INIT: No IK solution for Robot({robot_Px:.2f}, {robot_Py:.2f}). Steps set to 0,0 (likely wrong).")
        current_steps1_internal = 0
        current_steps2_internal = 0
        prev_theta_A_deg_internal = None
        prev_theta_B_deg_internal = None
        return False # Indicate failure

    selected_sol = penbot_math.select_solution(ik_solutions, None, None,
                                               const.PREFERRED_PE_ID, const.PREFERRED_ELBOW_B_SIGN)
    if not selected_sol:
        _debug_log_movement(f"CRITICAL INIT: Could not select IK solution for Robot({robot_Px:.2f}, {robot_Py:.2f}). Steps set to 0,0.")
        current_steps1_internal = 0
        current_steps2_internal = 0
        prev_theta_A_deg_internal = None
        prev_theta_B_deg_internal = None
        return False # Indicate failure

    initial_theta_A_deg = selected_sol['theta_A']
    initial_theta_B_deg = selected_sol['theta_B']

    prev_theta_A_deg_internal = initial_theta_A_deg
    prev_theta_B_deg_internal = initial_theta_B_deg

    current_steps1_internal = const.OFFSET_A_STEPS - round(initial_theta_A_deg * const.STEPS_PER_DEGREE)
    current_steps2_internal = const.OFFSET_B_STEPS + round(initial_theta_B_deg * const.STEPS_PER_DEGREE)

    _debug_log_movement(f"State initialized. Angles: A={initial_theta_A_deg:.2f}°, B={initial_theta_B_deg:.2f}°")
    _debug_log_movement(f"  (Using Pe_id={selected_sol['Pe_id']}, ElbowB_sign={selected_sol['elbow_B_sign']})")
    _debug_log_movement(f"  Initial Motor Steps: A={current_steps1_internal}, B={current_steps2_internal}")
    return True

def move_to_canvas_coordinates(kit_obj, target_x_canvas, target_y_canvas,
                               segment_length_mm=const.DEFAULT_SEGMENT_LENGTH_MM):
    """
    Moves the robot arm to specified canvas (x, y) in a linear path.
    kit_obj: The MotorKit instance for X-Y motors.
    Updates internal module state.
    Returns True on success, False on failure (e.g., unreachable point).
    """
    global current_steps1_internal, current_steps2_internal
    global current_canvas_x_internal, current_canvas_y_internal
    global prev_theta_A_deg_internal, prev_theta_B_deg_internal

    _debug_log_movement(f"Move from canvas ({current_canvas_x_internal:.2f}, {current_canvas_y_internal:.2f}) to ({target_x_canvas:.2f}, {target_y_canvas:.2f})")

    dx_total_canvas = target_x_canvas - current_canvas_x_internal
    dy_total_canvas = target_y_canvas - current_canvas_y_internal
    distance_canvas = math.sqrt(dx_total_canvas**2 + dy_total_canvas**2)

    if distance_canvas < 1e-3: # Effectively zero distance
        _debug_log_movement("Target is current position. No move needed.")
        return True

    num_segments = max(1, int(math.ceil(distance_canvas / segment_length_mm)))
    _debug_log_movement(f"Path divided into {num_segments} segments of ~{segment_length_mm:.2f}mm.")

    start_canvas_x_for_move = current_canvas_x_internal
    start_canvas_y_for_move = current_canvas_y_internal

    for i_segment in range(1, num_segments + 1):
        ratio = i_segment / float(num_segments)
        intermediate_canvas_x = start_canvas_x_for_move + ratio * dx_total_canvas
        intermediate_canvas_y = start_canvas_y_for_move + ratio * dy_total_canvas

        Px_robot, Py_robot = penbot_math.canvas_to_robot(intermediate_canvas_x, intermediate_canvas_y)
        ik_solutions = penbot_math.calculate_inverse_kinematics(Px_robot, Py_robot)

        if not ik_solutions:
            _debug_log_movement(f"  ERROR Seg {i_segment}: No IK for Canvas({intermediate_canvas_x:.2f}, {intermediate_canvas_y:.2f}) -> Robot({Px_robot:.2f}, {Py_robot:.2f})")
            prev_theta_A_deg_internal, prev_theta_B_deg_internal = None, None # Break continuity
            return False

        selected_sol = penbot_math.select_solution(ik_solutions,
                                                   prev_theta_A_deg_internal, prev_theta_B_deg_internal,
                                                   const.PREFERRED_PE_ID, const.PREFERRED_ELBOW_B_SIGN)
        if not selected_sol:
            _debug_log_movement(f"  ERROR Seg {i_segment}: Could not select IK for Canvas({intermediate_canvas_x:.2f}, {intermediate_canvas_y:.2f})")
            prev_theta_A_deg_internal, prev_theta_B_deg_internal = None, None
            return False

        theta_A_deg = selected_sol['theta_A']
        theta_B_deg = selected_sol['theta_B']

        target_motor_stepsA_at_inter = const.OFFSET_A_STEPS - round(theta_A_deg * const.STEPS_PER_DEGREE)
        target_motor_stepsB_at_inter = const.OFFSET_B_STEPS + round(theta_B_deg * const.STEPS_PER_DEGREE)

        delta_A_for_segment = target_motor_stepsA_at_inter - current_steps1_internal
        delta_B_for_segment = target_motor_stepsB_at_inter - current_steps2_internal

        # Periodic logging for long moves
        if const.DEBUG_MOVEMENT and (i_segment % max(1, num_segments // 10) == 0 or i_segment == 1 or i_segment == num_segments):
             _debug_log_movement(f"  Seg {i_segment}/{num_segments}: Canvas({intermediate_canvas_x:.2f}, {intermediate_canvas_y:.2f})")
             _debug_log_movement(f"    Angles: A={theta_A_deg:.2f}°, B={theta_B_deg:.2f}° (Pe_id={selected_sol['Pe_id']}, ElB_s={selected_sol['elbow_B_sign']})")
             _debug_log_movement(f"    Target Steps Abs: A={target_motor_stepsA_at_inter}, B={target_motor_stepsB_at_inter}")
             _debug_log_movement(f"    Delta Steps: dA={delta_A_for_segment}, dB={delta_B_for_segment}")

        _execute_dda_steps(kit_obj, delta_A_for_segment, delta_B_for_segment)

        current_steps1_internal = target_motor_stepsA_at_inter
        current_steps2_internal = target_motor_stepsB_at_inter
        prev_theta_A_deg_internal = theta_A_deg
        prev_theta_B_deg_internal = theta_B_deg

    current_canvas_x_internal = target_x_canvas
    current_canvas_y_internal = target_y_canvas

    _debug_log_movement(f"Movement to canvas ({target_x_canvas:.2f}, {target_y_canvas:.2f}) complete.")
    _debug_log_movement(f"  Final Steps: A={current_steps1_internal}, B={current_steps2_internal}. Angles A={prev_theta_A_deg_internal:.1f}, B={prev_theta_B_deg_internal:.1f}")
    return True

def lift_pen_movement(lift_kit_obj, target_z_state, current_z_state):
    """
    Controls the pen lift mechanism.
    lift_kit_obj: The MotorKit instance for the Z motor.
    target_z_state: 1 for up, 0 for down.
    current_z_state: The current known state of the pen.
    Returns the new Z state.
    """

    if target_z_state > current_z_state: # Lift UP
        _debug_log_movement("Lifting pen UP.")
        diff = target_z_state - current_z_state
        for _ in range(diff):
            lift_kit_obj.stepper1.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE)
            time.sleep(const.LIFT_STEP_DELAY)
        _debug_log_movement("Pen is UP.")
        return current_z_state + diff
    else: # Lower DOWN
        diff = current_z_state - target_z_state
        _debug_log_movement("Lowering pen DOWN.")
        for _ in range(diff):
            lift_kit_obj.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
            time.sleep(const.LIFT_STEP_DELAY)
        _debug_log_movement("Pen is DOWN.")
        return current_z_state - diff
    _debug_log_movement(f"Pen already at desired Z state: {current_z_state}.")

    return current_z_state # No change

# --- Getter functions for main driver to access internal state if needed ---
def get_current_canvas_position():
    return current_canvas_x_internal, current_canvas_y_internal

def get_current_motor_steps():
    return current_steps1_internal, current_steps2_internal

def get_current_motor_angles():
    return prev_theta_A_deg_internal, prev_theta_B_deg_internal

