# robot_driver.py

import time
import atexit
import os
import sys
from adafruit_motorkit import MotorKit

# Project-specific imports
import penbot_constants as const
import penbot_movement

# --- Global Variables for Driver ---
kit_xy = None
kit_lift = None
current_z_driver = 0
motors_enabled_driver = False

# Home position (canvas coordinates)
HOME_CANVAS_X = const.CANVAS_HOME_X
HOME_CANVAS_Y = const.CANVAS_HOME_Y

def _debug_log_driver(message):
    if const.DEBUG_SERIAL:
        print(f"[DRIVER][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def _initialize_motor_kits():
    global kit_xy, kit_lift
    # This function is now idempotent, safe to call multiple times.
    if kit_xy is None or kit_lift is None:
        try:
            kit_xy = MotorKit(address=0x61)
            kit_lift = MotorKit(address=0x62)
            _debug_log_driver("MotorKits initialized (XY at 0x61, Lift at 0x62).")
        except Exception as e:
            print(f"[CRITICAL] Error initializing MotorKits: {e}")
            kit_xy, kit_lift = None, None # Ensure they are None on failure
            return False
    return True

def _cleanup_resources():
    _debug_log_driver("Executing cleanup: Releasing motors.")
    global motors_enabled_driver
    if motors_enabled_driver:
        _disable_motors_driver_cmd()
    motors_enabled_driver = False
    _debug_log_driver("Cleanup finished.")

def _enable_motors_driver_cmd():
    global motors_enabled_driver
    motors_enabled_driver = True
    _debug_log_driver("Motors ENABLED by command.")

def _disable_motors_driver_cmd():
    global motors_enabled_driver
    motors_enabled_driver = False
    if kit_xy:
        if hasattr(kit_xy, 'stepper1'): kit_xy.stepper1.release()
        if hasattr(kit_xy, 'stepper2'): kit_xy.stepper2.release()
    if kit_lift:
        if hasattr(kit_lift, 'stepper1'): kit_lift.stepper1.release()
    _debug_log_driver("Motors DISABLED and released.")

def _process_command(command_line):
    global current_z_driver
    parts = command_line.strip().split(',')
    if not parts or not parts[0]: return
    cmd = parts[0].upper()

    try:
        if cmd == "PBM":
            delay_ms, x_c, y_c = int(parts[1]), float(parts[2]), float(parts[3])
            _debug_log_driver(f"PBM: Target Canvas X={x_c}, Y={y_c}")
            if motors_enabled_driver and kit_xy:
                penbot_movement.move_to_canvas_coordinates(kit_xy, x_c, y_c)
                if delay_ms > 0: time.sleep(delay_ms / 1000.0)
        elif cmd == "PBZ":
            z_target = int(parts[1])
            _debug_log_driver(f"PBZ: Target Z Steps={z_target}")
            if kit_lift:
                current_z_driver = penbot_movement.lift_pen_movement(kit_lift, z_target, current_z_driver)
        elif cmd == "PBR":
            state = int(parts[1])
            if state == 1: _enable_motors_driver_cmd()
            elif state == 0: _disable_motors_driver_cmd()
        elif cmd == "PBW":
            delay_w = int(parts[1])
            if delay_w > 0: time.sleep(delay_w / 1000.0)
    except (ValueError, IndexError) as e:
        _debug_log_driver(f"Invalid command format or value for '{command_line}': {e}")
    except Exception as e:
        _debug_log_driver(f"Error during command processing '{command_line}': {e}")

def run_drawing_from_file(filepath, stop_event, progress_callback=None):
    """
    Initializes the robot, executes all commands from a .pb file, and then cleans up.
    This function is designed to be run in a worker thread.
    Args:
        filepath (str): Path to the .pb file.
        stop_event (threading.Event): An event from the main thread to signal a stop.
        progress_callback (function, optional): Callback for reporting progress.
    """
    # --- Setup ---
    # Register cleanup to run when the program exits, regardless of how.
    atexit.register(_cleanup_resources)

    if not _initialize_motor_kits():
        return False
    
    if not penbot_movement.initialize_robot_state_movement(HOME_CANVAS_X, HOME_CANVAS_Y):
        _debug_log_driver("CRITICAL: Failed to initialize robot state.")
        _cleanup_resources()
        return False

    if not os.path.exists(filepath):
        _debug_log_driver(f"Error: Command file not found: {filepath}")
        return False

    # --- Execution ---
    try:
        with open(filepath, 'r') as f:
            commands = f.readlines()
        
        # Filter out empty lines and comments for accurate progress
        commands = [cmd for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]
        total_commands = len(commands)
        
        _debug_log_driver(f"Starting execution of {total_commands} commands from {filepath}.")
        if progress_callback:
            progress_callback(0, total_commands) # Initial state

        for i, line in enumerate(commands):
            # Check the stop event from the main thread at the start of each command
            if stop_event.is_set():
                _debug_log_driver("Stop event detected, halting execution.")
                break
            
            _process_command(line)
            if progress_callback:
                progress_callback(i + 1, total_commands)
        
        if not stop_event.is_set():
            _debug_log_driver("Finished processing all file commands normally.")

    except Exception as e:
        _debug_log_driver(f"An error occurred during file execution: {e}")
        return False
    finally:
        # --- Teardown (always runs) ---
        _debug_log_driver("Executing final driver cleanup sequence.")
        # Ensure pen is lifted and motors are disabled as a final safety measure.
        if kit_lift:
             _process_command(f"PBZ,{const.LIFT_STEPS}")
        if motors_enabled_driver:
            _disable_motors_driver_cmd()
        _debug_log_driver("Drawing process thread has finished.")

    return True

if __name__ == '__main__':
    # A simple standalone test for this module
    print("--- Robot Driver Standalone Test ---")
    TEST_FILE = "test_square.pb"
    with open(TEST_FILE, 'w') as f:
        f.write("# Test square\nPBR,1\n"); f.write(f"PBZ,{const.LIFT_STEPS}\n")
        f.write("PBM,0,80,20\n"); f.write("PBZ,0\n"); f.write("PBM,0,110,20\n")
        f.write("PBM,0,110,50\n"); f.write("PBM,0,80,50\n"); f.write("PBM,0,80,20\n")
        f.write(f"PBZ,{const.LIFT_STEPS}\n"); f.write(f"PBM,0,{HOME_CANVAS_X},{HOME_CANVAS_Y}\n")
        f.write("PBR,0\n")

    def my_test_callback(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

    # Create a dummy stop event for testing
    stop_ev = threading.Event()
    
    print(f"Running test file: {TEST_FILE}. Press Ctrl+C to test stop event.")
    
    # Simulate how main.py would use it
    def test_signal_handler(sig, frame):
        print("\nTest Ctrl+C caught! Setting stop event.")
        stop_ev.set()
    
    signal.signal(signal.SIGINT, test_signal_handler)

    success = run_drawing_from_file(TEST_FILE, stop_ev, my_test_callback)

    if success: print("Test completed.")
    else: print("Test failed.")
    
    os.remove(TEST_FILE)