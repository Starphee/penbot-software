import math

# Robot arm lengths (mm)
R_A = 100.0  # Motor A arm length
R_B1 = 60.0  # Motor B first segment length
R_B2 = 100.0 # Motor B second segment length (to P_e)
# L_Pe_JA = 60.0 (Distance from P_e to J_A on the pen arm)
# L_JA_Pp = 110.0 (Distance from J_A to Pen tip P_p on the pen arm)
L_Pe_Pp = 170.0 # Total length from Elbow P_e to Pen P_p

import penbot_constants as const

# Canvas definition (robot coordinates for canvas origin and Y-axis direction)
ROBOT_X_FOR_CANVAS_ORIGIN = const.CANVAS_OFFSET_X # Robot X for canvas
ROBOT_Y_FOR_CANVAS_ORIGIN_TOP_EDGE = const.CANVAS_OFFSET_Y # Robot Y for the top edge of canvas (canvas y_c=0)

def canvas_to_robot(x_c, y_c):
    """
    Converts canvas coordinates (origin top-left, y-down positive)
    to robot coordinates (origin at motor center, y-up positive).
    """
    P_x_robot = x_c + ROBOT_X_FOR_CANVAS_ORIGIN
    P_y_robot = ROBOT_Y_FOR_CANVAS_ORIGIN_TOP_EDGE - y_c # Subtract y_c as canvas y is down
    return P_x_robot, P_y_robot

def calculate_inverse_kinematics(P_x_target, P_y_target):
    """
    Calculates inverse kinematics for the given pen tip target in robot coordinates.
    Returns a list of solutions, where each solution is a dictionary:
    {'theta_A': angle_A_deg, 'theta_B': angle_B_deg,
     'Pe_id': id_of_Pe_solution (0 or 1),
     'elbow_B_sign': sign_of_motor_B_internal_elbow_solution (+1 for positive acos, -1 for negative acos)}
    """
    solutions = []

    # Circle 1 for P_e (derived from Motor A's linkage and pen position)
    C1_x = (-6.0 / 11.0) * P_x_target
    C1_y = (-6.0 / 11.0) * P_y_target
    R1 = (17.0 * R_A) / 11.0

    # Circle 2 for P_e (derived from fixed distance P_e to P_p)
    C2_x = P_x_target
    C2_y = P_y_target
    R2 = L_Pe_Pp

    # Solve for intersections of Circle 1 and Circle 2 to find P_e(x_e, y_e)
    d_sq = (C2_x - C1_x)**2 + (C2_y - C1_y)**2

    # Check reachability based on distance between circle centers C1 and C2
    # Add small tolerance (1e-6) for floating point comparisons
    if d_sq > (R1 + R2)**2 + 1e-6 or d_sq < (R1 - R2)**2 - 1e-6:
        # print(f"Debug IK: Circles for P_e do not intersect. Target ({P_x_target:.1f}, {P_y_target:.1f}) likely unreachable this way.")
        return []

    d = math.sqrt(max(0, d_sq)) # max(0,..) handles tiny negative d_sq from precision errors

    if d < 1e-6: # Concentric circles
        if abs(R1 - R2) < 1e-6: # Identical circles - problematic
            return []
        else: # Concentric, different radii - no intersection
            return []

    # 'a' is distance from C1 along line C1-C2 to projection of intersection points
    a = (R1**2 - R2**2 + d_sq) / (2.0 * d)

    # Midpoint P_mid (x_m, y_m) of the common chord, on the line C1-C2
    x_m = C1_x + a * (C2_x - C1_x) / d
    y_m = C1_y + a * (C2_y - C1_y) / d

    # Half-length 'h' of the common chord
    h_sq = R1**2 - a**2
    if h_sq < -1e-6: # Circles are tangent or not intersecting, h should be ~0 or positive
        h = 0.0
    else:
        h = math.sqrt(max(0, h_sq))

    P_e_candidates = []
    # P_e solution 1 (Pe_id = 0)
    P_e_cand1_x = x_m + h * (C2_y - C1_y) / d
    P_e_cand1_y = y_m - h * (C2_x - C1_x) / d
    P_e_candidates.append({'x': P_e_cand1_x, 'y': P_e_cand1_y, 'id': 0})

    if h > 1e-6: # Only add second P_e if distinct (circles not tangent)
        # P_e solution 2 (Pe_id = 1)
        P_e_cand2_x = x_m - h * (C2_y - C1_y) / d
        P_e_cand2_y = y_m + h * (C2_x - C1_x) / d
        P_e_candidates.append({'x': P_e_cand2_x, 'y': P_e_cand2_y, 'id': 1})

    for P_e_data in P_e_candidates:
        x_e, y_e = P_e_data['x'], P_e_data['y']
        pe_id = P_e_data['id']

        # Calculate J_A (joint on Motor A arm, also on the pen linkage)
        J_Ax = (11.0 * x_e + 6.0 * P_x_target) / 17.0
        J_Ay = (11.0 * y_e + 6.0 * P_y_target) / 17.0

        # Small check for J_A distance (should be R_A by construction)
        # if not math.isclose(J_Ax**2 + J_Ay**2, R_A**2, rel_tol=1e-3):
        #     print(f"Debug IK: J_A consistency check failed slightly for Pe_id={pe_id}. Dist_sq={J_Ax**2+J_Ay**2:.2f}, R_A^2={R_A**2:.2f}")
            # This could happen due to accumulated float errors; proceed if P_e was found.

        theta_A_rad = math.atan2(J_Ay, J_Ax)

        # For Motor B: P_e is the target for its 2-link arm (R_B1, R_B2)
        D_e_sq = x_e**2 + y_e**2
        D_e = math.sqrt(max(0, D_e_sq))

        min_reach_B = abs(R_B1 - R_B2)
        max_reach_B = R_B1 + R_B2
        if not (min_reach_B - 1e-6 <= D_e <= max_reach_B + 1e-6):
            continue # P_e unreachable by Motor B

        den_cos_teb = 2.0 * R_B1 * R_B2
        if abs(den_cos_teb) < 1e-9:
            continue # Should not happen if R_B1, R_B2 > 0

        cos_theta_elbow_B = (D_e_sq - R_B1**2 - R_B2**2) / den_cos_teb

        cos_theta_elbow_B = max(-1.0, min(1.0, cos_theta_elbow_B)) # Clip

        theta_elbow_B_abs = math.acos(cos_theta_elbow_B)

        for elbow_B_sign in [1, -1]: # Two configurations for Motor B's internal elbow
            theta_elbow_B_signed = elbow_B_sign * theta_elbow_B_abs

            term_atan2_y = R_B2 * math.sin(theta_elbow_B_signed)
            term_atan2_x = R_B1 + R_B2 * math.cos(theta_elbow_B_signed)

            # Avoid atan2(0,0) if arm is singular (fully extended/folded directly on X-axis of its local frame)
            if abs(term_atan2_x) < 1e-9 and abs(term_atan2_y) < 1e-9 :
                angle_offset_B = 0.0 # Or Pi, depending on definition, if R_B1 + R_B2*cos = 0
            else:
                angle_offset_B = math.atan2(term_atan2_y, term_atan2_x)

            theta_B_rad = math.atan2(y_e, x_e) - angle_offset_B

            solutions.append({
                'theta_A': math.degrees(theta_A_rad),
                'theta_B': math.degrees(theta_B_rad),
                'Pe_id': pe_id, # Which P_e solution (0 or 1)
                'elbow_B_sign': elbow_B_sign # Which internal elbow config for B (+1 or -1)
            })

    return solutions

def select_solution(solutions, theta_A_prev_deg, theta_B_prev_deg,
                    preferred_Pe_id, preferred_elbow_B_sign):
    if not solutions:
        return None

    # If no previous angles, try to use the preferred configuration directly
    if theta_A_prev_deg is None or theta_B_prev_deg is None:
        for sol in solutions: # Check for exact preferred match
            if sol['Pe_id'] == preferred_Pe_id and sol['elbow_B_sign'] == preferred_elbow_B_sign:
                return sol
        for sol in solutions: # Fallback: preferred elbow_B_sign with any Pe_id
            if sol['elbow_B_sign'] == preferred_elbow_B_sign:
                return sol
        if solutions: # Still no match, return first available
            return solutions[0]
        return None

    # If previous angles exist, choose solution closest to them
    min_dist_sq = float('inf')
    best_sol = None

    for sol in solutions:
        # Calculate angular distance, handling wrap-around from -180 to +180
        diff_A = (sol['theta_A'] - theta_A_prev_deg + 180.0) % 360.0 - 180.0
        diff_B = (sol['theta_B'] - theta_B_prev_deg + 180.0) % 360.0 - 180.0

        dist_sq = diff_A**2 + diff_B**2

        # Optional: Penalize changing configuration type if continuity of config is desired
        # This can be refined if a specific previous config state variable is maintained
        # if sol['Pe_id'] != preferred_Pe_id: dist_sq += (30)**2
        # if sol['elbow_B_sign'] != preferred_elbow_B_sign: dist_sq += (30)**2

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_sol = sol

    return best_sol

def main():
    print("Robot Drawing Script Initialized")
    print(f"Arm lengths: R_A={R_A}, R_B1={R_B1}, R_B2={R_B2}")
    print(f"Canvas Origin (Robot Coords for Canvas Top-Left): X={ROBOT_X_FOR_CANVAS_ORIGIN}, Y={ROBOT_Y_FOR_CANVAS_ORIGIN_TOP_EDGE}")

    # Define the square in canvas coordinates (mm)
    square_start_x_c = 50.0
    square_start_y_c = 10.0
    square_side_length = 30.0
    steps_per_side = 10  # Number of points along each side of the square

    path_canvas_coords = []
    # Top edge
    for i in range(steps_per_side + 1):
        path_canvas_coords.append((square_start_x_c + i * (square_side_length / steps_per_side), square_start_y_c))
    # Right edge (start from i=1 to avoid duplicate corner)
    for i in range(1, steps_per_side + 1):
        path_canvas_coords.append((square_start_x_c + square_side_length, square_start_y_c + i * (square_side_length / steps_per_side)))
    # Bottom edge
    for i in range(1, steps_per_side + 1):
        path_canvas_coords.append((square_start_x_c + square_side_length - i * (square_side_length / steps_per_side), square_start_y_c + square_side_length))
    # Left edge
    for i in range(1, steps_per_side + 1):
        path_canvas_coords.append((square_start_x_c, square_start_y_c + square_side_length - i * (square_side_length / steps_per_side)))

    print(f"\nDefined a square path with {len(path_canvas_coords)} target points.")

    motor_A_path_deg = []
    motor_B_path_deg = []

    prev_theta_A, prev_theta_B = None, None

    # Preferred configuration from previous successful test (canvas (30,30) -> robot (-80,-120))
    # This specific point used Pe_id=1 and elbow_B_sign=-1 to get theta_A ~ -74.1, theta_B ~ 12.5
    PREFERRED_PE_ID_FOR_SELECTION = 1
    PREFERRED_ELBOW_B_SIGN_FOR_SELECTION = -1

    for point_idx, (xc, yc) in enumerate(path_canvas_coords):
        P_x_robot, P_y_robot = canvas_to_robot(xc, yc)

        print(f"\nProcessing Point {point_idx+1}/{len(path_canvas_coords)}: Canvas({xc:.2f}, {yc:.2f}) -> Robot({P_x_robot:.2f}, {P_y_robot:.2f})")

        ik_sols = calculate_inverse_kinematics(P_x_robot, P_y_robot)

        if not ik_sols:
            print(f"  ERROR: No IK solution found for Robot({P_x_robot:.2f}, {P_y_robot:.2f}). Skipping.")
            motor_A_path_deg.append(float('nan'))
            motor_B_path_deg.append(float('nan'))
            prev_theta_A, prev_theta_B = None, None # Reset continuity preference
            continue

        # Uncomment to see all raw solutions for a point:
        # print(f"  Found {len(ik_sols)} raw IK solutions:")
        # for s_idx, s_val in enumerate(ik_sols):
        #    print(f"    Sol {s_idx}: A={s_val['theta_A']:.1f}, B={s_val['theta_B']:.1f}, Pe_id={s_val['Pe_id']}, ElbowB_sign={s_val['elbow_B_sign']}")

        chosen_solution = select_solution(ik_sols, prev_theta_A, prev_theta_B,
                                          PREFERRED_PE_ID_FOR_SELECTION,
                                          PREFERRED_ELBOW_B_SIGN_FOR_SELECTION)

        if chosen_solution:
            current_theta_A = chosen_solution['theta_A']
            current_theta_B = chosen_solution['theta_B']
            print(f"  Selected Solution: θ_A={current_theta_A:.2f}°, θ_B={current_theta_B:.2f}° (Pe_id={chosen_solution['Pe_id']}, ElbowB_sign={chosen_solution['elbow_B_sign']})")
            motor_A_path_deg.append(current_theta_A)
            motor_B_path_deg.append(current_theta_B)
            prev_theta_A, prev_theta_B = current_theta_A, current_theta_B
        else:
            print(f"  ERROR: Could not select a solution for Robot({P_x_robot:.2f}, {P_y_robot:.2f}). Skipping.")
            motor_A_path_deg.append(float('nan'))
            motor_B_path_deg.append(float('nan'))
            prev_theta_A, prev_theta_B = None, None

    print("\n--- Generated Motor Angle Path (degrees) ---")
    print("Motor A angles:")
    print([f"{a:.2f}" if not math.isnan(a) else "NaN" for a in motor_A_path_deg])
    print("\nMotor B angles:")
    print([f"{b:.2f}" if not math.isnan(b) else "NaN" for b in motor_B_path_deg])

    # If you have matplotlib, you can uncomment the following to visualize:
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    # axs[0].plot(motor_A_path_deg, '.-', label="Motor A Angles (deg)")
    # axs[0].set_ylabel("Angle (deg)")
    # axs[0].set_title("Motor A Path")
    # axs[0].grid(True)
    # axs[0].legend()
    # axs[1].plot(motor_B_path_deg, '.-', label="Motor B Angles (deg)", color='orange')
    # axs[1].set_ylabel("Angle (deg)")
    # axs[1].set_xlabel("Path Point Index")
    # axs[1].set_title("Motor B Path")
    # axs[1].grid(True)
    # axs[1].legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
