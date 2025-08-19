# svg_to_pb.py

import svgpathtools
import os
import math
import time
import sys

# Project-specific imports
import penbot_constants as const

# Define conversion constants
LIFT_PEN_Z = const.LIFT_STEPS
LOWER_PEN_Z = 0
COMMAND_DELAY_MS = 0

def _debug_log(message):
    """Simple debug logger for this script."""
    print(f"[SVG2PB][{time.strftime('%H:%M:%S')}] {message}")

def svg_to_drawing_strokes(svg_file_path, segment_length_mm):
    """
    Parses an SVG file and converts its drawing elements into a list of
    'drawing strokes'. Each stroke is a list of (x, y) coordinates.
    """
    _debug_log(f"Processing SVG file: {svg_file_path}")
    strokes = []
    try:
        paths, attributes = svgpathtools.svg2paths(svg_file_path)
    except Exception as e:
        _debug_log(f"Error parsing SVG with svgpathtools: {e}")
        return []

    _debug_log(f"Found {len(paths)} main path objects in SVG.")

    def clip_point(x, y):
        x = max(0.0, min(const.CANVAS_WIDTH, x))
        y = max(0.0, min(const.CANVAS_HEIGHT, y))
        return x, y

    for path_obj in paths:
        for subpath in path_obj.continuous_subpaths():
            if subpath.length() < 1e-6:
                continue

            subpath_points = []
            start_point_complex = subpath.point(0.0)
            start_x, start_y = clip_point(start_point_complex.real, start_point_complex.imag)
            subpath_points.append((start_x, start_y))

            num_intermediate_segments = math.ceil(subpath.length() / segment_length_mm)
            for i in range(1, int(num_intermediate_segments) + 1):
                t = i / num_intermediate_segments
                point_complex = subpath.point(t)
                x, y = clip_point(point_complex.real, point_complex.imag)
                subpath_points.append((x, y))

            cleaned_points = []
            if subpath_points:
                cleaned_points.append(subpath_points[0])
                for i in range(1, len(subpath_points)):
                    if abs(subpath_points[i][0] - cleaned_points[-1][0]) > 1e-6 or \
                       abs(subpath_points[i][1] - cleaned_points[-1][1]) > 1e-6:
                        cleaned_points.append(subpath_points[i])
            
            if cleaned_points:
                strokes.append(cleaned_points)

    _debug_log(f"Generated a total of {len(strokes)} drawing strokes from SVG.")
    return strokes

def convert_svg_to_pb(svg_file_path, pb_file_path):
    """
    Converts SVG to Penbot PB commands and returns statistics.
    Returns: A dictionary of stats on success, None on failure.
    """
    _debug_log(f"Converting SVG to PB: {svg_file_path} -> {pb_file_path}")
    strokes = svg_to_drawing_strokes(svg_file_path, const.DEFAULT_SEGMENT_LENGTH_MM)

    if not strokes:
        _debug_log("No drawing strokes found or generated. Aborting conversion.")
        return None
        
    commands = []
    lifts = 0
    
    # --- Build command list ---
    commands.append(f"PBR,1") # Enable motors
    commands.append(f"PBZ,{LIFT_PEN_Z}") # Ensure pen is up
    
    current_x, current_y = const.CANVAS_HOME_X, const.CANVAS_HOME_Y
    current_z_state = LIFT_PEN_Z
    
    first_stroke_first_point_x, first_stroke_first_point_y = strokes[0][0]
    if abs(first_stroke_first_point_x - current_x) > 1e-6 or abs(first_stroke_first_point_y - current_y) > 1e-6:
        commands.append(f"PBM,{COMMAND_DELAY_MS},{first_stroke_first_point_x:.2f},{first_stroke_first_point_y:.2f}")
        current_x, current_y = first_stroke_first_point_x, first_stroke_first_point_y
        
    for stroke_points in strokes:
        if not stroke_points: continue
        
        stroke_start_x, stroke_start_y = stroke_points[0]
        
        if current_z_state != LIFT_PEN_Z:
            commands.append(f"PBZ,{LIFT_PEN_Z}")
            lifts += 1
            current_z_state = LIFT_PEN_Z
            
        if abs(stroke_start_x - current_x) > 1e-6 or abs(stroke_start_y - current_y) > 1e-6:
            commands.append(f"PBM,{COMMAND_DELAY_MS},{stroke_start_x:.2f},{stroke_start_y:.2f}")
            current_x, current_y = stroke_start_x, stroke_start_y
            
        commands.append(f"PBZ,{LOWER_PEN_Z}")
        current_z_state = LOWER_PEN_Z
        
        for point_x, point_y in stroke_points:
            commands.append(f"PBM,{COMMAND_DELAY_MS},{point_x:.2f},{point_y:.2f}")
            current_x, current_y = point_x, point_y

    # --- Finalize ---
    if current_z_state != LIFT_PEN_Z:
        commands.append(f"PBZ,{LIFT_PEN_Z}")
        lifts += 1
    
    if abs(const.CANVAS_HOME_X - current_x) > 1e-6 or abs(const.CANVAS_HOME_Y - current_y) > 1e-6:
        commands.append(f"PBM,{COMMAND_DELAY_MS},{const.CANVAS_HOME_X:.2f},{const.CANVAS_HOME_Y:.2f}")
        
    commands.append(f"PBR,0") # Disable motors

    # --- Write to file and return stats ---
    with open(pb_file_path, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
            
    _debug_log(f"Conversion complete. PB commands written to: {pb_file_path}")
    
    stats = {
        'strokes': len(strokes),
        'lifts': lifts,
        'commands': len(commands)
    }
    return stats

def main():
    # Standalone test function
    parser = argparse.ArgumentParser(description="SVG to Penbot PB Code Converter")
    parser.add_argument("input_svg", help="Path to the input SVG file.")
    parser.add_argument("output_pb", help="Path for the output Penbot (.pb) file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_svg):
        print(f"Error: Input SVG file not found at '{args.input_svg}'")
        sys.exit(1)

    stats = convert_svg_to_pb(args.input_svg, args.output_pb)
    if stats:
        print("Conversion successful!")
        print(f"Stats: {stats}")

if __name__ == "__main__":
    main()