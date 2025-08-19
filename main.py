# main.py

import cv2
import numpy as np
import os
import time
import threading
import signal
import sys
import svgpathtools

# --- Project Modules ---
import face_capture
import svg_to_pb
import robot_driver

# --- Configuration ---
SVG_OUTPUT_FILE = "face.svg"
PB_OUTPUT_FILE = "face.pb"

# --- UI Helper Functions ---

def draw_multiline_text(image, text, start_pos, font=cv2.FONT_HERSHEY_DUPLEX, scale=1.0, color=(0,0,0), thickness=2):
    """Draws text with multiple lines on a cv2 image."""
    x, y = start_pos
    for i, line in enumerate(text.split('\n')):
        text_size = cv2.getTextSize(line, font, scale, thickness)[0]
        line_y = y + i * (text_size[1] + 15) # 15px line spacing
        cv2.putText(image, line, (x, line_y), font, scale, (200,200,200), thickness+1, cv2.LINE_AA) # Shadow
        cv2.putText(image, line, (x, line_y), font, scale, color, thickness, cv2.LINE_AA)

def draw_progress_bar(image, progress_ratio, position, size, text):
    """Draws a progress bar with text on a cv2 image."""
    x, y, w, h = *position, *size
    # Draw background
    cv2.rectangle(image, (x, y), (x + w, y + h), (180, 180, 180), -1)
    # Draw progress
    cv2.rectangle(image, (x, y), (x + int(w * progress_ratio), y + h), (0, 200, 50), -1)
    # Draw border
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # Draw text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx, ty = x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

def display_message_screen(window_name, message, wait_key=True):
    """Displays a simple message screen and waits for a key press."""
    screen = np.full((720, 1280, 3), 255, dtype=np.uint8)
    draw_multiline_text(screen, message, (100, 150))
    cv2.imshow(window_name, screen)
    return cv2.waitKey(0) if wait_key else cv2.waitKey(1)

def render_svg_to_image(svg_path, output_size=(640, 480)):
    """Renders an SVG file to a numpy array for display in OpenCV."""
    try:
        paths, attributes, svg_attributes = svgpathtools.svg2paths2(svg_path)
    except Exception as e:
        print(f"Could not read SVG for preview: {e}")
        return None

    # Get SVG dimensions from its own attributes
    svg_w = float(svg_attributes.get('width', '150').replace('mm', ''))
    svg_h = float(svg_attributes.get('height', '100').replace('mm', ''))

    if svg_w == 0 or svg_h == 0:
        return None

    # Calculate scaling to fit into the output_size, with a small margin
    margin = 0.95
    scale = min((output_size[0] / svg_w) * margin, (output_size[1] / svg_h) * margin)

    # Create a blank white canvas
    canvas = np.full((output_size[1], output_size[0], 3), 255, dtype=np.uint8)
    
    # Calculate offsets to center the scaled drawing
    offset_x = (output_size[0] - (svg_w * scale)) / 2
    offset_y = (output_size[1] - (svg_h * scale)) / 2

    # Render each path
    for path in paths:
        points = []
        num_segments = max(2, int(path.length() / 0.5)) # 0.5mm resolution for preview
        for i in range(num_segments + 1):
            p = path.point(i / num_segments)
            px, py = int(p.real * scale + offset_x), int(p.imag * scale + offset_y)
            points.append((px, py))
        
        if len(points) > 1:
            pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_np], isClosed=False, color=(0, 0, 0), thickness=1)

    return canvas

# --- Main Application Logic ---

def main_workflow():
    """The main state machine for the application."""
    WINDOW_NAME = "Penbot Portrait Studio"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    drawing_progress = {'current': 0, 'total': 1, 'active': False}
    stop_drawing_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nCtrl+C detected! Requesting drawing thread to stop...")
        stop_drawing_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def progress_callback(current_line, total_lines):
        drawing_progress['current'] = current_line
        drawing_progress['total'] = total_lines

    state = "CAPTURE"
    while not stop_drawing_event.is_set():
        if state == "CAPTURE":
            svg_path = face_capture.run_capture_ui()
            if svg_path:
                state = "CONFIRM"
            else: # User quit the capture UI
                break

        elif state == "CONFIRM":
            screen = np.full((720, 1280, 3), 255, dtype=np.uint8)
            message = (
                "Vectorized Preview:\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                "Press [ENTER] to convert to robot code.\n"
                "Press [N] to take another picture."
            )
            draw_multiline_text(screen, message, (50, 50))
            
            svg_preview = render_svg_to_image(SVG_OUTPUT_FILE, output_size=(800, 533))
            if svg_preview is not None:
                h, w, _ = svg_preview.shape
                x_start = (1280 - w) // 2
                y_start = 100
                screen[y_start:y_start+h, x_start:x_start+w] = svg_preview

            cv2.imshow(WINDOW_NAME, screen)
            key = cv2.waitKey(0)

            if key == 13: # 13 is Enter
                state = "CONVERT"
            elif key == ord('n'): 
                state = "CAPTURE"
            elif key == ord('q'): 
                break

        elif state == "CONVERT":
            display_message_screen(WINDOW_NAME, "Converting SVG to PB code...", wait_key=False)
            time.sleep(0.5)
            stats = svg_to_pb.convert_svg_to_pb(SVG_OUTPUT_FILE, PB_OUTPUT_FILE)
            if not stats:
                 display_message_screen(WINDOW_NAME, "Conversion failed. No strokes found.\nPress any key to try again.")
                 state = "CAPTURE"; continue
            message = (f"Conversion Complete!\n\n - Total Strokes: {stats['strokes']}\n - Pen Lifts: {stats['lifts']}\n - Total Commands: {stats['commands']}\n\n"
                       "Ready to draw? Make sure the robot is positioned.\n\n"
                       "[S] - START DRAWING\n[C] - Cancel & Restart")
            key = display_message_screen(WINDOW_NAME, message)
            if key == ord('s'):
                state = "DRAW"
            elif key == ord('c'):
                state = "CAPTURE"
            elif key == ord('q'):
                break

        elif state == "DRAW":
            stop_drawing_event.clear()
            drawing_progress['active'] = True
            drawing_thread = threading.Thread(
                target=robot_driver.run_drawing_from_file,
                args=(PB_OUTPUT_FILE, stop_drawing_event, progress_callback)
            )
            drawing_thread.start()

            while drawing_thread.is_alive():
                screen = np.full((720, 1280, 3), 255, dtype=np.uint8)
                draw_multiline_text(screen, "Drawing in progress...", (100, 150), scale=1.5)
                progress = drawing_progress['current'] / drawing_progress['total'] if drawing_progress['total'] > 0 else 0
                progress_text = f"{drawing_progress['current']} / {drawing_progress['total']} Commands"
                draw_progress_bar(screen, progress, (240, 300), (800, 60), progress_text)
                draw_multiline_text(screen, "Press Ctrl+C in the terminal to stop gracefully.", (100, 500))
                cv2.imshow(WINDOW_NAME, screen)
                if cv2.waitKey(100) == ord('q'):
                    stop_drawing_event.set()
            
            drawing_thread.join()
            drawing_progress['active'] = False
            state = "FINISHED" if not stop_drawing_event.is_set() else "CAPTURE"

        elif state == "FINISHED":
            message = "Drawing Complete!\n\n[R] - Reset and take another picture\n[Q] - Quit the program"
            key = display_message_screen(WINDOW_NAME, message)
            if key == ord('r'):
                state = "CAPTURE"
            elif key == ord('q'):
                break

    # Cleanup
    print("\nShutting down...")
    cv2.destroyAllWindows()
    if os.path.exists(SVG_OUTPUT_FILE): os.remove(SVG_OUTPUT_FILE)
    if os.path.exists(PB_OUTPUT_FILE): os.remove(PB_OUTPUT_FILE)
    print("Goodbye!")
    sys.exit(0)

if __name__ == "__main__":
    main_workflow()