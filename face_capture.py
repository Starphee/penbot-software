# face_capture.py

import cv2
import numpy as np
import mediapipe as mp
from skimage.morphology import skeletonize
import time
import io
import svgpathtools
import svgelements as se

# Project-specific import for canvas dimensions
import penbot_constants as const

# Global Setup
VECTOR_OUTPUT_FILE = "face.svg"

# Initialize Mediapipe Models
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

mp_face_detection = mp.solutions.face_detection
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Functions ---

def create_control_panel():
    """Creates a window with trackbars for adjusting parameters."""
    cv2.namedWindow("ARTIST CONTROLS")
    def nothing(x): pass
    cv2.createTrackbar("SMOOTHNESS", "ARTIST CONTROLS", 1, 10, nothing)
    cv2.createTrackbar("DETAIL FINDER", "ARTIST CONTROLS", 3, 25, nothing)
    cv2.createTrackbar("SENSITIVITY", "ARTIST CONTROLS", 3, 20, nothing)
    cv2.createTrackbar("NOISE FILTER", "ARTIST CONTROLS", 22, 100, nothing)

def save_svg(contours, source_width, source_height):
    """Saves contours and a logo to a plotter-friendly SVG file."""
    target_w, target_h = const.CANVAS_WIDTH, const.CANVAS_HEIGHT
    face_size_factor, face_horizontal_position_factor = 0.6, 0.25
    if contours and source_width > 0 and source_height > 0:
        face_scale = min(target_w / source_width, target_h / source_height) * face_size_factor
        scaled_face_w, scaled_face_h = source_width * face_scale, source_height * face_scale
        face_offset_y = (target_h - scaled_face_h) / 2
        face_offset_x = max(0, (target_w * face_horizontal_position_factor) - (scaled_face_w / 2))
    else:
        contours, face_scale, face_offset_x, face_offset_y = [], 1, 0, 0
    logo_svg_data = """<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xcs="https://www.xtool.com/pages/software" version="1.1" preserveAspectRatio="xMinYMin meet" width="98.5mm" height="50.6mm" viewBox="1 1 98.5 50.6" xcs:version="1.7.8"><g transform="matrix(1,0,0,1,-0.52,18.780001)" stroke="#2366ff" fill="none"><path transform="matrix(1,0,0,1,5.588,1.42109e-14)" stroke="#2366ff" fill="none" d="M2.03 0L2.03 -17.78L8.84 -17.78Q10.44 -17.78 11.7 -17.16Q12.95 -16.54 13.69 -15.39Q14.43 -14.25 14.43 -12.62L14.43 -12.29Q14.43 -10.69 13.68 -9.54Q12.93 -8.38 11.66 -7.77Q10.39 -7.16 8.84 -7.16L4.17 -7.16L4.17 0ZM4.17 -9.09L8.61 -9.09Q10.31 -9.09 11.3 -9.94Q12.29 -10.79 12.29 -12.34L12.29 -12.6Q12.29 -14.15 11.32 -15Q10.34 -15.85 8.61 -15.85L4.17 -15.85Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,20.8534,1.42109e-14)" stroke="#2366ff" fill="none" d="M7.8 0.36Q5.92 0.36 4.5 -0.44Q3.07 -1.24 2.29 -2.69Q1.5 -4.14 1.5 -6.05L1.5 -6.35Q1.5 -8.28 2.29 -9.73Q3.07 -11.18 4.47 -11.98Q5.87 -12.78 7.67 -12.78Q9.42 -12.78 10.77 -12.03Q12.12 -11.28 12.88 -9.88Q13.64 -8.48 13.64 -6.6L13.64 -5.69L3.51 -5.69Q3.58 -3.68 4.8 -2.55Q6.02 -1.42 7.85 -1.42Q9.45 -1.42 10.31 -2.16Q11.18 -2.9 11.63 -3.91L13.36 -3.07Q12.98 -2.29 12.31 -1.5Q11.63 -0.71 10.55 -0.18Q9.47 0.36 7.8 0.36ZM3.53 -7.34L11.61 -7.34Q11.51 -9.07 10.43 -10.03Q9.35 -11 7.67 -11Q5.97 -11 4.88 -10.03Q3.78 -9.07 3.53 -7.34Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,35.8902,1.42109e-14)" stroke="#2366ff" fill="none" d="M2.11 0L2.11 -12.42L4.06 -12.42L4.06 -10.31L4.42 -10.31Q4.83 -11.2 5.8 -11.93Q6.78 -12.65 8.69 -12.65Q10.08 -12.65 11.19 -12.06Q12.29 -11.48 12.95 -10.34Q13.61 -9.19 13.61 -7.52L13.61 0L11.61 0L11.61 -7.37Q11.61 -9.22 10.68 -10.06Q9.75 -10.9 8.18 -10.9Q6.38 -10.9 5.25 -9.73Q4.11 -8.56 4.11 -6.25L4.11 0Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,51.4604,1.42109e-14)" stroke="#2366ff" fill="none" d="M1.19 0L1.19 -1.9L3.68 -1.9L3.68 -15.87L1.19 -15.87L1.19 -17.78L10.11 -17.78Q11.66 -17.78 12.87 -17.22Q14.07 -16.66 14.74 -15.66Q15.42 -14.66 15.42 -13.31L15.42 -13.11Q15.42 -11.86 14.94 -11.06Q14.45 -10.26 13.79 -9.82Q13.13 -9.37 12.5 -9.17L12.5 -8.81Q13.13 -8.66 13.82 -8.22Q14.5 -7.77 14.97 -6.96Q15.44 -6.15 15.44 -4.9L15.44 -4.65Q15.44 -3.2 14.74 -2.15Q14.05 -1.09 12.84 -0.55Q11.63 0 10.08 0ZM5.82 -1.93L9.86 -1.93Q11.53 -1.93 12.42 -2.72Q13.31 -3.51 13.31 -4.85L13.31 -5.08Q13.31 -6.43 12.42 -7.21Q11.53 -8 9.86 -8L5.82 -8ZM5.82 -9.93L9.93 -9.93Q11.46 -9.93 12.37 -10.71Q13.28 -11.48 13.28 -12.75L13.28 -13Q13.28 -14.3 12.38 -15.07Q11.48 -15.85 9.93 -15.85L5.82 -15.85Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,68.2498,1.42109e-14)" stroke="#2366ff" fill="none" d="M7.85 0.36Q5.97 0.36 4.53 -0.43Q3.1 -1.22 2.3 -2.65Q1.5 -4.09 1.5 -6.05L1.5 -6.38Q1.5 -8.31 2.3 -9.75Q3.1 -11.2 4.53 -11.99Q5.97 -12.78 7.85 -12.78Q9.73 -12.78 11.16 -11.99Q12.6 -11.2 13.4 -9.75Q14.2 -8.31 14.2 -6.38L14.2 -6.05Q14.2 -4.09 13.4 -2.65Q12.6 -1.22 11.16 -0.43Q9.73 0.36 7.85 0.36ZM7.85 -1.45Q9.83 -1.45 11.01 -2.71Q12.19 -3.96 12.19 -6.1L12.19 -6.32Q12.19 -8.46 11.01 -9.72Q9.83 -10.97 7.85 -10.97Q5.89 -10.97 4.7 -9.72Q3.51 -8.46 3.51 -6.32L3.51 -6.1Q3.51 -3.96 4.7 -2.71Q5.89 -1.45 7.85 -1.45Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,83.947,1.42109e-14)" stroke="#2366ff" fill="none" d="M6.68 0Q5.61 0 5.04 -0.58Q4.47 -1.17 4.47 -2.18L4.47 -10.67L0.74 -10.67L0.74 -12.42L4.47 -12.42L4.47 -16.84L6.48 -16.84L6.48 -12.42L10.54 -12.42L10.54 -10.67L6.48 -10.67L6.48 -2.49Q6.48 -1.73 7.21 -1.73L9.98 -1.73L9.98 0Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,0,32.4104)" stroke="#2366ff" fill="none" d="M8.48 0.36Q5.31 0.36 3.42 -1.49Q1.52 -3.33 1.52 -6.86L1.52 -10.92Q1.52 -14.45 3.42 -16.29Q5.31 -18.14 8.48 -18.14Q11.68 -18.14 13.58 -16.29Q15.47 -14.45 15.47 -10.92L15.47 -6.86Q15.47 -3.33 13.58 -1.49Q11.68 0.36 8.48 0.36ZM8.48 -1.55Q10.8 -1.55 12.07 -2.93Q13.34 -4.32 13.34 -6.78L13.34 -11Q13.34 -13.46 12.07 -14.85Q10.8 -16.23 8.48 -16.23Q6.2 -16.23 4.93 -14.85Q3.66 -13.46 3.66 -11L3.66 -6.78Q3.66 -4.32 4.93 -2.93Q6.2 -1.55 8.48 -1.55Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,16.9926,32.4104)" stroke="#2366ff" fill="none" d="M7.95 0.36Q6.02 0.36 4.5 -0.33Q2.97 -1.02 2.1 -2.4Q1.22 -3.78 1.22 -5.87L1.22 -6.4L3.33 -6.4L3.33 -5.87Q3.33 -3.66 4.62 -2.58Q5.92 -1.5 7.95 -1.5Q10.03 -1.5 11.13 -2.44Q12.22 -3.38 12.22 -4.8Q12.22 -5.79 11.72 -6.38Q11.23 -6.96 10.34 -7.33Q9.45 -7.7 8.25 -7.98L6.93 -8.31Q5.38 -8.71 4.19 -9.3Q3 -9.88 2.32 -10.83Q1.65 -11.79 1.65 -13.28Q1.65 -14.78 2.4 -15.87Q3.15 -16.97 4.51 -17.55Q5.87 -18.14 7.65 -18.14Q9.45 -18.14 10.88 -17.51Q12.32 -16.89 13.14 -15.68Q13.97 -14.48 13.97 -12.65L13.97 -11.58L11.86 -11.58L11.86 -12.65Q11.86 -13.94 11.32 -14.73Q10.77 -15.52 9.82 -15.9Q8.86 -16.28 7.65 -16.28Q5.89 -16.28 4.83 -15.51Q3.76 -14.73 3.76 -13.31Q3.76 -12.37 4.22 -11.79Q4.67 -11.2 5.51 -10.83Q6.35 -10.46 7.52 -10.19L8.84 -9.86Q10.39 -9.52 11.62 -8.95Q12.85 -8.38 13.59 -7.4Q14.33 -6.43 14.33 -4.85Q14.33 -3.28 13.54 -2.11Q12.75 -0.94 11.32 -0.29Q9.88 0.36 7.95 0.36Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,39.0906,32.4104)" stroke="#2366ff" fill="none" d="M1.35 0L1.35 -1.85Q1.35 -3.76 2.02 -4.9Q2.69 -6.05 3.9 -6.74Q5.11 -7.44 6.78 -7.95L8.18 -8.38Q9.32 -8.74 10.19 -9.26Q11.05 -9.78 11.53 -10.58Q12.01 -11.38 12.01 -12.55L12.01 -12.65Q12.01 -14.33 10.85 -15.3Q9.68 -16.28 7.85 -16.28Q5.92 -16.28 4.7 -15.25Q3.48 -14.22 3.48 -12.14L3.48 -11.79L1.45 -11.79L1.45 -12.12Q1.45 -14.05 2.29 -15.39Q3.12 -16.74 4.57 -17.44Q6.02 -18.14 7.85 -18.14Q9.68 -18.14 11.09 -17.45Q12.5 -16.76 13.28 -15.54Q14.07 -14.33 14.07 -12.7L14.07 -12.47Q14.07 -10.74 13.39 -9.61Q12.7 -8.48 11.49 -7.77Q10.29 -7.06 8.66 -6.55L7.29 -6.12Q5.99 -5.71 5.14 -5.27Q4.29 -4.83 3.86 -4.14Q3.43 -3.45 3.43 -2.29L3.43 -1.85L13.92 -1.85L13.92 0Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,54.4068,32.4104)" stroke="#2366ff" fill="none" d="M8.13 0.36Q5.11 0.36 3.31 -1.44Q1.52 -3.23 1.52 -6.81L1.52 -10.97Q1.52 -14.53 3.31 -16.33Q5.11 -18.14 8.13 -18.14Q11.18 -18.14 12.97 -16.33Q14.76 -14.53 14.76 -10.97L14.76 -6.81Q14.76 -3.23 12.97 -1.44Q11.18 0.36 8.13 0.36ZM8.13 -1.5Q10.41 -1.5 11.56 -2.87Q12.7 -4.24 12.7 -6.71L12.7 -11.1Q12.7 -13.56 11.53 -14.92Q10.36 -16.28 8.13 -16.28Q5.89 -16.28 4.74 -14.91Q3.58 -13.54 3.58 -11.1L3.58 -6.71Q3.58 -4.22 4.72 -2.86Q5.87 -1.5 8.13 -1.5Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,70.6882,32.4104)" stroke="#2366ff" fill="none" d="M1.35 0L1.35 -1.85Q1.35 -3.76 2.02 -4.9Q2.69 -6.05 3.9 -6.74Q5.11 -7.44 6.78 -7.95L8.18 -8.38Q9.32 -8.74 10.19 -9.26Q11.05 -9.78 11.53 -10.58Q12.01 -11.38 12.01 -12.55L12.01 -12.65Q12.01 -14.33 10.85 -15.3Q9.68 -16.28 7.85 -16.28Q5.92 -16.28 4.7 -15.25Q3.48 -14.22 3.48 -12.14L3.48 -11.79L1.45 -11.79L1.45 -12.12Q1.45 -14.05 2.29 -15.39Q3.12 -16.74 4.57 -17.44Q6.02 -18.14 7.85 -18.14Q9.68 -18.14 11.09 -17.45Q12.5 -16.76 13.28 -15.54Q14.07 -14.33 14.07 -12.7L14.07 -12.47Q14.07 -10.74 13.39 -9.61Q12.7 -8.48 11.49 -7.77Q10.29 -7.06 8.66 -6.55L7.29 -6.12Q5.99 -5.71 5.14 -5.27Q4.29 -4.83 3.86 -4.14Q3.43 -3.45 3.43 -2.29L3.43 -1.85L13.92 -1.85L13.92 0Z" fillRule="nonzero"></path><path transform="matrix(1,0,0,1,86.0044,32.4104)" stroke="#2366ff" fill="none" d="M7.67 0.36Q5.72 0.36 4.29 -0.39Q2.87 -1.14 2.08 -2.48Q1.3 -3.81 1.3 -5.56L1.3 -5.87L3.35 -5.87L3.35 -5.64Q3.35 -3.63 4.52 -2.57Q5.69 -1.5 7.62 -1.5Q9.65 -1.5 10.8 -2.64Q11.94 -3.78 11.94 -5.77L11.94 -5.99Q11.94 -7.9 10.83 -9.03Q9.73 -10.16 7.98 -10.16Q6.96 -10.16 6.3 -9.84Q5.64 -9.52 5.26 -9.06Q4.88 -8.59 4.62 -8.15L1.83 -8.15L1.83 -17.78L13.51 -17.78L13.51 -15.93L3.89 -15.93L3.89 -9.78L4.24 -9.78Q4.52 -10.29 5.02 -10.77Q5.51 -11.25 6.35 -11.58Q7.19 -11.91 8.43 -11.91Q9.98 -11.91 11.24 -11.2Q12.5 -10.49 13.25 -9.17Q14 -7.85 14 -6.02L14 -5.74Q14 -3.94 13.23 -2.57Q12.47 -1.19 11.05 -0.42Q9.63 0.36 7.67 0.36Z" fillRule="nonzero"></path></g></svg>"""
    svg_root = se.SVG.parse(io.StringIO(logo_svg_data.strip()))
    logo_paths = []
    for element in svg_root.elements():
        if isinstance(element, se.Path) and len(element) > 0:
            logo_paths.append(svgpathtools.parse_path(element.d()))
    if logo_paths:
        overall_min_x, overall_max_x = float('inf'), float('-inf')
        overall_min_y, overall_max_y = float('inf'), float('-inf')
        for path in logo_paths:
            try:
                min_x_path, max_x_path, min_y_path, max_y_path = path.bbox()
                overall_min_x, overall_max_x = min(overall_min_x, min_x_path), max(overall_max_x, max_x_path)
                overall_min_y, overall_max_y = min(overall_min_y, min_y_path), max(overall_max_y, max_y_path)
            except (ValueError, IndexError): continue
        min_x, max_x, min_y, max_y = overall_min_x, overall_max_x, overall_min_y, overall_max_y
        if min_x == float('inf'): min_x, max_x, min_y, max_y = 0, 0, 0, 0
        text_source_width, text_source_height = max_x - min_x, max_y - min_y
        text_target_width = target_w * 0.45
        text_scale = text_target_width / text_source_width if text_source_width > 0 else 1
        text_target_height = text_source_height * text_scale
        text_offset_y = (target_h - text_target_height) / 2
        text_offset_x = target_w * 0.75 - (text_target_width / 2)
        final_tx, final_ty = text_offset_x - (min_x * text_scale), text_offset_y - (min_y * text_scale)
    else: text_scale, final_tx, final_ty = 1, 0, 0
    with open(VECTOR_OUTPUT_FILE, "w") as f:
        f.write(f'<svg width="{target_w}mm" height="{target_h}mm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {target_w} {target_h}">\n')
        if contours:
            for contour in contours:
                if len(contour) < 1: continue
                start_point = contour[0][0]
                path_data = f"M {(start_point[0] * face_scale) + face_offset_x:.3f} {(start_point[1] * face_scale) + face_offset_y:.3f}"
                for point in contour[1:]: path_data += f" L {(point[0][0] * face_scale) + face_offset_x:.3f} {(point[0][1] * face_scale) + face_offset_y:.3f}"
                f.write(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="0.5"/>\n')
        if logo_paths:
            for path in logo_paths:
                for subpath in path.continuous_subpaths():
                    if subpath.length() < 1e-6: continue
                    num_segments = max(2, int(subpath.length() / const.DEFAULT_SEGMENT_LENGTH_MM))
                    start_complex = subpath.point(0)
                    path_data = f"M {start_complex.real * text_scale + final_tx:.3f} {start_complex.imag * text_scale + final_ty:.3f}"
                    for i in range(1, num_segments + 1):
                        p_complex = subpath.point(i / num_segments)
                        path_data += f" L {p_complex.real * text_scale + final_tx:.3f} {p_complex.imag * text_scale + final_ty:.3f}"
                    f.write(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="0.5"/>\n')
        f.write('</svg>')
    print(f"\n[FACE_CAPTURE] Artwork saved to: {VECTOR_OUTPUT_FILE}")

def process_frame_to_contours(frame, blur_val, block_val, c_val, min_line_length):
    """Processes a frame to find face contours."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    white_bg_frame = np.full_like(frame, 255, dtype=np.uint8)
    results_segmentation = selfie_segmentation.process(rgb_frame)
    if results_segmentation.segmentation_mask is None: return None, None, white_bg_frame, None, None
    mask = results_segmentation.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.6
    frame_no_bg = np.where(condition, frame, white_bg_frame)
    results_face = face_detection.process(cv2.cvtColor(frame_no_bg, cv2.COLOR_BGR2RGB))
    if not results_face.detections: return None, None, frame_no_bg, None, None
    bboxC = results_face.detections[0].location_data.relative_bounding_box
    ih, iw, _ = frame_no_bg.shape
    padding = 0.1
    x, y, w, h = int(bboxC.xmin*iw*(1-padding)), int(bboxC.ymin*ih*(1-padding)), int(bboxC.width*iw*(1+padding*2)), int(bboxC.height*ih*(1+padding*2))
    x,y,w,h = max(0,x), max(0,y), min(iw-x,w), min(ih-y,h)
    if w <= 0 or h <= 0: return None, None, frame_no_bg, None, None
    circular_mask = cv2.ellipse(np.full((h,w,3), 255, dtype=np.uint8), (w//2,h//2), (w//2,h//2), 0,0,360,(0,0,0),-1)
    face_crop = cv2.bitwise_and(frame_no_bg[y:y+h, x:x+w], cv2.bitwise_not(circular_mask))
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    k_size_blur, k_size_thresh = (blur_val * 2) + 1, (block_val * 2) + 3
    binary = cv2.adaptiveThreshold(cv2.medianBlur(gray_face, k_size_blur), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k_size_thresh, c_val)
    skeleton = (skeletonize(binary > 0) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.arcLength(c, False) > min_line_length]
    return valid_contours, face_crop, frame_no_bg, (x, y, w, h), gray_face.shape[::-1]

def run_capture_ui():
    """Main function to run the real-time face drawer UI."""
    camNum = 0
    cap = cv2.VideoCapture(camNum)
    if not cap.isOpened():
        for camNum in range(1, 10):
            cap = cv2.VideoCapture(camNum)
            if cap.isOpened(): break
        else: return None

    controls_visible = False
    MAIN_WINDOW_NAME = "Penbot Portrait Studio"
    
    STATE_PREVIEW, STATE_COUNTDOWN, STATE_FLASH, STATE_PROCESSING = "preview", "countdown", "flash", "processing"
    current_state = STATE_PREVIEW
    animation_start_time, countdown_number, frame_for_processing, last_valid_contours = 0, 3, None, []
    BASE_CONTENT_WIDTH, BASE_CONTENT_HEIGHT = 1280, 560

    blur_val, block_val, c_val, min_line_length = 1, 3, 3, 22

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        if controls_visible:
            if cv2.getWindowProperty("ARTIST CONTROLS", cv2.WND_PROP_VISIBLE) < 1:
                controls_visible = False
            else:
                blur_val, block_val, c_val, min_line_length = (
                    cv2.getTrackbarPos(name, "ARTIST CONTROLS") for name in 
                    ["SMOOTHNESS", "DETAIL FINDER", "SENSITIVITY", "NOISE FILTER"]
                )
        
        _, _, win_w, win_h = cv2.getWindowImageRect(MAIN_WINDOW_NAME)
        display_frame = np.full((win_h if win_h > 0 else 720, win_w if win_w > 0 else 1280, 3), 255, dtype=np.uint8)
        raw_content_frame = None

        if current_state in [STATE_PREVIEW, STATE_COUNTDOWN]:
            contours, face_crop, frame_display_bg, bbox, _ = process_frame_to_contours(frame, blur_val, block_val, c_val, min_line_length)
            pane_h, pane_w = 480, 640
            live_pane = cv2.resize(frame_display_bg, (pane_w, pane_h))
            vector_pane = np.full((pane_h, pane_w, 3), 255, dtype=np.uint8)
            if contours is not None and bbox is not None:
                last_valid_contours = contours
                x, y, w, h = bbox; ih, iw, _ = frame.shape
                cv2.rectangle(live_pane, (int(x/iw*pane_w), int(y/ih*pane_h)), (int((x+w)/iw*pane_w), int((y+h)/ih*pane_h)), (0, 255, 0), 2)
                if face_crop is not None:
                    vector_vis = np.full_like(face_crop, 255)
                    cv2.drawContours(vector_vis, contours, -1, (0, 0, 0), 1)
                    vector_pane = cv2.resize(vector_vis, (pane_w, pane_h))
            else: last_valid_contours = []
            
            top_row = np.hstack((live_pane, vector_pane))
            if current_state == STATE_COUNTDOWN:
                if time.time() - animation_start_time > 1.0:
                    countdown_number -= 1; animation_start_time = time.time()
                if countdown_number <= 0:
                    current_state, animation_start_time, frame_for_processing = STATE_FLASH, time.time(), frame.copy()
                else:
                    text = str(countdown_number); f_scale, thick = 10, 15
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, f_scale, thick)
                    tx, ty = (top_row.shape[1] - tw) // 2, (top_row.shape[0] + th) // 2
                    cv2.putText(top_row, text, (tx + 5, ty + 5), cv2.FONT_HERSHEY_TRIPLEX, f_scale, (200, 200, 200), thick + 5, cv2.LINE_AA)
                    cv2.putText(top_row, text, (tx, ty), cv2.FONT_HERSHEY_TRIPLEX, f_scale, (0, 0, 0), thick, cv2.LINE_AA)
            
            msg_bar = np.full((BASE_CONTENT_HEIGHT - pane_h, BASE_CONTENT_WIDTH, 3), 255, dtype=np.uint8)
            msg1 = "PRESS ENTER TO CAPTURE!" if last_valid_contours else "GET YOUR FACE IN THE FRAME!"
            msg2 = "Press 'A' to toggle artist controls"
            (tw1, th1), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            (tw2, th2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
            cv2.putText(msg_bar, msg1, ((msg_bar.shape[1]-tw1)//2+2, (msg_bar.shape[0]+th1)//2-13), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100,100,100), 2, cv2.LINE_AA)
            cv2.putText(msg_bar, msg1, ((msg_bar.shape[1]-tw1)//2, (msg_bar.shape[0]+th1)//2-15), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(msg_bar, msg2, ((msg_bar.shape[1]-tw2)//2, (msg_bar.shape[0]+th1)//2+30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50,50,50), 1, cv2.LINE_AA)
            raw_content_frame = np.vstack((top_row, msg_bar))

        elif current_state == STATE_FLASH:
            elapsed = time.time() - animation_start_time
            if elapsed < 0.1: raw_content_frame = np.full((BASE_CONTENT_HEIGHT, BASE_CONTENT_WIDTH, 3), 255, dtype=np.uint8)
            elif elapsed < 0.3: raw_content_frame = np.zeros((BASE_CONTENT_HEIGHT, BASE_CONTENT_WIDTH, 3), dtype=np.uint8)
            else: current_state = STATE_PROCESSING

        elif current_state == STATE_PROCESSING:
            contours, _, _, _, crop_dims = process_frame_to_contours(frame_for_processing, blur_val, block_val, c_val, min_line_length)
            save_svg(contours, crop_dims[0] if crop_dims else 0, crop_dims[1] if crop_dims else 0)
            cap.release()
            if controls_visible: cv2.destroyWindow("ARTIST CONTROLS")
            return VECTOR_OUTPUT_FILE

        if raw_content_frame is not None:
            content_h, content_w, _ = raw_content_frame.shape
            win_h, win_w, _ = display_frame.shape
            scale = min(win_w / content_w, win_h / content_h)
            sw, sh = max(1, int(content_w * scale)), max(1, int(content_h * scale))
            resized = cv2.resize(raw_content_frame, (sw, sh), interpolation=cv2.INTER_AREA)
            px, py = (win_w - sw) // 2, (win_h - sh) // 2
            if py >= 0 and px >= 0:
                 display_frame[py:py+sh, px:px+sw] = resized

        cv2.imshow(MAIN_WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('a'):
            if not controls_visible:
                create_control_panel(); controls_visible = True
            else:
                cv2.destroyWindow("ARTIST CONTROLS"); controls_visible = False
        elif key == 13 and current_state == STATE_PREVIEW: # 13 is Enter
            if last_valid_contours:
                current_state, animation_start_time, countdown_number = STATE_COUNTDOWN, time.time(), 3
            else: print("\nRobot says: Get your face in the frame first!")

    cap.release()
    if controls_visible: cv2.destroyWindow("ARTIST CONTROLS")
    selfie_segmentation.close(); face_detection.close()
    return None

if __name__ == "__main__":
    svg_file = run_capture_ui()
    if svg_file: print(f"Capture UI finished, generated: {svg_file}")
    else: print("Capture UI was quit by the user.")
    cv2.destroyAllWindows()