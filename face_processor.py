import math
import os
import cv2
import dlib
import numpy as np
from scipy.stats import linregress
import sys # Import sys module for PyInstaller path handling

# ---------------------------- Configuration -----------------------------
# Determine Dlib model path based on whether it's a PyInstaller executable
if getattr(sys, 'frozen', False):
    # If it's a bundled exe, model path is in the temporary directory
    PREDICTOR_PATH = os.path.join(sys._MEIPASS, "model", "shape_predictor_68_face_landmarks.dat")
else:
    # Otherwise, it's a development environment
    PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"

MAX_SIDE = 10000 # Scaling threshold
# -----------------------------------------------------------------------

# Core image processing function
def process_and_draw_face_geometry(image_path,
                                    hsv_lower=(5, 53, 92),
                                    hsv_upper=(18, 140, 225),
                                    ycbcr_lower=(95, 135, 80),
                                    ycbcr_upper=(220, 165, 116),
                                    ref_real_width_mm_at_row=170.0):
    """
    Processes a face image to draw geometric features,
    allowing customization of skin detection thresholds.

    Args:
        image_path (str): Path to the input image file.
        hsv_lower (tuple): Lower bound for HSV skin detection (H, S, V).
        hsv_upper (tuple): Upper bound for HSV skin detection (H, S, V).
        ycbcr_lower (tuple): Lower bound for YCbCr skin detection (Y, Cb, Cr).
        ycbcr_upper (tuple): Upper bound for YCbCr skin detection (Y, Cb, Cr).

    Returns:
        numpy.ndarray: The processed image with drawn features.
    """
    # --- Define constants and helper functions within the function ---
    THICK = 2
    TAN_COLOR = (0, 0, 255) # Red

    # Parameters for draw_angle function
    ARC_COLOR = (0, 0, 255) # Red
    ARC_THICK = 4
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 1.2
    TEXT_THICK = 4
    RADIUS = 80
    TEXT_DIST = 120
    BG_COLOR = (255, 255, 255) # White text background
    BG_ALPHA = 0.6 # Translucency

    def draw_extended(pt1, pt2, color, thickness, img_ref):
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        norm = np.hypot(dx, dy) + 1e-6
        ux, uy = dx / norm, dy / norm
        L_local = max(img_ref.shape[:2]) * 2
        x1, y1 = int(pt1[0] - ux * L_local), int(pt1[1] - uy * L_local)
        x2, y2 = int(pt2[0] + ux * L_local), int(pt2[1] + uy * L_local)
        cv2.line(img_ref, (x1, y1), (x2, y2), color, thickness)

    def draw_parallel(pt1, pt2, through, color, thickness, img_ref):
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        norm = np.hypot(dx, dy) + 1e-6
        ux, uy = dx / norm, dy / norm
        L_local = max(img_ref.shape[:2]) * 2
        x1, y1 = int(through[0] - ux * L_local), int(through[1] - uy * L_local)
        x2, y2 = int(through[0] + ux * L_local), int(through[1] + uy * L_local)
        cv2.line(img_ref, (x1, y1), (x2, y2), color, thickness)
        return (ux, uy)

    def is_supporting_line(P, Q, contour):
        v = np.array(Q) - np.array(P)
        signs = []
        for R in contour:
            w = np.array(R) - np.array(P)
            cross = v[0] * w[1] - v[1] * w[0]
            signs.append(cross)
        signs = np.array(signs)
        return np.all(signs >= -1e-6) or np.all(signs <= 1e-6)

    def draw_angle(img_ref, center, vec1, vec2, color, prefer_obtuse=False):
        cosang = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))
        if angle > 90:
            angle = 180 - angle

        a1 = math.degrees(math.atan2(vec1[1], vec1[0])) % 360
        a2 = math.degrees(math.atan2(vec2[1], vec2[0])) % 360
        diff = (a2 - a1) % 360
        if diff > 180:
            diff -= 360
        if abs(diff) > 180 - abs(diff):
            diff = -np.sign(diff) * (360 - abs(diff))
        if abs(diff) > 90:
            diff = -np.sign(diff) * (180 - abs(diff))
        start = a1
        end = (a1 + diff) % 360

        cv2.ellipse(img_ref, tuple(center), (RADIUS, RADIUS),
                    0, float(start), float(end), color, ARC_THICK)

        txt = f"{angle:.3f}deg"
        (w, h), _ = cv2.getTextSize(txt, FONT, TEXT_SCALE, TEXT_THICK)
        mid_ang = math.radians(start + diff / 2)
        bx = int(center[0] + math.cos(mid_ang) * (RADIUS + TEXT_DIST)) - w // 2
        by = int(center[1] + math.sin(mid_ang) * (RADIUS + TEXT_DIST)) + h // 2

        overlay = img_ref.copy()
        cv2.rectangle(overlay, (bx - 5, by - h - 5), (bx + w + 5, by + 5), BG_COLOR, -1)
        cv2.addWeighted(overlay, BG_ALPHA, img_ref, 1 - BG_ALPHA, 0, img_ref)

        cv2.putText(img_ref, txt, (bx, by), FONT, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)

        return angle
    # --- Constants and helper functions definition end ---

    # 1. Validate files
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"❌ Dlib model file not found: {PREDICTOR_PATH}. Please ensure it's downloaded and placed in the model/ directory.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    # 2. Read image & scale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {image_path}. Please check file path or format.")

    h0, w0 = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h0, w0))
    img_small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # --- Define LARGE after img is loaded ---
    LARGE = max(img.shape[:2]) * 2
    # --- LARGE definition end ---

    # 3. Face landmark detection
    gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    faces = detector(gray_small.copy())
    if not faces:
        raise RuntimeError("No face detected. Please ensure the image contains a clear face.")
    face = faces[0]
    shape = predictor(gray_small, face)
    pts_small = np.array([[p.x, p.y] for p in shape.parts()])
    pts = (pts_small / scale).astype(int)

    # 4. Extract key points
    left_eyebrow_pts = pts[17:22]
    right_eyebrow_pts = pts[22:27]
    left_eye_inner = pts[39]
    left_eye_outer = pts[36]
    right_eye_inner = pts[42]
    right_eye_outer = pts[45]
    nose_bridge_pts = pts[27:31]
    pt_lip_top_center = pts[51]
    pt_lip_bottom_center = pts[57]
    pt_chin_bottom = pts[8]
    left_corner = tuple(pts[48])
    right_corner = tuple(pts[54])
    pt_nose = tuple(pts[27]) # Ensure pt_nose is defined here for subsequent boundary calculations

    # --- Facial midline calculation (moved up to calculate para_vec) ---
    midline_fit_points = []
    if len(nose_bridge_pts) > 0:
        midline_fit_points.extend(nose_bridge_pts)
    if len(left_eyebrow_pts) > 0 and len(right_eyebrow_pts) > 0:
        midline_fit_points.append(np.mean(left_eyebrow_pts, axis=0).astype(int))
        midline_fit_points.append(np.mean(right_eyebrow_pts, axis=0).astype(int))
    midline_fit_points.append(((left_eye_inner + left_eye_outer) / 2).astype(int))
    midline_fit_points.append(((right_eye_inner + right_eye_outer) / 2).astype(int))
    midline_fit_points.append(pt_lip_top_center)
    midline_fit_points.append(pt_lip_bottom_center)
    midline_fit_points.append(pt_chin_bottom)

    if not midline_fit_points:
        raise RuntimeError("Could not find enough key points to fit the facial midline.")

    midline_fit_points_np = np.array(midline_fit_points)
    avg_midline_x = int(np.mean(midline_fit_points_np[:, 0]))

    y_coords_for_regress = midline_fit_points_np[:, 1]
    x_coords_for_regress = midline_fit_points_np[:, 0]
    slope_prime, intercept_prime, r_value, p_value, std_err = linregress(y_coords_for_regress, x_coords_for_regress)

    midline_y_top_limit = min(p[1] for p in (pts[19], pts[24])) - 50
    midline_y_bottom_limit = pts[8][1] + 50

    pt_midline_top_y = midline_y_top_limit
    pt_midline_bottom_y = midline_y_bottom_limit

    pt_midline_top_x = int(slope_prime * pt_midline_top_y + intercept_prime)
    pt_midline_bottom_x = int(slope_prime * pt_midline_bottom_y + intercept_prime)

    pt_midline_top = (pt_midline_top_x, pt_midline_top_y)
    pt_midline_bottom = (pt_midline_bottom_x, pt_midline_bottom_y)

    img_h, img_w = img.shape[:2]
    pt_midline_top = (max(0, min(img_w - 1, pt_midline_top[0])), max(0, min(img_h - 1, pt_midline_top[1])))
    pt_midline_bottom = (max(0, min(img_w - 1, pt_midline_bottom[0])), max(0, min(img_h - 1, pt_midline_bottom[1])))

    # Calculate para_vec (before draw_angle is defined)
    midline_vec_x = pt_midline_bottom[0] - pt_midline_top[0]
    midline_vec_y = pt_midline_bottom[1] - pt_midline_top[1]
    norm_midline = np.hypot(midline_vec_x, midline_vec_y) + 1e-6
    para_vec = np.array([midline_vec_x / norm_midline, midline_vec_y / norm_midline])
    # --- Facial midline calculation and para_vec end ---

    # 5. Draw main line (now facial midline) & parallel lines on original image
    out = img.copy()

    # Draw the new facial midline in magenta (255,0,255)
    draw_extended(pt_midline_top, pt_midline_bottom, (255, 0, 255), 3, out)

    # --- Visualize key points used for midline calculation ---
    for p_fit in midline_fit_points:
        cv2.circle(out, tuple(p_fit), 5, (255, 0, 255), -1)

    cv2.circle(out, pt_midline_top, 6, (255, 255, 255), -1)
    cv2.circle(out, pt_midline_bottom, 6, (255, 255, 255), -1)
    # --- Key point visualization end ---

    # ux, uy are already calculated as components of para_vec, use them directly
    ux, uy = para_vec[0], para_vec[1]

    # Left corner parallel line
    draw_parallel(pt_midline_top, pt_midline_bottom, left_corner, (0, 255, 255), 2, out)
    # Right corner parallel line (ux,uy are the same)
    draw_parallel(pt_midline_top, pt_midline_bottom, right_corner, (0, 255, 255), 2, out)

    # 6. Detect tray: simple BGR white threshold
    mask_white = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No white tray detected. Please ensure the image contains a clear white tray.")
    tray = max(contours, key=cv2.contourArea)
    y_tray = min(pt[1] for pt in tray.reshape(-1, 2))

    # 7. Draw 4th horizontal line
    cv2.line(out, (0, y_tray), (img.shape[1], y_tray), (0, 255, 0), 2)

    # 8. Calculate intersection points: solving (x,y_tray) lies on parallel lines through left/right corners
    ints = []
    for through in (left_corner, right_corner):
        t = (y_tray - through[1]) / (uy + 1e-6)
        x_int = int(through[0] + ux * t)
        ints.append((x_int, y_tray))
        cv2.circle(out, (x_int, y_tray), 6, (0, 0, 255), -1)

    # --- New face contour extraction method ---
    img_orig_copy = img.copy()

    # 1. Skin color detection (HSV space)
    hsv = cv2.cvtColor(img_orig_copy, cv2.COLOR_BGR2HSV)
    lower_skin_np = np.array(hsv_lower, dtype="uint8")
    upper_skin_np = np.array(hsv_upper, dtype="uint8")
    skin_mask = cv2.inRange(hsv, lower_skin_np, upper_skin_np)

    # 2. Auxiliary detection (YCbCr space)
    ycbcr = cv2.cvtColor(img_orig_copy, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycbcr_np = np.array(ycbcr_lower, dtype="uint8")
    upper_skin_ycbcr_np = np.array(ycbcr_upper, dtype="uint8")
    skin_mask_ycbcr = cv2.inRange(ycbcr, lower_skin_ycbcr_np, upper_skin_ycbcr_np)

    # 3. Combine both masks
    final_skin_mask = cv2.bitwise_and(skin_mask, skin_mask_ycbcr)

    # 4. Morphological operations: remove noise, fill small holes, connect broken regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_skin_mask = cv2.morphologyEx(final_skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_skin_mask = cv2.morphologyEx(final_skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 5. Find the largest connected component (face)
    contours, _ = cv2.findContours(final_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    face_contour_pts = []
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        face_contour_pts = [tuple(p[0]) for p in approx_contour]

    if not face_contour_pts:
        # Fallback: use dlib landmarks jawline (0..16) when color mask fails
        jaw_pts = pts[:17].tolist()
    else:
        jaw_pts = face_contour_pts

    # Visualize the new contour line (green)
    cv2.polylines(out, [np.array(jaw_pts, dtype=np.int32)], True, (0, 255, 0), 3)

    # Temporary visualization of boundary lines
    jawline_bottom_y = pts[8][1] - 50
    cheek_upper_bound_y = min(p[1] for p in (pts[3], pts[4], pts[5], pts[11], pts[12], pts[13])) - 50
    pt_nose = tuple(pts[27])
    cheek_upper_bound_y = max(cheek_upper_bound_y, pt_nose[1])
    cv2.line(out, (0, jawline_bottom_y), (img.shape[1], jawline_bottom_y), (0, 0, 0), 2)
    cv2.line(out, (0, cheek_upper_bound_y), (img.shape[1], cheek_upper_bound_y), (0, 0, 0), 2)

    # 13. Calculate for left and right intersection points separately
    angles = [None, None]
    for side, P in enumerate(ints):
        tangents = []

        current_cheek_upper_y = cheek_upper_bound_y

        is_left_side_P = (P[0] < avg_midline_x)

        for Q_candidate in jaw_pts:
            if Q_candidate[1] > P[1] and Q_candidate[1] < jawline_bottom_y and Q_candidate[1] > current_cheek_upper_y:
                if is_left_side_P:
                    if Q_candidate[0] < P[0]:
                        if is_supporting_line(P, Q_candidate, jaw_pts):
                            tangents.append((Q_candidate, Q_candidate))
                else:
                    if Q_candidate[0] > P[0]:
                        if is_supporting_line(P, Q_candidate, jaw_pts):
                            tangents.append((Q_candidate, Q_candidate))

        if not tangents:
            for Q_full_contour in jaw_pts:
                if Q_full_contour[1] < jawline_bottom_y and Q_full_contour[1] > current_cheek_upper_y:
                    if is_left_side_P:
                        if Q_full_contour[0] < P[0]:
                            if is_supporting_line(P, Q_full_contour, jaw_pts):
                                tangents.append((Q_full_contour, Q_full_contour))
                    else:
                        if Q_full_contour[0] > P[0]:
                            if is_supporting_line(P, Q_full_contour, jaw_pts):
                                tangents.append((Q_full_contour, Q_full_contour))
            if not tangents:
                print(f"⚠️ Warning: No suitable cheek support line found for intersection point {P}. Skipping angle drawing.")
                continue

        _, Q = min(tangents, key=lambda x: np.hypot(x[1][0] - P[0], x[1][1] - P[1]))

        tan_vec = (np.array(Q) - np.array(P)) / (np.linalg.norm(np.array(Q) - np.array(P)) + 1e-6)

        v = np.array(Q) - np.array(P)
        v = v / (np.linalg.norm(v) + 1e-6)

        # --- Store baseline jaw tangent for this side ---
        try:
            side_tangents
        except NameError:
            side_tangents = [None, None]
        side_tangents[side] = {'P': P, 'v': v}

        pt1 = (int(P[0] - v[0] * LARGE), int(P[1] - v[1] * LARGE))
        pt2 = (int(P[0] + v[0] * LARGE), int(P[1] + v[1] * LARGE))
        cv2.line(out, pt1, pt2, TAN_COLOR, THICK)
        cv2.circle(out, Q, 12, TAN_COLOR, -1)

        draw_angle(out, P, para_vec, tan_vec, ARC_COLOR)
        angle=draw_angle(out, P, para_vec, tan_vec, ARC_COLOR)
        angles[side] = angle

        # --- Additional measurements: lower-face width (horizontal mouth line) and side angles (upper only) ---
        # 1) Lower-face width: horizontal line at average mouth-corner y, intersect with jaw contour
        y_mouth_h = int(0.5*(left_corner[1] + right_corner[1]))
        # Draw the mouth horizontal reference
        cv2.line(out, (0, y_mouth_h), (img_w, y_mouth_h), (255, 255, 0), 2)
        xs = []
        for (x1,y1),(x2,y2) in zip(jaw_pts, jaw_pts[1:] + jaw_pts[:1]):
            if (y1 - y_mouth_h) * (y2 - y_mouth_h) <= 0 and (y1 != y2):
                # linear interpolation to find intersection x
                x = x1 + (y_mouth_h - y1) * (x2 - x1) / (y2 - y1 + 1e-6)
                xs.append(x)
        lower_face_width_px = None
        left_intersect = None
        right_intersect = None
        if len(xs) >= 2:
            xs_sorted = sorted(xs)
            xL, xR = int(xs_sorted[0]), int(xs_sorted[-1])
            left_intersect = (xL, y_mouth_h)
            right_intersect = (xR, y_mouth_h)
            lower_face_width_px = float(abs(xR - xL))
            # visualize
            cv2.circle(out, left_intersect, 8, (255, 0, 0), -1)
            cv2.circle(out, right_intersect, 8, (255, 0, 0), -1)
            cv2.line(out, left_intersect, right_intersect, (255, 0, 0), 4)

        # mm conversion using provided real width for this row (default 170 mm)
        mm_per_px = float(ref_real_width_mm_at_row) / float(img_w) if ref_real_width_mm_at_row else None
        lower_face_width_mm = (lower_face_width_px * mm_per_px) if (mm_per_px and lower_face_width_px is not None) else None

        # annotate lower face width in cm on the image
        if lower_face_width_mm is not None and left_intersect and right_intersect:
            lw_cm = lower_face_width_mm / 10.0
            midx = int((left_intersect[0] + right_intersect[0]) / 2)
            midy = y_mouth_h
            label = f"{lw_cm:.2f} cm"
            (tw, th), _ = cv2.getTextSize(label, FONT, 1.0, 2)
            bx, by = midx + 10, midy - 10
            cv2.rectangle(out, (bx - 4, by - th - 6), (bx + tw + 4, by + 4), (0,0,0), -1)
            cv2.putText(out, label, (bx, by), FONT, 1.0, (0,0,255), 2, cv2.LINE_AA)


        # 2) Side angles (upper tangent from S = intersection of baseline jaw tangent with mouth horizontal)
        side_angles_deg = {'left': None, 'right': None}
        def compute_side_angle_for(side_idx):
            info = side_tangents[side_idx] if 'side_tangents' in locals() else None
            if info is None:
                return None
            P0 = info['P']; v0 = info['v']
            # S: intersection with mouth horizontal; handle near-horizontal v0 robustly
            if abs(v0[1]) < 1e-6:
                S = (int(P0[0]), y_mouth_h)
            else:
                tS = (y_mouth_h - P0[1]) / (v0[1] + 1e-6)
                S = (int(P0[0] + v0[0]*tS), y_mouth_h)
            cv2.circle(out, S, 6, (128, 0, 255), -1)
            # find upper supporting tangent from S
            best = None
            for Qc in jaw_pts:
                if Qc[1] >= y_mouth_h:
                    continue
                if is_supporting_line(S, Qc, jaw_pts):
                    d2 = (Qc[0]-S[0])**2 + (Qc[1]-S[1])**2
                    if best is None or d2 < best[0]:
                        best = (d2, Qc)
            if best is None:
                # fallback: nearest upper point
                cand = [Q for Q in jaw_pts if Q[1] < y_mouth_h]
                if cand:
                    Qc = min(cand, key=lambda q: (q[0]-S[0])**2 + (q[1]-S[1])**2)
                    best = ((Qc[0]-S[0])**2 + (Qc[1]-S[1])**2, Qc)
                else:
                    return None
            Qs = best[1]
            vs = np.array(Qs, dtype=float) - np.array(S, dtype=float)
            vs = vs / (np.linalg.norm(vs)+1e-6)
            pt1s = (int(S[0] - vs[0]*LARGE), int(S[1] - vs[1]*LARGE))
            pt2s = (int(S[0] + vs[0]*LARGE), int(S[1] + vs[1]*LARGE))
            cv2.line(out, pt1s, pt2s, (255, 128, 0), 2)
            # draw obtuse arc at S
            draw_angle(out, S, v0, vs, (255, 128, 0), prefer_obtuse=True)
            # numeric obtuse angle
            cosang = float(np.dot(vs, v0) / (np.linalg.norm(vs)*np.linalg.norm(v0) + 1e-6))
            cosang = max(min(cosang, 1.0), -1.0)
            ang = math.degrees(math.acos(cosang))
            ang = 180.0 - min(ang, 180.0 - ang)
            return ang
        side_angles_deg['left'] = compute_side_angle_for(0)
        side_angles_deg['right'] = compute_side_angle_for(1)

        # package metrics
        metrics = {
            'lower_face_width_px': lower_face_width_px,
            'lower_face_width_mm': lower_face_width_mm,
            'mm_per_px': mm_per_px,
            'mouth_line': {'y': y_mouth_h, 'left_intersect': left_intersect, 'right_intersect': right_intersect},
            'side_angles_deg': side_angles_deg
        }
        # 14. 返回处理图 + 左右角（None 表示未测到）
        left_angle = angles[0] if angles[0] is not None else 0.0
        right_angle = angles[1] if angles[1] is not None else 0.0
    return out, left_angle, right_angle, metrics # Return the processed image and metrics
