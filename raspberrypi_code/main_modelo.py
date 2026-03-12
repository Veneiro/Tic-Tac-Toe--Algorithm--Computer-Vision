from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# ==========================================
# CONFIGURACIÓN (RANGOS DE COLOR)
# ==========================================
RED_L1 = np.array([0,   70, 55])
RED_U1 = np.array([12, 255, 255])
RED_L2 = np.array([163, 70, 55])
RED_U2 = np.array([180, 255, 255])
BLUE_L = np.array([85,  35, 75])
BLUE_U = np.array([118, 255, 255])
WHITE_L = np.array([0,   0,  165])
WHITE_U = np.array([180, 55, 255])
COLOR_THRESH = 0.055
FOCUS_MIN_VAR = 85.0
TARGET_WARP_SIDE = 520
DEBUG_OUTPUT_DIR = 'debug_steps'
DEBUG_SAVE_STEPS_DEFAULT = True

T_SOFTMAX = 0.5
SEARCH_DEPTH = 2

class DebugSaver:
    def __init__(self, enabled=False, root_dir=DEBUG_OUTPUT_DIR):
        self.enabled = bool(enabled)
        self.root_dir = root_dir
        self.run_dir = None
        self.step_counter = 0

        if self.enabled:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.run_dir = os.path.join(self.root_dir, f'run_{timestamp}')
            os.makedirs(self.run_dir, exist_ok=True)

    def _next_name(self, name):
        self.step_counter += 1
        return f'{self.step_counter:03d}_{name}'

    def save_image(self, name, img):
        if not self.enabled or img is None:
            return None

        filename = self._next_name(name)
        path = os.path.join(self.run_dir, f'{filename}.png')
        cv2.imwrite(path, img)
        return path

    def save_mask(self, name, mask):
        if not self.enabled or mask is None:
            return None

        mask_u8 = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
        return self.save_image(name, mask_u8)

    def save_text(self, name, data):
        if not self.enabled:
            return None

        filename = self._next_name(name)
        path = os.path.join(self.run_dir, f'{filename}.txt')
        with open(path, 'w', encoding='utf-8') as file:
            file.write(str(data))
        return path

def parse_debug_flag(flag_value):
    if flag_value is None:
        return DEBUG_SAVE_STEPS_DEFAULT

    value = str(flag_value).strip().lower()
    if value in {'1', 'true', 'yes', 'si', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    return DEBUG_SAVE_STEPS_DEFAULT

def order_points(pts):
    pts = np.array(pts, dtype="float32").reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def focus_variance(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def unsharp_mask(img, sigma=1.1, amount=1.35):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def red_score(hsv_patch, bgr_patch):
    m1 = cv2.inRange(hsv_patch, RED_L1, RED_U1)
    m2 = cv2.inRange(hsv_patch, RED_L2, RED_U2)
    strong_mask = cv2.bitwise_or(m1, m2)

    red_relaxed_l1 = np.array([0, 45, 40], dtype=np.uint8)
    red_relaxed_u1 = np.array([15, 255, 255], dtype=np.uint8)
    red_relaxed_l2 = np.array([160, 45, 40], dtype=np.uint8)
    red_relaxed_u2 = np.array([180, 255, 255], dtype=np.uint8)
    mr1 = cv2.inRange(hsv_patch, red_relaxed_l1, red_relaxed_u1)
    mr2 = cv2.inRange(hsv_patch, red_relaxed_l2, red_relaxed_u2)
    relaxed_mask = cv2.bitwise_or(mr1, mr2)

    b, g, r = cv2.split(bgr_patch)
    red_dom = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16) > 22) & (r > 55)

    area = max(hsv_patch.shape[0] * hsv_patch.shape[1], 1)
    strong_ratio = np.sum(strong_mask > 0) / area
    relaxed_ratio = np.sum(relaxed_mask > 0) / area
    red_dom_ratio = np.sum(red_dom) / area

    return max(strong_ratio, 0.85 * relaxed_ratio, 0.9 * red_dom_ratio)

def blue_score(hsv_patch, bgr_patch):
    strong_mask = cv2.inRange(hsv_patch, BLUE_L, BLUE_U)

    blue_relaxed_l = np.array([80, 25, 55], dtype=np.uint8)
    blue_relaxed_u = np.array([125, 255, 255], dtype=np.uint8)
    relaxed_mask = cv2.inRange(hsv_patch, blue_relaxed_l, blue_relaxed_u)

    b, g, r = cv2.split(bgr_patch)
    blue_dom = (b.astype(np.int16) - np.maximum(g, r).astype(np.int16) > 18) & (b > 50)

    area = max(hsv_patch.shape[0] * hsv_patch.shape[1], 1)
    strong_ratio = np.sum(strong_mask > 0) / area
    relaxed_ratio = np.sum(relaxed_mask > 0) / area
    blue_dom_ratio = np.sum(blue_dom) / area

    return max(strong_ratio, 0.85 * relaxed_ratio, 0.9 * blue_dom_ratio)

def _clean_mask(mask, side):
    k = max(2, int(side * 0.06))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def _largest_blob_features(mask):
    h, w = mask.shape[:2]
    area_total = max(h * w, 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 1.0, 0.0

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0.0:
        return 0.0, 1.0, 0.0

    moments = cv2.moments(cnt)
    if moments['m00'] > 1e-6:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = w * 0.5, h * 0.5

    center_dist = np.hypot(cx - (w * 0.5), cy - (h * 0.5))
    center_norm = center_dist / max(np.hypot(w * 0.5, h * 0.5), 1e-6)

    bx, by, bw, bh = cv2.boundingRect(cnt)
    box_area = max(float(bw * bh), 1.0)
    extent = area / box_area
    return area / area_total, center_norm, extent

def token_confidence(patch_hsv, patch_bgr, color='red'):
    h, w = patch_hsv.shape[:2]
    side = min(h, w)

    if color == 'red':
        m1 = cv2.inRange(patch_hsv, RED_L1, RED_U1)
        m2 = cv2.inRange(patch_hsv, RED_L2, RED_U2)
        strong = cv2.bitwise_or(m1, m2)
        raw_score = red_score(patch_hsv, patch_bgr)
    else:
        strong = cv2.inRange(patch_hsv, BLUE_L, BLUE_U)
        raw_score = blue_score(patch_hsv, patch_bgr)

    cleaned = _clean_mask(strong, side)
    blob_ratio, center_norm, extent = _largest_blob_features(cleaned)

    center_weight = max(0.0, 1.0 - center_norm)
    shape_weight = min(1.0, max(0.0, (extent - 0.2) / 0.6))
    blob_weight = min(1.0, blob_ratio / 0.11)

    return 0.45 * raw_score + 0.35 * blob_weight + 0.15 * center_weight + 0.05 * shape_weight

def white_fraction(bgr_patch):
    return np.sum(cv2.inRange(cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV), WHITE_L, WHITE_U) > 0) / max(bgr_patch.shape[0] * bgr_patch.shape[1], 1)

def analyze_precapture(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    brightness = float(np.mean(v))
    contrast = float(np.std(v))
    focus_var = focus_variance(gray)
    highlights = float(np.mean((v > 245) & (s < 70)))
    shadows = float(np.mean(v < 40))

    issues = []
    if brightness < 60:
        issues.append('oscura')
    elif brightness > 210:
        issues.append('sobreexpuesta')
    if contrast < 30:
        issues.append('poco_contraste')
    if highlights > 0.08:
        issues.append('reflejos')
    if shadows > 0.22:
        issues.append('sombras_fuertes')
    if focus_var < FOCUS_MIN_VAR:
        issues.append('desenfoque')

    quality_ok = len(issues) == 0
    return {
        'brightness': brightness,
        'contrast': contrast,
        'focus_var': focus_var,
        'highlights_ratio': highlights,
        'shadows_ratio': shadows,
        'issues': issues,
        'ok': quality_ok,
    }

def apply_adaptive_enhancement(img, capture_info):
    enhanced = img.copy()
    brightness = capture_info['brightness']
    contrast = capture_info['contrast']
    focus_var = capture_info['focus_var']

    if brightness < 80:
        gamma = 0.78
    elif brightness > 200:
        gamma = 1.25
    else:
        gamma = 1.0

    if abs(gamma - 1.0) > 1e-3:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)

    if contrast < 38:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if focus_var < FOCUS_MIN_VAR:
        enhanced = unsharp_mask(enhanced)

    return enhanced

def warp_panel(img, debugger=None):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    line_len = max(20, int(min(h, w) * 0.10))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vertical = cv2.morphologyEx(black, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.morphologyEx(black, cv2.MORPH_OPEN, h_kernel)
    line_mask = cv2.bitwise_or(vertical, horizontal)

    k_join = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(min(h, w) * 0.02)), max(3, int(min(h, w) * 0.02))))
    line_joined = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k_join)
    line_joined = cv2.dilate(line_joined, k_join, iterations=2)

    x_proj = np.sum(vertical > 0, axis=0)
    y_proj = np.sum(horizontal > 0, axis=1)

    x_proj_inner = x_proj.copy()
    y_proj_inner = y_proj.copy()
    margin_x = max(8, int(w * 0.07))
    margin_y = max(8, int(h * 0.07))
    x_proj_inner[:margin_x] = 0
    x_proj_inner[-margin_x:] = 0
    y_proj_inner[:margin_y] = 0
    y_proj_inner[-margin_y:] = 0

    x_pair = _pick_internal_lines(x_proj_inner, w)
    y_pair = _pick_internal_lines(y_proj_inner, h)

    line_based_warp = None
    line_based_src = None
    cell_w = None
    cell_h = None
    area_ratio = None
    spacing_ok = False
    touches_outer = None
    if x_pair is not None and y_pair is not None:
        x1, x2 = x_pair
        y1, y2 = y_pair
        cell_w = max(12, x2 - x1)
        cell_h = max(12, y2 - y1)

        min_cell_w = int(w * 0.06)
        max_cell_w = int(w * 0.52)
        min_cell_h = int(h * 0.06)
        max_cell_h = int(h * 0.52)
        spacing_ok = (
            min_cell_w <= cell_w <= max_cell_w and
            min_cell_h <= cell_h <= max_cell_h
        )

        x0 = max(0, int(x1 - cell_w * 1.02))
        x3 = min(w - 1, int(x2 + cell_w * 1.02))
        y0 = max(0, int(y1 - cell_h * 1.02))
        y3 = min(h - 1, int(y2 + cell_h * 1.02))

        bw = x3 - x0
        bh = y3 - y0
        ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        area_ratio = (bw * bh) / max(float(w * h), 1.0)

        outer_margin = max(8, int(min(w, h) * 0.02))
        touches_outer = (
            x0 <= outer_margin or y0 <= outer_margin or
            x3 >= (w - outer_margin) or y3 >= (h - outer_margin)
        )

        if (
            bw > 40 and bh > 40 and
            ratio > 0.62 and
            0.08 < area_ratio < 0.80 and
            spacing_ok and
            not touches_outer
        ):
            src_guess = np.float32([[x0, y0], [x3, y0], [x3, y3], [x0, y3]])
            side = TARGET_WARP_SIDE
            dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
            M_line = cv2.getPerspectiveTransform(src_guess, dst)
            warped_line = cv2.warpPerspective(img, M_line, (side, side), flags=cv2.INTER_CUBIC)

            line_based_warp = (warped_line, M_line)
            line_based_src = src_guess

    if debugger is not None:
        debugger.save_mask('03_panel_black_mask', black)
        debugger.save_mask('04_panel_vertical_lines', vertical)
        debugger.save_mask('05_panel_horizontal_lines', horizontal)
        debugger.save_mask('06_panel_line_mask', line_mask)
        debugger.save_mask('07_panel_line_joined', line_joined)
        debugger.save_text('08_panel_line_pairs', {
            'x_pair': x_pair,
            'y_pair': y_pair,
            'line_based_available': line_based_warp is not None,
            'line_based_reason': {
                'cell_w': int(cell_w) if cell_w is not None else None,
                'cell_h': int(cell_h) if cell_h is not None else None,
                'spacing_ok': bool(spacing_ok),
                'area_ratio': float(area_ratio) if area_ratio is not None else None,
                'touches_outer': bool(touches_outer) if touches_outer is not None else None,
            }
        })

    if line_based_warp is not None:
        warped_line, M_line = line_based_warp
        if debugger is not None:
            overlay = img.copy()
            poly = np.int32(line_based_src)
            cv2.polylines(overlay, [poly], isClosed=True, color=(255, 0, 255), thickness=3)
            debugger.save_image('09_panel_contour', overlay)
            debugger.save_image('10_warped_panel', warped_line)
        return warped_line, M_line

    cnts, _ = cv2.findContours(line_joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_cnt = None
    panel_rect = None
    best_score = -1.0
    candidate_debug = []
    border_margin = max(8, int(min(h, w) * 0.03))

    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.015:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        ratio = min(bw, bh) / max(bw, bh)
        if ratio < 0.55:
            continue

        touches_border = (
            bx < border_margin or by < border_margin or
            (bx + bw) > (w - border_margin) or (by + bh) > (h - border_margin)
        )

        area_ratio = float(area / max(h * w, 1))
        if touches_border and area_ratio > 0.85:
            candidate_debug.append({
                'bbox': (int(bx), int(by), int(bw), int(bh)),
                'area': float(area),
                'area_ratio': float(area_ratio),
                'ratio': float(ratio),
                'touches_border': bool(touches_border),
                'score': -1.0,
                'discarded': 'extreme_border_and_too_large',
            })
            continue

        roi_lines = line_mask[by:by + bh, bx:bx + bw]
        if roi_lines.size == 0:
            continue

        line_ratio = float(np.mean(roi_lines > 0))

        cx = bx + (bw * 0.5)
        cy = by + (bh * 0.5)
        center_dist = np.hypot(cx - (w * 0.5), cy - (h * 0.5))
        center_score = 1.0 - min(1.0, center_dist / max(np.hypot(w * 0.5, h * 0.5), 1e-6))

        area_norm = min(1.0, area / max((h * w) * 0.22, 1.0))
        score = (
            0.25 * area_norm +
            0.22 * ratio +
            2.80 * line_ratio +
            0.18 * center_score
        )
        if touches_border:
            score -= 0.20

        candidate_debug.append({
            'bbox': (int(bx), int(by), int(bw), int(bh)),
            'area': float(area),
            'area_ratio': float(area_ratio),
            'ratio': float(ratio),
            'line_ratio': float(line_ratio),
            'touches_border': bool(touches_border),
            'score': float(score),
        })

        if score > best_score:
            best_score = score
            panel_cnt = cnt
            panel_rect = cv2.minAreaRect(cnt)

    if debugger is not None:
        candidate_debug = sorted(candidate_debug, key=lambda item: float(item.get('score', -1.0)), reverse=True)[:6]
        debugger.save_text('11_panel_candidates', candidate_debug)
            
    if panel_cnt is None or best_score < 0.14:
        ys, xs = np.where(line_joined > 0)
        if xs.size > 40 and ys.size > 40:
            x0, x1 = int(np.min(xs)), int(np.max(xs))
            y0, y1 = int(np.min(ys)), int(np.max(ys))
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            pad_x = max(8, int(bw * 0.12))
            pad_y = max(8, int(bh * 0.12))
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(w - 1, x1 + pad_x)
            y1 = min(h - 1, y1 + pad_y)
        else:
            side_fallback = int(min(h, w) * 0.78)
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - side_fallback // 2)
            y0 = max(0, cy - side_fallback // 2)
            x1 = min(w - 1, x0 + side_fallback)
            y1 = min(h - 1, y0 + side_fallback)

        src_fb = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        side = TARGET_WARP_SIDE
        dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        M_fb = cv2.getPerspectiveTransform(src_fb, dst)
        warped_fb = cv2.warpPerspective(img, M_fb, (side, side), flags=cv2.INTER_CUBIC)

        if debugger is not None:
            overlay = img.copy()
            cv2.polylines(overlay, [np.int32(src_fb)], isClosed=True, color=(0, 255, 255), thickness=3)
            status_msg = 'Fallback por line_joined' if panel_cnt is None else f'Fallback por score bajo: {best_score:.4f}'
            debugger.save_text('12_panel_status', status_msg)
            debugger.save_image('13_panel_contour', overlay)
            debugger.save_image('14_warped_panel', warped_fb)

        return warped_fb, M_fb

    box = cv2.boxPoints(panel_rect)
    
    # --- CORRECCIÓN AQUÍ ---
    box = np.int32(box) 
    # -----------------------
    
    src = order_points(box)
    side = TARGET_WARP_SIDE
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    M   = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (side, side), flags=cv2.INTER_CUBIC)

    if debugger is not None:
        overlay = img.copy()
        cv2.drawContours(overlay, [panel_cnt], -1, (0, 255, 0), 3)
        cv2.polylines(overlay, [box], isClosed=True, color=(255, 0, 0), thickness=2)
        debugger.save_image('13_panel_contour', overlay)
        debugger.save_image('14_warped_panel', warped)

    return warped, M

def find_grid_bbox(warped, debugger=None):
    side = warped.shape[0]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(4, int(side * 0.016))
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    black_d = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k_dil)

    line_len = max(18, int(side * 0.14))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vertical = cv2.morphologyEx(black_d, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.morphologyEx(black_d, cv2.MORPH_OPEN, h_kernel)
    line_mask = cv2.bitwise_or(vertical, horizontal)

    k_join = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k_size // 2), max(3, k_size // 2)))
    line_mask_joined = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k_join)
    line_mask_joined = cv2.dilate(line_mask_joined, k_join, iterations=1)

    border_margin = max(10, int(side * 0.08))
    line_mask_inner = line_mask_joined.copy()
    line_mask_inner[:border_margin, :] = 0
    line_mask_inner[-border_margin:, :] = 0
    line_mask_inner[:, :border_margin] = 0
    line_mask_inner[:, -border_margin:] = 0

    if debugger is not None:
        debugger.save_mask('08_grid_black_mask', black)
        debugger.save_mask('09_grid_black_closed', black_d)
        debugger.save_mask('10_grid_vertical_lines', vertical)
        debugger.save_mask('11_grid_horizontal_lines', horizontal)
        debugger.save_mask('12_grid_line_mask', line_mask_inner)

    best_box = None

    ys, xs = np.where(line_mask_inner > 0)
    if ys.size > 0 and xs.size > 0:
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        area = bw * bh
        ratio = min(bw, bh) / max(bw, bh)
        touches_border = (x0 <= border_margin or y0 <= border_margin or x1 >= side - border_margin - 1 or y1 >= side - border_margin - 1)
        too_large = bw >= int(side * 0.92) or bh >= int(side * 0.92)

        if area >= int((side ** 2) * 0.05) and ratio >= 0.68 and not touches_border and not too_large:
            pad_x = max(4, int(bw * 0.05))
            pad_y = max(4, int(bh * 0.05))
            gx = max(0, x0 - pad_x)
            gy = max(0, y0 - pad_y)
            gw = min(side - gx, bw + 2 * pad_x)
            gh = min(side - gy, bh + 2 * pad_y)
            best_box = (gx, gy, gw, gh)

    cnts, _ = cv2.findContours(black_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts[:8]:
        if best_box is not None:
            break
        area = cv2.contourArea(cnt)
        if area < (side ** 2) * 0.05: continue
        gx, gy, gw, gh = cv2.boundingRect(cnt)
        if gx <= border_margin or gy <= border_margin or (gx + gw) >= (side - border_margin) or (gy + gh) >= (side - border_margin):
            continue
        if gw >= int(side * 0.92) or gh >= int(side * 0.92):
            continue
        if min(gw, gh) / max(gw, gh) < 0.7: continue
        interior = warped[gy:gy + gh, gx:gx + gw]
        if white_fraction(interior) > 0.4:
            best_box = (gx, gy, gw, gh)
            break
            
    if best_box is None:
        pad = int(side * 0.12)
        best_box = (pad, pad, side - 2 * pad, side - 2 * pad)

    if debugger is not None:
        gx, gy, gw, gh = best_box
        bbox_preview = warped.copy()
        cv2.rectangle(bbox_preview, (gx, gy), (gx + gw, gy + gh), (0, 255, 255), 2)
        debugger.save_image('13_grid_bbox', bbox_preview)

    return best_box

def sanitize_board_turn_consistency(board, confidence):
    flat = []
    for row in range(3):
        for col in range(3):
            mark = board[row][col]
            if mark is not None:
                flat.append((row, col, mark, float(confidence[row, col])))

    count_x = sum(1 for _, _, mark, _ in flat if mark == 'X')
    count_o = sum(1 for _, _, mark, _ in flat if mark == 'O')

    if count_x == 0 or count_o == 0:
        return board

    while abs(count_x - count_o) > 1:
        dominant = 'X' if count_x > count_o else 'O'
        candidates = [(r, c, conf) for (r, c, mark, conf) in flat if mark == dominant]
        if not candidates:
            break
        r, c, _ = min(candidates, key=lambda item: item[2])
        board[r][c] = None
        confidence[r, c] = 0.0
        flat = [(rr, cc, mm, cf) for (rr, cc, mm, cf) in flat if not (rr == r and cc == c)]
        count_x = sum(1 for _, _, mark, _ in flat if mark == 'X')
        count_o = sum(1 for _, _, mark, _ in flat if mark == 'O')

    return board

def _pick_internal_lines(projection, length):
    if projection.size == 0:
        return None

    proj = projection.astype(np.float32).reshape(1, -1)
    smooth = cv2.GaussianBlur(proj, (0, 0), sigmaX=2.0).ravel()
    peak = float(np.max(smooth)) if smooth.size else 0.0
    if peak <= 1e-6:
        return None

    thr = max(peak * 0.45, np.mean(smooth) + np.std(smooth) * 0.6)
    idx = np.where(smooth >= thr)[0]
    if idx.size == 0:
        return None

    groups = []
    start = idx[0]
    prev = idx[0]
    for value in idx[1:]:
        if value <= prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = value
        prev = value
    groups.append((start, prev))

    centers = [int((a + b) / 2) for (a, b) in groups if (b - a + 1) >= 2]
    if len(centers) < 2:
        return None

    exp1 = length / 3.0
    exp2 = 2.0 * length / 3.0
    expected_spacing = length / 3.0
    min_spacing = max(8, int(length * 0.08))
    max_spacing = max(min_spacing + 2, int(length * 0.72))
    best_pair = None
    best_cost = float('inf')
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1, c2 = centers[i], centers[j]
            spacing = c2 - c1
            if spacing < min_spacing or spacing > max_spacing:
                continue

            pair_mid = 0.5 * (c1 + c2)
            center_cost = abs(pair_mid - (length * 0.5))
            spacing_cost = abs(spacing - expected_spacing)
            thirds_cost = abs(c1 - exp1) + abs(c2 - exp2)

            cost = 0.55 * thirds_cost + 0.30 * spacing_cost + 0.15 * center_cost
            if cost < best_cost:
                best_cost = cost
                best_pair = (c1, c2)

    return best_pair

def infer_grid_edges(warped, gx, gy, gw, gh):
    roi = warped[gy:gy + gh, gx:gx + gw]
    if roi.size == 0:
        return [gx, gx + gw // 3, gx + (2 * gw) // 3, gx + gw], [gy, gy + gh // 3, gy + (2 * gh) // 3, gy + gh], None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    side = max(gw, gh)
    line_len = max(14, int(side * 0.16))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vertical = cv2.morphologyEx(black, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.morphologyEx(black, cv2.MORPH_OPEN, h_kernel)

    x_proj = np.sum(vertical > 0, axis=0)
    y_proj = np.sum(horizontal > 0, axis=1)

    x_pair = _pick_internal_lines(x_proj, gw)
    y_pair = _pick_internal_lines(y_proj, gh)

    if x_pair is not None:
        x1, x2 = x_pair
        cell_w = max(10, x2 - x1)
        x0 = max(0, int(x1 - cell_w))
        x3 = min(gw, int(x2 + cell_w))
        if x3 - x0 < max(36, int(gw * 0.55)):
            x0, x1, x2, x3 = 0, gw // 3, (2 * gw) // 3, gw
    else:
        x0, x1, x2, x3 = 0, gw // 3, (2 * gw) // 3, gw

    if y_pair is not None:
        y1, y2 = y_pair
        cell_h = max(10, y2 - y1)
        y0 = max(0, int(y1 - cell_h))
        y3 = min(gh, int(y2 + cell_h))
        if y3 - y0 < max(36, int(gh * 0.55)):
            y0, y1, y2, y3 = 0, gh // 3, (2 * gh) // 3, gh
    else:
        y0, y1, y2, y3 = 0, gh // 3, (2 * gh) // 3, gh

    x_edges = [gx + x0, gx + x1, gx + x2, gx + x3]
    y_edges = [gy + y0, gy + y1, gy + y2, gy + y3]
    return x_edges, y_edges, (vertical, horizontal)

def classify_cells(warped, gx, gy, gw, gh, debugger=None):
    side = warped.shape[0]
    whsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    wbgr = warped
    board = [[None] * 3 for _ in range(3)]
    confidence = np.zeros((3, 3), dtype=float)
    x_edges, y_edges, line_maps = infer_grid_edges(warped, gx, gy, gw, gh)
    cell_h = max(1, min(y_edges[1] - y_edges[0], y_edges[2] - y_edges[1], y_edges[3] - y_edges[2]))
    cell_w = max(1, min(x_edges[1] - x_edges[0], x_edges[2] - x_edges[1], x_edges[3] - x_edges[2]))
    MARGIN_H, MARGIN_W = max(int(cell_h * 0.09), 3), max(int(cell_w * 0.09), 3)
    grid_gray = cv2.cvtColor(warped[gy:gy + gh, gx:gx + gw], cv2.COLOR_BGR2GRAY)
    focus_on_grid = focus_variance(grid_gray)
    grid_saturation = float(np.mean(whsv[gy:gy + gh, gx:gx + gw, 1]))
    dominance_ratio = 1.16 if focus_on_grid < FOCUS_MIN_VAR else 1.26

    gx_eff, gy_eff = x_edges[0], y_edges[0]
    gw_eff, gh_eff = x_edges[3] - x_edges[0], y_edges[3] - y_edges[0]

    red_scores = np.zeros((3, 3), dtype=float)
    blue_scores = np.zeros((3, 3), dtype=float)
    white_scores = np.ones((3, 3), dtype=float)
    mask_ratios = np.zeros((3, 3), dtype=float)
    valid_cell = np.zeros((3, 3), dtype=bool)
    used_center = np.zeros((3, 3), dtype=bool)

    for row in range(3):
        for col in range(3):
            y1 = max(0, y_edges[row] + MARGIN_H)
            y2 = min(side, y_edges[row + 1] - MARGIN_H)
            x1 = max(0, x_edges[col] + MARGIN_W)
            x2 = min(side, x_edges[col + 1] - MARGIN_W)
            
            patch_hsv = whsv[y1:y2, x1:x2]
            patch_bgr = wbgr[y1:y2, x1:x2]
            if patch_hsv.size == 0: continue

            if debugger is not None:
                debugger.save_image(f'cell_{row}_{col}_patch', patch_bgr)

            ph, pw = patch_hsv.shape[:2]
            yy, xx = np.ogrid[:ph, :pw]
            cy, cx = ph * 0.5, pw * 0.5
            radius = min(ph, pw) * 0.48
            center_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
            if np.count_nonzero(center_mask) < 12:
                continue

            center_hsv = patch_hsv.copy()
            center_bgr = patch_bgr.copy()
            center_hsv[~center_mask] = 0
            center_bgr[~center_mask] = 0

            rs_center = token_confidence(center_hsv, center_bgr, color='red')
            bs_center = token_confidence(center_hsv, center_bgr, color='blue')
            rs_full = token_confidence(patch_hsv, patch_bgr, color='red')
            bs_full = token_confidence(patch_hsv, patch_bgr, color='blue')

            center_peak = max(rs_center, bs_center)
            full_peak = max(rs_full, bs_full)
            use_center = center_peak >= full_peak * 0.92
            analysis_hsv = center_hsv if use_center else patch_hsv
            analysis_bgr = center_bgr if use_center else patch_bgr

            rs = rs_center if use_center else rs_full
            bs = bs_center if use_center else bs_full
            center_white_ratio = white_fraction(analysis_bgr)

            red_scores[row, col] = rs
            blue_scores[row, col] = bs
            white_scores[row, col] = center_white_ratio
            mask_ratios[row, col] = float(np.count_nonzero(center_mask) / max(ph * pw, 1))
            valid_cell[row, col] = True
            used_center[row, col] = use_center

            if debugger is not None:
                red_m1 = cv2.inRange(analysis_hsv, RED_L1, RED_U1)
                red_m2 = cv2.inRange(analysis_hsv, RED_L2, RED_U2)
                red_mask = cv2.bitwise_or(red_m1, red_m2)
                blue_mask = cv2.inRange(analysis_hsv, BLUE_L, BLUE_U)
                debugger.save_image(f'cell_{row}_{col}_center', center_bgr)
                debugger.save_mask(f'cell_{row}_{col}_red_mask', red_mask)
                debugger.save_mask(f'cell_{row}_{col}_blue_mask', blue_mask)

    valid_red = red_scores[valid_cell]
    valid_blue = blue_scores[valid_cell]

    if valid_red.size > 0:
        red_q50 = float(np.percentile(valid_red, 50))
        red_q75 = float(np.percentile(valid_red, 75))
        blue_q50 = float(np.percentile(valid_blue, 50))
        blue_q75 = float(np.percentile(valid_blue, 75))
    else:
        red_q50 = red_q75 = 0.0
        blue_q50 = blue_q75 = 0.0

    red_thresh = max(0.12, min(0.32, 0.35 * red_q50 + 0.65 * red_q75))
    blue_thresh = max(0.18, min(0.42, 0.35 * blue_q50 + 0.65 * blue_q75))

    if grid_saturation < 95:
        red_thresh -= 0.02
        blue_thresh -= 0.01
    if focus_on_grid < FOCUS_MIN_VAR:
        red_thresh -= 0.015
        blue_thresh -= 0.015

    red_thresh = max(0.11, red_thresh)
    blue_thresh = max(0.16, blue_thresh)

    for row in range(3):
        for col in range(3):
            if not valid_cell[row, col]:
                continue

            rs = float(red_scores[row, col])
            bs = float(blue_scores[row, col])
            center_white_ratio = float(white_scores[row, col])

            decision = None
            if center_white_ratio > 0.92 and max(rs, bs) < max(red_thresh, blue_thresh) * 0.85:
                decision = None
            elif rs >= red_thresh and rs >= bs * dominance_ratio:
                decision = 'X'
                confidence[row, col] = rs
            elif bs >= blue_thresh and bs >= rs * dominance_ratio:
                decision = 'O'
                confidence[row, col] = bs
            else:
                relaxed_dom = max(1.08, dominance_ratio - 0.12)
                if rs >= red_thresh * 0.86 and rs >= bs * relaxed_dom and rs >= 0.11:
                    decision = 'X'
                    confidence[row, col] = rs
                elif bs >= blue_thresh * 0.86 and bs >= rs * relaxed_dom and bs >= 0.15:
                    decision = 'O'
                    confidence[row, col] = bs

            board[row][col] = decision

            if debugger is not None:
                debugger.save_text(
                    f'cell_{row}_{col}_scores',
                    {
                        'red_score': rs,
                        'blue_score': bs,
                        'white_ratio': center_white_ratio,
                        'red_threshold': float(red_thresh),
                        'blue_threshold': float(blue_thresh),
                        'dominance_ratio': float(dominance_ratio),
                        'mask_area_ratio': float(mask_ratios[row, col]),
                        'margin_h': int(MARGIN_H),
                        'margin_w': int(MARGIN_W),
                        'grid_saturation': float(grid_saturation),
                        'analysis_mode': 'center' if bool(used_center[row, col]) else 'full_patch',
                        'decision': decision,
                    }
                )
            
    board = sanitize_board_turn_consistency(board, confidence)

    if debugger is not None:
        preview = warped.copy()
        cv2.rectangle(preview, (gx_eff, gy_eff), (gx_eff + gw_eff, gy_eff + gh_eff), (0, 255, 255), 2)
        if line_maps is not None:
            vertical, horizontal = line_maps
            line_preview = np.zeros_like(warped)
            line_preview[gy:gy + gh, gx:gx + gw, 1] = vertical
            line_preview[gy:gy + gh, gx:gx + gw, 2] = horizontal
            blend = cv2.addWeighted(warped, 0.75, line_preview, 0.9, 0)
            debugger.save_image('10_grid_line_overlay', blend)

        for xv in x_edges[1:3]:
            cv2.line(preview, (xv, y_edges[0]), (xv, y_edges[3]), (255, 0, 255), 1)
        for yv in y_edges[1:3]:
            cv2.line(preview, (x_edges[0], yv), (x_edges[3], yv), (255, 0, 255), 1)

        for row in range(3):
            for col in range(3):
                x_center = (x_edges[col] + x_edges[col + 1]) // 2
                y_center = (y_edges[row] + y_edges[row + 1]) // 2
                label = board[row][col] if board[row][col] is not None else '-'
                cv2.putText(preview, label, (x_center - 12, y_center + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        debugger.save_image('11_classification_preview', preview)
        debugger.save_text('12_classification_board', board)

    return board

def board_to_string(board):
    mapping = {None: '0', 'X': '1', 'O': '2'}
    rows = []
    for row in board:
        rows.append(','.join(mapping[cell] for cell in row))
    return 'tablero={' + ';'.join(rows) + '}'

def possible_moves_numeric(board):
    aux = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == 0:
                aux.append((i, j))
    return aux

def evaluate_numeric(board):
    lines = []
    lines.extend(board)
    lines.extend(board.T)
    lines.append(np.diag(board))
    lines.append(np.diag(np.fliplr(board)))

    score = 0
    for line in lines:
        if np.all(line == 2):
            return 100
        if np.all(line == 1):
            return -100

        if np.count_nonzero(line == 2) == 2 and np.count_nonzero(line == 0) == 1:
            score += 10
        if np.count_nonzero(line == 1) == 2 and np.count_nonzero(line == 0) == 1:
            score -= 8

    return score

def minimax_numeric(board, depth, maximizing):
    score = evaluate_numeric(board)

    if abs(score) == 100 or depth == 0:
        return score

    moves = possible_moves_numeric(board)
    if not moves:
        return score

    if maximizing:
        best = -np.inf
        for (i, j) in moves:
            b = board.copy()
            b[i, j] = 2
            best = max(best, minimax_numeric(b, depth - 1, False))
        return best

    best = np.inf
    for (i, j) in moves:
        b = board.copy()
        b[i, j] = 1
        best = min(best, minimax_numeric(b, depth - 1, True))
    return best

def evaluate_moves_numeric(board, depth):
    moves = possible_moves_numeric(board)
    scores = []

    for (i, j) in moves:
        new_board = board.copy()
        new_board[i, j] = 2
        score = minimax_numeric(new_board, depth, maximizing=False)
        scores.append(score)

    return moves, np.array(scores, dtype=float)

def softmax(scores, temperature=T_SOFTMAX):
    if len(scores) == 0:
        return np.array([])
    safe_t = max(float(temperature), 1e-6)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted / safe_t)
    return exp_scores / np.sum(exp_scores)

def choose_move_softmax(board, temperature=T_SOFTMAX, base_depth=SEARCH_DEPTH):
    empty = int(np.count_nonzero(board == 0))

    if empty >= 7:
        depth = max(1, base_depth - 1)
        temp = temperature + 0.3
    elif empty >= 4:
        depth = base_depth
        temp = temperature
    else:
        depth = base_depth + 2
        temp = max(0.05, temperature - 0.3)

    moves, scores = evaluate_moves_numeric(board, depth)
    if len(moves) == 0:
        return None, None, depth, temp, []

    noisy_scores = scores + np.random.normal(0, 0.3, size=len(scores))
    probs = softmax(noisy_scores, temp)
    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx], float(noisy_scores[idx]), depth, temp, probs.tolist()

# ==========================================
# SERVIDOR WEB (FLASK)
# ==========================================
@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        data = request.data
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None: return "Error imagen", 400

        debug_enabled = parse_debug_flag(request.args.get('debug'))
        debugger = DebugSaver(enabled=debug_enabled)
        debugger.save_image('01_original_image', img)

        print("Recibida imagen. Analizando precaptura...")
        capture_info = analyze_precapture(img)
        debugger.save_text('02_precapture_info', capture_info)
        preprocessed = apply_adaptive_enhancement(img, capture_info)
        debugger.save_image('03_preprocessed_image', preprocessed)

        warped, _ = warp_panel(preprocessed, debugger=debugger)
        if warped is None: 
            print("No se encontró tablero")
            if debugger.enabled:
                print(f"Debug guardado en: {debugger.run_dir}")
            return "tablero={Error: No Panel}", 200
        
        gx, gy, gw, gh = find_grid_bbox(warped, debugger=debugger)
        board = classify_cells(warped, gx, gy, gw, gh, debugger=debugger)
        resultado = board_to_string(board)
        
        print(f"Precaptura ok={capture_info['ok']} issues={capture_info['issues']}")
        print(f"RESULTADO: {resultado}")
        if debugger.enabled:
            print(f"Debug guardado en: {debugger.run_dir}")
        return resultado

    except Exception as e:
        print(f"Error: {e}")
        return "Error Server", 500

@app.route('/precaptura', methods=['POST'])
def precaptura():
    try:
        data = request.data
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Error imagen'}), 400

        capture_info = analyze_precapture(img)
        return jsonify(capture_info), 200

    except Exception as e:
        print(f"Error en /precaptura: {e}")
        return jsonify({'error': 'Error Server'}), 500

@app.route('/movimiento', methods=['POST'])
def movimiento():
    try:
        payload = request.get_json(silent=True)
        if not payload or 'matriz' not in payload:
            return jsonify({'error': 'Debes enviar JSON con la clave "matriz"'}), 400

        matrix = np.array(payload['matriz'])
        if matrix.shape != (3, 3):
            return jsonify({'error': 'La matriz debe ser de tamaño 3x3'}), 400

        if not np.isin(matrix, [0, 1, 2]).all():
            return jsonify({'error': 'La matriz solo puede contener valores 0, 1 y 2'}), 400

        matrix = matrix.astype(int)
        move, score, depth, temperature, probabilities = choose_move_softmax(matrix)

        if move is None:
            return jsonify({
                'movimiento': None,
                'score': None,
                'mensaje': 'No hay movimientos posibles'
            }), 200

        return jsonify({
            'movimiento': {'fila': int(move[0]), 'columna': int(move[1])},
            'score': score,
            'depth': depth,
            'temperatura': temperature,
            'probabilidades': probabilities
        }), 200

    except Exception as e:
        print(f"Error en /movimiento: {e}")
        return jsonify({'error': 'Error Server'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)