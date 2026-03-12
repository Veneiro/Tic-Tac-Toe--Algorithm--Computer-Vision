"""
capturador_ml.py
----------------
Servidor Flask simple para probar detección de tablero usando modelo ML de celdas
(en lugar de umbrales de color de classify_cells).

Recibe imagen por POST /procesar (bytes JPEG/PNG), detecta tablero, clasifica 9 celdas
con el modelo y guarda imagen + estado en disco.

Uso:
  python capturador_ml.py
  python capturador_ml.py --model ml_models/cell_classifier_best.joblib --port 5000 --out capturas_ml
"""

import argparse
import os
from collections import deque
from datetime import datetime

import cv2
import joblib
import numpy as np
from flask import Flask, request
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from main import (
        analyze_precapture,
        apply_adaptive_enhancement,
        warp_panel,
        find_grid_bbox,
        infer_grid_edges,
        board_to_string,
        sanitize_board_turn_consistency,
    )
except ImportError:
    from main_modelo import (
        analyze_precapture,
        apply_adaptive_enhancement,
        warp_panel,
        find_grid_bbox,
        infer_grid_edges,
        board_to_string,
        sanitize_board_turn_consistency,
    )


INT_TO_LABEL = {0: None, 1: 'X', 2: 'O'}

OUTPUT_DIR = 'capturas_ml'
MODEL_PATH = 'ml_models/cell_classifier_best.joblib'
MODEL = None
CONF_MIN_X = 0.30
CONF_MIN_O = 0.42
TEMPORAL_HISTORY_SIZE = 1
ENABLE_TTA = True
BOARD_HISTORY = deque(maxlen=TEMPORAL_HISTORY_SIZE)

app = Flask(__name__)


class ColorFeatureSelector(BaseEstimator, TransformerMixin):
    """Compatibilidad para modelos antiguos serializados con __main__.ColorFeatureSelector."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        gray_end = 24 * 24
        return x[:, gray_end:]


def extract_features(image):
    """Extrae 625 features: gris(576) + H-hist(24) + S-hist(8) + V-hist(8) + color_explicito(9)."""
    resized = cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    h_hist = h_hist / max(np.sum(h_hist), 1e-6)
    s_hist = s_hist / max(np.sum(s_hist), 1e-6)
    v_hist = v_hist / max(np.sum(v_hist), 1e-6)

    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)
    total = float(24 * 24)

    sat_mask = s_ch > 45
    red_mask = ((h_ch < 18) | (h_ch > 162)) & sat_mask
    blue_mask = (h_ch > 88) & (h_ch < 132) & sat_mask
    green_mask = (h_ch > 38) & (h_ch < 86) & sat_mask

    red_ratio = float(np.sum(red_mask)) / total
    blue_ratio = float(np.sum(blue_mask)) / total
    green_ratio = float(np.sum(green_mask)) / total

    white_mask = (v_ch > 175) & (s_ch < 55)
    white_ratio = float(np.sum(white_mask)) / total
    mean_sat = float(np.mean(s_ch)) / 255.0

    center = resized[8:16, 8:16]
    c_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    c_h = c_hsv[:, :, 0].astype(np.float32)
    c_s = c_hsv[:, :, 1].astype(np.float32)
    c_total = float(8 * 8)
    c_sat = c_s > 45
    c_red = ((c_h < 18) | (c_h > 162)) & c_sat
    c_blue = (c_h > 88) & (c_h < 132) & c_sat
    c_red_ratio = float(np.sum(c_red)) / c_total
    c_blue_ratio = float(np.sum(c_blue)) / c_total
    c_mean_s = float(np.mean(c_s)) / 255.0
    c_mean_v = float(np.mean(c_hsv[:, :, 2])) / 255.0

    color_feats = np.array([
        red_ratio, blue_ratio, green_ratio, white_ratio, mean_sat,
        c_red_ratio, c_blue_ratio, c_mean_s, c_mean_v,
    ], dtype=np.float32)

    gray_flat = gray.flatten()
    return np.concatenate([gray_flat, h_hist, s_hist, v_hist, color_feats], axis=0)


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No existe el modelo: {path}')
    return joblib.load(path)


def _extract_cell_patches_from_warp(warped, gx, gy, gw, gh):
    x_edges, y_edges, _ = infer_grid_edges(warped, gx, gy, gw, gh)

    cell_h = max(1, min(y_edges[1] - y_edges[0], y_edges[2] - y_edges[1], y_edges[3] - y_edges[2]))
    cell_w = max(1, min(x_edges[1] - x_edges[0], x_edges[2] - x_edges[1], x_edges[3] - x_edges[2]))
    margin_h = max(int(cell_h * 0.09), 3)
    margin_w = max(int(cell_w * 0.09), 3)

    h, w = warped.shape[:2]
    patches = []
    for row in range(3):
        for col in range(3):
            y1 = max(0, y_edges[row] + margin_h)
            y2 = min(h, y_edges[row + 1] - margin_h)
            x1 = max(0, x_edges[col] + margin_w)
            x2 = min(w, x_edges[col + 1] - margin_w)
            if y2 <= y1 or x2 <= x1:
                patches.append((row, col, None))
            else:
                patches.append((row, col, warped[y1:y2, x1:x2]))

    return patches


def _int_to_labels(board_int):
    return [[INT_TO_LABEL.get(int(board_int[r][c]), None) for c in range(3)] for r in range(3)]


def _labels_to_int(board_labels):
    out = [[0] * 3 for _ in range(3)]
    for row in range(3):
        for col in range(3):
            val = board_labels[row][col]
            if val == 'X':
                out[row][col] = 1
            elif val == 'O':
                out[row][col] = 2
            else:
                out[row][col] = 0
    return out


def _tta_variants(patch):
    variants = [patch]
    if not ENABLE_TTA:
        return variants

    variants.append(cv2.convertScaleAbs(patch, alpha=1.08, beta=6))
    variants.append(cv2.convertScaleAbs(patch, alpha=0.92, beta=-6))
    variants.append(cv2.GaussianBlur(patch, (3, 3), 0))
    return variants


def _predict_cell_ml(patch):
    if patch is None or patch.size == 0:
        return 0, 0.0

    variants = _tta_variants(patch)
    probs = []
    preds = []

    for img in variants:
        feat = extract_features(img).astype(np.float32).reshape(1, -1)
        pred = int(MODEL.predict(feat)[0])
        preds.append(pred)

        if hasattr(MODEL, 'predict_proba'):
            prob = np.asarray(MODEL.predict_proba(feat)[0], dtype=np.float32)
            probs.append(prob)

    if probs:
        mean_prob = np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)
        prob_sum = float(np.sum(mean_prob))
        if prob_sum > 1e-8:
            mean_prob = mean_prob / prob_sum

        pred_int = int(np.argmax(mean_prob))
        conf = float(np.max(mean_prob))
        return pred_int, conf

    counts = np.bincount(np.array(preds, dtype=np.int32), minlength=3)
    pred_int = int(np.argmax(counts))
    conf = float(np.max(counts) / max(len(preds), 1))
    return pred_int, conf


def _apply_confidence_floor(board_int, confidences):
    out = [row[:] for row in board_int]
    for row in range(3):
        for col in range(3):
            value = out[row][col]
            conf = float(confidences[row][col])
            if value == 1 and conf < CONF_MIN_X:
                out[row][col] = 0
                confidences[row][col] = 0.0
            elif value == 2 and conf < CONF_MIN_O:
                out[row][col] = 0
                confidences[row][col] = 0.0
    return out


def _temporal_vote(current_board_int):
    if TEMPORAL_HISTORY_SIZE <= 1:
        BOARD_HISTORY.append(np.array(current_board_int, dtype=np.int32))
        return current_board_int

    BOARD_HISTORY.append(np.array(current_board_int, dtype=np.int32))
    stack = np.stack(list(BOARD_HISTORY), axis=0)

    voted = [[0] * 3 for _ in range(3)]
    for row in range(3):
        for col in range(3):
            values = stack[:, row, col]
            counts = np.bincount(values, minlength=3)
            voted[row][col] = int(np.argmax(counts))
    return voted


def _predict_board_ml(warped):
    gx, gy, gw, gh = find_grid_bbox(warped)
    patches = _extract_cell_patches_from_warp(warped, gx, gy, gw, gh)

    board_int = [[0] * 3 for _ in range(3)]
    confidences = [[0.0] * 3 for _ in range(3)]

    for row, col, patch in patches:
        pred_int, conf = _predict_cell_ml(patch)

        board_int[row][col] = pred_int
        confidences[row][col] = conf

    board_int = _apply_confidence_floor(board_int, confidences)

    board_labels = _int_to_labels(board_int)
    conf_np = np.array(confidences, dtype=np.float32)
    board_labels = sanitize_board_turn_consistency(board_labels, conf_np)
    board_int = _labels_to_int(board_labels)

    board_int_temporal = _temporal_vote(board_int)
    board_labels_temporal = _int_to_labels(board_int_temporal)

    return board_labels_temporal, board_int_temporal, confidences, board_int


def _save_run(img: np.ndarray, estado: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    cv2.imwrite(os.path.join(run_dir, 'original_image.png'), img)
    with open(os.path.join(run_dir, 'estado_tablero.txt'), 'w', encoding='utf-8') as file:
        file.write(estado + '\n')

    return run_dir


@app.route('/procesar', methods=['POST'])
def procesar_ml():
    try:
        print('\n[capturador_ml] ── Nueva petición ─────────────────────────────')
        data = request.data
        print(f'[capturador_ml] Bytes recibidos: {len(data)}')

        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print('[capturador_ml] ERROR: imagen no decodificable')
            return 'Error: imagen no decodificable', 400

        print(f'[capturador_ml] Imagen: {img.shape[1]}x{img.shape[0]}')

        capture_info = analyze_precapture(img)
        preprocessed = apply_adaptive_enhancement(img, capture_info)

        warped, _ = warp_panel(preprocessed)
        if warped is None:
            estado = 'tablero={Error: No Panel}'
            print('[capturador_ml] No se detectó tablero')
        else:
            board_labels, board_int, confs, board_int_single = _predict_board_ml(warped)
            estado = board_to_string(board_labels)
            print(f'[capturador_ml] Estado ML (final): {estado}')
            print(f'[capturador_ml] Matriz int (single): {board_int_single}')
            print(f'[capturador_ml] Matriz int (temporal): {board_int}')
            print(f'[capturador_ml] Confianzas: {[[round(x, 3) for x in row] for row in confs]}')

        run_dir = _save_run(img, estado)
        print(f'[capturador_ml] Guardado en: {run_dir}')
        print('[capturador_ml] ── Fin ───────────────────────────────────────\n')
        return estado, 200

    except Exception as exc:
        import traceback
        print(f'[capturador_ml] ERROR: {exc}')
        traceback.print_exc()
        return 'Error Server', 500


@app.route('/reset_history', methods=['POST'])
def reset_history():
    BOARD_HISTORY.clear()
    print('[capturador_ml] Historial temporal reiniciado')
    return 'ok', 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capturador con clasificación ML de celdas')
    parser.add_argument('--model', default='ml_models/cell_classifier_best.joblib', help='Ruta del modelo .joblib')
    parser.add_argument('--port', type=int, default=5000, help='Puerto HTTP')
    parser.add_argument('--out', default='capturas_ml', help='Carpeta de salida de runs')
    parser.add_argument('--conf-min-x', type=float, default=0.30, help='Confianza mínima para aceptar X')
    parser.add_argument('--conf-min-o', type=float, default=0.42, help='Confianza mínima para aceptar O')
    parser.add_argument('--history', type=int, default=1, help='Nº de tableros para votación temporal (1 desactiva)')
    parser.add_argument('--disable-tta', action='store_true', help='Desactiva test-time augmentation por celda')
    args = parser.parse_args()

    MODEL_PATH = args.model
    OUTPUT_DIR = args.out
    CONF_MIN_X = float(args.conf_min_x)
    CONF_MIN_O = float(args.conf_min_o)
    TEMPORAL_HISTORY_SIZE = max(1, int(args.history))
    ENABLE_TTA = not bool(args.disable_tta)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL = _load_model(MODEL_PATH)
    BOARD_HISTORY = deque(maxlen=TEMPORAL_HISTORY_SIZE)

    print(f'Modelo cargado: {MODEL_PATH}')
    print(f'Capturador ML arrancando en http://0.0.0.0:{args.port}/procesar')
    print(f'Guardando runs en: {os.path.abspath(OUTPUT_DIR)}')
    print(f'Configuración ML: conf_min_x={CONF_MIN_X}, conf_min_o={CONF_MIN_O}, history={TEMPORAL_HISTORY_SIZE}, tta={ENABLE_TTA}')

    app.run(host='0.0.0.0', port=args.port)
