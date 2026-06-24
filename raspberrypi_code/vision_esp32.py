"""
vision_esp32.py
Detector Tic-Tac-Toe en tiempo real (Roboflow + MediaPipe) + servidor Flask para ESP32-CAM.

Modos:
    python vision_esp32.py                    # live + Flask en background (default)
    python vision_esp32.py --modo server      # solo servidor Flask
    python vision_esp32.py --modo live        # solo ventana en tiempo real (sin Flask)

Controles en ventana live:
    q  — salir
    m  — toggle MediaPipe landmarks
    y  — toggle detecciones Roboflow
    c  — toggle labels de confianza

Respuesta de /procesar (JSON):
    {
      "tablero":    "{0,0,0;1,0,0;0,2,0}",
      "left":       "{0,1,0,0,0}",
      "right":      "{0,0,2,0,0}",
      "fuera_rojo": 1,
      "fuera_azul": 0
    }
"""

import time
import threading
import traceback
import urllib.request
import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    TIENE_MEDIAPIPE = True
except ImportError:
    TIENE_MEDIAPIPE = False
    print('[WARN] mediapipe no instalado — se omite detección de manos')

try:
    from ultralytics import YOLO as _YOLO
    TIENE_ULTRALYTICS = True
except ImportError:
    TIENE_ULTRALYTICS = False
    print('[WARN] ultralytics no instalado — inferencia local no disponible')


# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────

ROBOFLOW_API_KEY  = 'C6jJ9kSIF8VRiJWR3rFE'
ROBOFLOW_MODEL_ID = 'misovos/7'

GESTURE_MODEL_PATH = 'gesture_recognizer.task'
GESTURE_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
)

DETECT_CONF  = 0.50
MP_DETECT    = 0.70
MP_TRACK     = 0.55
MAX_HANDS    = 2
INFER_WIDTH  = 640

# Inferencia local (YOLO)
USE_LOCAL_MODEL  = True      # True → usa best.pt en lugar de Roboflow
LOCAL_MODEL_PATH = 'C:/Users/mateo/Documents/GitHub/Tic Tac Toe - Algorithm - Computer Vision/raspberrypi_code/runs/detect/runs_ttt/train-3/weights/best.pt'  # ruta al archivo .pt

_yolo_model = None            # se inicializa la primera vez que se llama a infer()

USE_ESP32        = True
ESP32_STREAM_URL = 'http://10.191.81.88:81/stream'
CAMERA_ID        = 0
CAM_W, CAM_H     = 1280, 720

# Servidor Flask
DEFAULT_HOST  = '0.0.0.0'
DEFAULT_PORT  = 5000

# IA (minimax + softmax)
T_SOFTMAX    = 0.5
SEARCH_DEPTH = 2

# ──────────────────────────────────────────────
# Colores BGR
# ──────────────────────────────────────────────

CLASS_COLORS: dict = {
    'red cross':   (40,  40,  220),
    'blue circle': (220, 120,  30),
    'board':       (0,   210, 210),
    'grid':        (0,   160, 255),
    'cells':       (200, 140, 255),
}
CLASS_TEXT_COLOR: dict = {
    'red cross':   (255, 255, 255),
    'blue circle': (255, 255, 255),
    'board':       (0,   0,   0),
    'grid':        (0,   0,   0),
    'cells':       (0,   0,   0),
}
C_FALLBACK_BOX  = (200, 200, 200)
C_FALLBACK_TEXT = (0,   0,   0)
C_FPS       = (0,   255, 255)
C_STATUS    = (180, 180, 180)
C_HAND_SIDE = (255, 255,   0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2),  (2, 3),  (3, 4),
    (0, 5), (5, 6),  (6, 7),  (7, 8),
    (0, 9), (9, 10), (10, 11),(11, 12),
    (0, 13),(13, 14),(14, 15),(15, 16),
    (0, 17),(17, 18),(18, 19),(19, 20),
    (5, 9), (9, 13), (13, 17),
]
FINGERTIP_IDS = {4, 8, 12, 16, 20}

# ──────────────────────────────────────────────
# Detección de gestos
# ──────────────────────────────────────────────

# Gestos reconocidos por el GestureRecognizer estándar de MediaPipe:
#   None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
GESTO_COLORES: dict = {
    'Closed_Fist': ( 50,  50, 200),   # azul oscuro  — puño cerrado
    'Open_Palm':   ( 40, 200,  40),   # verde         — palma abierta
    'Victory':     (  0, 200, 220),   # cian          — dos dedos (V)
    'Pointing_Up': (  0, 170, 255),   # naranja claro — dedo índice arriba
    'Thumb_Up':    ( 20, 220, 120),   # verde-azul    — pulgar arriba
    'Thumb_Down':  ( 60,  60, 200),   # azul-rojo     — pulgar abajo
    'ILoveYou':    (170,  40, 220),   # morado        — índice + meñique + pulgar (ASL)
}
C_GESTO_DEFAULT = (160, 160, 160)


def get_gestos(mp_result) -> list:
    """
    Devuelve [(nombre_gesto, lado, confianza), ...] para cada mano detectada.
    Usa directamente el campo 'gestures' del GestureRecognizerResult de MediaPipe.
    """
    if not mp_result or not mp_result.gestures:
        return []
    gestos = []
    for i, gesture_list in enumerate(mp_result.gestures):
        side = 'Right'
        if mp_result.handedness and i < len(mp_result.handedness):
            side = mp_result.handedness[i][0].category_name
        if gesture_list:
            gestos.append((gesture_list[0].category_name, side, gesture_list[0].score))
    return gestos


# ──────────────────────────────────────────────
# Helpers de dibujo
# ──────────────────────────────────────────────

def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_FPS, 2, cv2.LINE_AA)


def draw_status(frame, show_det: bool, show_mp: bool) -> None:
    h = frame.shape[0]
    text = (f'[y] Detección: {"ON " if show_det else "OFF"}   '
            f'[m] MediaPipe: {"ON " if show_mp else "OFF"}   '
            f'[q] quit')
    cv2.putText(frame, text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_STATUS, 1, cv2.LINE_AA)


def roboflow_infer(frame) -> list:
    """Envía un frame a la REST API de Roboflow y devuelve las predicciones."""
    h, w = frame.shape[:2]
    if w > INFER_WIDTH:
        scale = INFER_WIDTH / w
        small = cv2.resize(frame, (INFER_WIDTH, int(h * scale)))
    else:
        small = frame
        scale = 1.0

    _, buf   = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64  = base64.b64encode(buf).decode('utf-8')
    resp = requests.post(
        f'https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}',
        params={'api_key': ROBOFLOW_API_KEY, 'confidence': DETECT_CONF},
        data=img_b64,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        timeout=5,
    )
    resp.raise_for_status()
    preds = resp.json().get('predictions', [])

    if w > INFER_WIDTH:
        inv = 1.0 / scale
        for p in preds:
            p['x']      *= inv
            p['y']      *= inv
            p['width']  *= inv
            p['height'] *= inv
    return preds


def yolo_infer(frame) -> list:
    """Inferencia local con YOLO. Devuelve predicciones en el mismo formato que roboflow_infer."""
    global _yolo_model
    if _yolo_model is None:
        if not TIENE_ULTRALYTICS:
            raise RuntimeError('ultralytics no instalado — ejecuta: pip install ultralytics')
        print(f'[YOLO] Cargando modelo {LOCAL_MODEL_PATH}…')
        _yolo_model = _YOLO(LOCAL_MODEL_PATH)
        print('[YOLO] Modelo cargado.')

    results = _yolo_model(frame, imgsz=INFER_WIDTH, conf=DETECT_CONF, verbose=False)
    preds = []
    if results:
        boxes = results[0].boxes
        names = results[0].names
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            x, y, w, h = boxes.xywh[i].tolist()
            preds.append({
                'class':      names[cls_id],
                'x':          x,
                'y':          y,
                'width':      w,
                'height':     h,
                'confidence': float(boxes.conf[i].item()),
            })
    return preds


def infer(frame) -> list:
    """Dispatcher: usa YOLO local si USE_LOCAL_MODEL=True, si no llama a Roboflow."""
    if USE_LOCAL_MODEL:
        return yolo_infer(frame)
    return roboflow_infer(frame)


def draw_detections(frame, predictions: list, show_conf: bool = True) -> None:
    for pred in predictions:
        cls_name = pred['class']
        conf     = pred['confidence']
        if conf < DETECT_CONF:
            continue

        x1 = int(pred['x'] - pred['width']  / 2)
        y1 = int(pred['y'] - pred['height'] / 2)
        x2 = int(pred['x'] + pred['width']  / 2)
        y2 = int(pred['y'] + pred['height'] / 2)

        box_color  = CLASS_COLORS.get(cls_name,     C_FALLBACK_BOX)
        text_color = CLASS_TEXT_COLOR.get(cls_name, C_FALLBACK_TEXT)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        label = f'{cls_name} {conf:.2f}' if show_conf else cls_name
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        pad = 4
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - pad * 2),
                      (x1 + tw + pad * 2, y1),
                      box_color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + pad, y1 - baseline - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)


def draw_hand_landmarks(frame, mp_result) -> None:
    if not mp_result or not mp_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_idx, landmarks in enumerate(mp_result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            color = (0, 0, 255) if i in FINGERTIP_IDS else (255, 255, 255)
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

        side = 'Right'
        if mp_result.handedness and hand_idx < len(mp_result.handedness):
            side  = mp_result.handedness[hand_idx][0].category_name
            score = mp_result.handedness[hand_idx][0].score
            wrist = pts[0]
            cv2.putText(frame, f'{side} {score:.2f}',
                        (wrist[0] - 35, wrist[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_HAND_SIDE, 2, cv2.LINE_AA)

        # Etiqueta del gesto encima de la muñeca (del GestureRecognizer nativo)
        gesto = 'None'
        if mp_result.gestures and hand_idx < len(mp_result.gestures) and mp_result.gestures[hand_idx]:
            gesto = mp_result.gestures[hand_idx][0].category_name
        if gesto == 'None':
            continue
        color_g   = GESTO_COLORES.get(gesto, C_GESTO_DEFAULT)
        wrist     = pts[0]
        (tw, th), bl = cv2.getTextSize(gesto, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        lx = wrist[0] - tw // 2
        ly = wrist[1] - 55
        cv2.rectangle(frame, (lx - 5, ly - th - 4), (lx + tw + 5, ly + bl + 2),
                      color_g, cv2.FILLED)
        cv2.putText(frame, gesto, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Análisis del tablero
# ──────────────────────────────────────────────

def _inside(piece: dict, cell: dict) -> bool:
    x1 = cell['x'] - cell['width']  / 2
    y1 = cell['y'] - cell['height'] / 2
    x2 = cell['x'] + cell['width']  / 2
    y2 = cell['y'] + cell['height'] / 2
    return x1 <= piece['x'] <= x2 and y1 <= piece['y'] <= y2


def _grid_bbox(cells: list):
    xs1 = [c['x'] - c['width']  / 2 for c in cells]
    ys1 = [c['y'] - c['height'] / 2 for c in cells]
    xs2 = [c['x'] + c['width']  / 2 for c in cells]
    ys2 = [c['y'] + c['height'] / 2 for c in cells]
    return min(xs1), min(ys1), max(xs2), max(ys2)


def _bbox_grid_desde_predicciones(predictions: list):
    """
    Devuelve (gx1, gy1, gx2, gy2) buscando primero la clase 'grid' y luego
    la clase 'board' (recortando al 65 % central para excluir los laterales).
    Retorna None si no hay ninguna de las dos.
    """
    grids  = [p for p in predictions if p['class'] == 'grid']
    boards = [p for p in predictions if p['class'] == 'board']

    if grids:
        g = max(grids, key=lambda p: p['width'] * p['height'])
        return (g['x'] - g['width']  / 2,
                g['y'] - g['height'] / 2,
                g['x'] + g['width']  / 2,
                g['y'] + g['height'] / 2)

    if boards:
        b = max(boards, key=lambda p: p['width'] * p['height'])
        # El board incluye los laterales de reserva; usamos el 65 % central
        # para aproximar el área de la cuadrícula
        w65 = b['width']  * 0.65 / 2
        h65 = b['height'] * 0.65 / 2
        return (b['x'] - w65, b['y'] - h65,
                b['x'] + w65, b['y'] + h65)

    return None


def _asignar_celdas_virtuales(gx1: float, gy1: float, gx2: float, gy2: float,
                               red_crosses: list, blue_circles: list):
    """
    Divide el bbox (gx1,gy1)-(gx2,gy2) en 9 celdas virtuales 3×3
    y asigna las piezas detectadas a cada celda.
    Devuelve (matrix, used_red, used_blue).
    """
    cw = (gx2 - gx1) / 3
    ch = (gy2 - gy1) / 3

    matrix: list  = [[0] * 3 for _ in range(3)]
    used_red:  set = set()
    used_blue: set = set()

    for ri in range(3):
        for ci in range(3):
            cx = gx1 + (ci + 0.5) * cw
            cy = gy1 + (ri + 0.5) * ch
            virtual_cell = {'x': cx, 'y': cy, 'width': cw, 'height': ch}

            for idx, piece in enumerate(red_crosses):
                if idx not in used_red and _inside(piece, virtual_cell):
                    matrix[ri][ci] = 1
                    used_red.add(idx)
                    break
            else:
                for idx, piece in enumerate(blue_circles):
                    if idx not in used_blue and _inside(piece, virtual_cell):
                        matrix[ri][ci] = 2
                        used_blue.add(idx)
                        break

    return matrix, used_red, used_blue


def _calcular_laterales_y_fuera(red_crosses: list, blue_circles: list,
                                  used_red: set, used_blue: set,
                                  gx1: float, gy1: float, gx2: float, gy2: float,
                                  predictions: list):
    """
    Las piezas de reserva están a IZQUIERDA y DERECHA del tablero en cámara.
    Los 5 slots de cada lateral se distinguen por posición vertical (arriba=0, abajo=4).

    - Separación de laterales: por posición horizontal (px vs gcx)
        px <  gcx  →  left[]   (lado imagen-izquierda → left_fichas  → array_piezas_enemigas)
        px >= gcx  →  right[]  (lado imagen-derecha   → right_fichas → array_piezas)
    - Slots 0-4: por posición vertical (py), arriba → abajo en imagen.
    - Rango Y para slots: board completo si está disponible, si no el grid
      extendido un 40% arriba y abajo para cubrir las piezas más allá del grid.
    """
    gcx = (gx1 + gx2) / 2

    left  = [0] * 5
    right = [0] * 5

    outside = (
        [(p, 1) for i, p in enumerate(red_crosses)  if i not in used_red] +
        [(p, 2) for i, p in enumerate(blue_circles) if i not in used_blue]
    )

    boards = [p for p in predictions if p['class'] == 'board']
    has_board = False
    ebx1 = ebx2 = eby1 = eby2 = 0.0

    if boards:
        b = max(boards, key=lambda p: p['width'] * p['height'])
        bw2 = b['width']  / 2
        bh2 = b['height'] / 2
        mx  = b['width']  * 0.20
        my  = b['height'] * 0.20
        ebx1 = b['x'] - bw2 - mx
        ebx2 = b['x'] + bw2 + mx
        eby1 = b['y'] - bh2 - my
        eby2 = b['y'] + bh2 + my
        # El board incluye los laterales: su rango Y cubre todos los slots
        slot_y1 = b['y'] - bh2
        slot_y2 = b['y'] + bh2
        has_board = True
    else:
        # Sin detección de board: extender el grid un 40% arriba y abajo
        # para cubrir las piezas que sobresalen del área del grid 3×3
        ext = (gy2 - gy1) * 0.40
        slot_y1 = gy1 - ext
        slot_y2 = gy2 + ext

    slot_h = max(slot_y2 - slot_y1, 1) / 5

    n_red_fuera  = 0
    n_blue_fuera = 0

    for piece, val in outside:
        px, py = piece['x'], piece['y']
        in_board = (not has_board) or (ebx1 <= px <= ebx2 and eby1 <= py <= eby2)
        if in_board:
            slot = min(4, max(0, int((py - slot_y1) / slot_h)))
            if px < gcx:
                if left[slot] == 0:
                    left[slot] = val
            else:
                if right[slot] == 0:
                    right[slot] = val
        else:
            if val == 1:
                n_red_fuera += 1
            else:
                n_blue_fuera += 1

    return left[::-1], right[::-1], n_red_fuera, n_blue_fuera


def analyze_board(predictions: list):
    """
    Devuelve (matrix_3x3, left, right, n_red_fuera, n_blue_fuera) o None.

    Estrategia en cascada:
      1. Celdas reales (clase 'cells') — necesita exactamente 9.
      2. Fallback grid   — usa la detección de clase 'grid' para dividir en celdas virtuales.
      3. Fallback board  — usa el 65 % central de la clase 'board' como cuadrícula virtual.
      Si ninguno está disponible, retorna None.
    """
    cells        = [p for p in predictions if p['class'] == 'cells']
    red_crosses  = [p for p in predictions if p['class'] == 'red cross']
    blue_circles = [p for p in predictions if p['class'] == 'blue circle']

    if len(cells) >= 9:
        # ── Camino principal: 9 celdas detectadas ──────────────────
        by_row = sorted(cells, key=lambda c: c['y'])[:9]
        rows   = [sorted(by_row[i*3:(i+1)*3], key=lambda c: c['x']) for i in range(3)]

        matrix: list  = [[0] * 3 for _ in range(3)]
        used_red:  set = set()
        used_blue: set = set()

        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                for idx, piece in enumerate(red_crosses):
                    if idx not in used_red and _inside(piece, cell):
                        matrix[ri][ci] = 1
                        used_red.add(idx)
                        break
                else:
                    for idx, piece in enumerate(blue_circles):
                        if idx not in used_blue and _inside(piece, cell):
                            matrix[ri][ci] = 2
                            used_blue.add(idx)
                            break

        gx1, gy1, gx2, gy2 = _grid_bbox(cells)

    else:
        # ── Fallback: grid o board como cuadrícula virtual ─────────
        bbox = _bbox_grid_desde_predicciones(predictions)
        if bbox is None:
            return None

        gx1, gy1, gx2, gy2 = bbox
        matrix, used_red, used_blue = _asignar_celdas_virtuales(
            gx1, gy1, gx2, gy2, red_crosses, blue_circles
        )
        modo = 'grid' if any(p['class'] == 'grid' for p in predictions) else 'board_central'
        print(f'[analyze_board] fallback activado — modo={modo} '
              f'(cells detectadas: {len(cells)})')

    left, right, n_red_fuera, n_blue_fuera = _calcular_laterales_y_fuera(
        red_crosses, blue_circles, used_red, used_blue,
        gx1, gy1, gx2, gy2, predictions
    )

    return matrix, left, right, n_red_fuera, n_blue_fuera


def hand_over_board(mp_result, predictions: list, frame_w: int, frame_h: int) -> bool:
    if not mp_result or not mp_result.hand_landmarks:
        return False
    cells  = [p for p in predictions if p['class'] == 'cells']
    boards = [p for p in predictions if p['class'] == 'board']
    if len(cells) >= 9:
        ax1, ay1, ax2, ay2 = _grid_bbox(cells)
    elif boards:
        b = max(boards, key=lambda p: p['width'] * p['height'])
        ax1 = b['x'] - b['width']  / 2
        ay1 = b['y'] - b['height'] / 2
        ax2 = b['x'] + b['width']  / 2
        ay2 = b['y'] + b['height'] / 2
    else:
        return False
    for hand_landmarks in mp_result.hand_landmarks:
        for lm in hand_landmarks:
            if ax1 <= lm.x * frame_w <= ax2 and ay1 <= lm.y * frame_h <= ay2:
                return True
    return False


def print_board(matrix, left, right, n_red_fuera, n_blue_fuera) -> None:
    sym = {1: ' 1 ', 2: ' 2 ', 0: ' 0 '}
    print('left:  ' + str(left))
    print('right: ' + str(right))
    print('┌───┬───┬───┐')
    for i, row in enumerate(matrix):
        print('│' + '│'.join(sym[c] for c in row) + '│')
        print('├───┼───┼───┤' if i < 2 else '└───┴───┴───┘')
    if n_red_fuera or n_blue_fuera:
        print(f'Fuera del tablero → red: {n_red_fuera}  blue: {n_blue_fuera}')
    print()


# ──────────────────────────────────────────────
# Conversión al formato tablero={} para Flask
# ──────────────────────────────────────────────

def _resultado_a_json(matrix, left, right, n_red_fuera, n_blue_fuera) -> dict:
    """
    Convierte el resultado de analyze_board a dict JSON siguiendo el estilo tablero={}.
    Ejemplo:
      tablero = "{0,0,0;1,0,0;0,2,0}"
      left    = "{0,1,0,0,0}"
      right   = "{0,0,2,0,0}"
    """
    tablero_str = '{' + ';'.join(','.join(map(str, row)) for row in matrix) + '}'
    left_str    = '{' + ','.join(map(str, left))  + '}'
    right_str   = '{' + ','.join(map(str, right)) + '}'
    return {
        'tablero':    tablero_str,
        'left':       left_str,
        'right':      right_str,
        'fuera_rojo': n_red_fuera,
        'fuera_azul': n_blue_fuera,
    }


# ──────────────────────────────────────────────
# Lógica IA  (minimax α-β + posición + libro de aperturas)
# ──────────────────────────────────────────────

# Bonus posicional: centro domina 4 líneas, esquinas 3, bordes 2
_POS_BONUS = np.array([
    [0.3, 0.1, 0.3],
    [0.1, 0.5, 0.1],
    [0.3, 0.1, 0.3],
], dtype=float)


def possible_moves_numeric(board: np.ndarray) -> list:
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def evaluate_numeric(board: np.ndarray) -> float:
    lineas = (
        list(board)
        + list(board.T)
        + [np.diag(board), np.diag(np.fliplr(board))]
    )
    score = 0.0
    for linea in lineas:
        if np.all(linea == 2): return 100.0
        if np.all(linea == 1): return -100.0
        n2 = np.count_nonzero(linea == 2)
        n1 = np.count_nonzero(linea == 1)
        n0 = np.count_nonzero(linea == 0)
        if n2 == 2 and n0 == 1: score += 10.0
        if n1 == 2 and n0 == 1: score -= 10.0  # simétrico: ataque ≡ defensa
    for r in range(3):
        for c in range(3):
            if   board[r, c] == 2: score += _POS_BONUS[r, c]
            elif board[r, c] == 1: score -= _POS_BONUS[r, c]
    return score


def minimax_numeric(board: np.ndarray, depth: int, maximizing: bool,
                    alpha: float = -np.inf, beta: float = np.inf) -> float:
    score = evaluate_numeric(board)
    if abs(score) >= 100 or depth == 0:
        return score
    moves = possible_moves_numeric(board)
    if not moves:
        return score
    if maximizing:
        best = -np.inf
        for i, j in moves:
            b = board.copy(); b[i, j] = 2
            best  = max(best, minimax_numeric(b, depth - 1, False, alpha, beta))
            alpha = max(alpha, best)
            if beta <= alpha: break
    else:
        best = np.inf
        for i, j in moves:
            b = board.copy(); b[i, j] = 1
            best = min(best, minimax_numeric(b, depth - 1, True, alpha, beta))
            beta = min(beta, best)
            if beta <= alpha: break
    return best


def evaluate_moves_numeric(board: np.ndarray, depth: int):
    moves = possible_moves_numeric(board)
    scores = []
    for i, j in moves:
        b = board.copy(); b[i, j] = 2
        scores.append(minimax_numeric(b, depth, maximizing=False))
    return moves, np.array(scores, dtype=float)


def softmax(scores: np.ndarray, temperature: float = T_SOFTMAX) -> np.ndarray:
    if len(scores) == 0:
        return np.array([])
    t = max(float(temperature), 1e-6)
    exp_scores = np.exp((scores - np.max(scores)) / t)
    return exp_scores / np.sum(exp_scores)


# ── Capa 1: reflejos (ganar / bloquear) ────────────────────────

def _buscar_ganadora(board: np.ndarray, jugador: int):
    """Devuelve la celda que completa una línea ganadora para `jugador`, o None."""
    for r, c in possible_moves_numeric(board):
        b = board.copy(); b[r, c] = jugador
        s = evaluate_numeric(b)
        if (jugador == 2 and s >= 100) or (jugador == 1 and s <= -100):
            return (r, c)
    return None


# ── Capa 2: libro de aperturas ─────────────────────────────────

_ESQUINAS    = [(0, 0), (0, 2), (2, 0), (2, 2)]
_BORDES      = [(0, 1), (1, 0), (1, 2), (2, 1)]
_OPUESTA     = {(0,0):(2,2), (2,2):(0,0), (0,2):(2,0), (2,0):(0,2)}
_ADJ_BORDE   = {
    (0,1): [(0,0),(0,2)], (1,0): [(0,0),(2,0)],
    (1,2): [(0,2),(2,2)], (2,1): [(2,0),(2,2)],
}
# Pares de esquinas adyacentes (no opuestas) para horquilla lateral
_PARES_ESQ   = [[(0,0),(0,2)], [(0,0),(2,0)], [(2,2),(0,2)], [(2,2),(2,0)]]


def _libre(board: np.ndarray, c) -> bool:  return board[c[0], c[1]] == 0
def _es_mia(board: np.ndarray, c) -> bool: return board[c[0], c[1]] == 2
def _rnd(lst: list):                        return lst[int(np.random.randint(len(lst)))]


def _ap_esquina_fork(board: np.ndarray):
    """
    Esquina → esquina opuesta.
    Si el rival no juega en el centro crea una horquilla imbatible.
    """
    mi_esq = next((c for c in _ESQUINAS if _es_mia(board, c)), None)
    if mi_esq is None:
        libres = [c for c in _ESQUINAS if _libre(board, c)]
        return _rnd(libres) if libres else None
    op = _OPUESTA.get(mi_esq)
    return op if op and _libre(board, op) else None


def _ap_centro_control(board: np.ndarray):
    """
    Centro primero — controla las 4 diagonales/líneas centrales.
    Segundo movimiento en esquina para montar la horquilla clásica.
    """
    if _libre(board, (1, 1)):
        return (1, 1)
    libres = [c for c in _ESQUINAS if _libre(board, c)]
    return _rnd(libres) if libres else None


def _ap_trampa_borde(board: np.ndarray):
    """
    Apertura por borde + esquina adyacente → forma L.
    Confunde porque la amenaza no es obvia hasta el tercer movimiento.
    """
    mi_borde = next((c for c in _BORDES if _es_mia(board, c)), None)
    if mi_borde is None:
        libres = [c for c in _BORDES if _libre(board, c)]
        return _rnd(libres) if libres else None
    candidatos = [c for c in _ADJ_BORDE.get(mi_borde, []) if _libre(board, c)]
    return _rnd(candidatos) if candidatos else None


def _ap_doble_esquina(board: np.ndarray):
    """
    Dos esquinas adyacentes (no opuestas) → horquilla lateral
    que fuerza al rival a bloquear pero deja otra amenaza abierta.
    """
    for par in _PARES_ESQ:
        a, b = par
        if _es_mia(board, a) and _libre(board, b): return b
        if _es_mia(board, b) and _libre(board, a): return a
    libres = [c for c in _ESQUINAS if _libre(board, c)]
    return _rnd(libres) if libres else None


_APERTURAS_FN = {
    'esquina_fork':   _ap_esquina_fork,
    'centro_control': _ap_centro_control,
    'trampa_borde':   _ap_trampa_borde,
    'doble_esquina':  _ap_doble_esquina,
}

# (nombre, peso de selección al inicio de partida)
ESTRATEGIAS = [
    ('minimax',         0.30),  # minimax puro, sin libro
    ('esquina_fork',    0.25),  # la más conocida; efectiva y reconocible
    ('centro_control',  0.20),  # clásica humana
    ('doble_esquina',   0.15),  # menos obvia, confunde más
    ('trampa_borde',    0.10),  # la más extraña; rareza táctica
]

_estrategia_actual: str = 'minimax'


def _seleccionar_estrategia() -> str:
    nombres = [e[0] for e in ESTRATEGIAS]
    pesos   = np.array([e[1] for e in ESTRATEGIAS], dtype=float)
    pesos  /= pesos.sum()
    return nombres[int(np.random.choice(len(nombres), p=pesos))]


def choose_move_softmax(board: np.ndarray,
                        temperature: float = T_SOFTMAX,
                        base_depth: int = SEARCH_DEPTH):
    global _estrategia_actual

    empty = int(np.count_nonzero(board == 0))
    if not possible_moves_numeric(board):
        return None, None, 0, temperature, []

    # Seleccionar estrategia al inicio de cada partida (tablero vacío)
    if empty == 9:
        _estrategia_actual = _seleccionar_estrategia()
        print(f'[IA] Estrategia elegida: {_estrategia_actual}')

    # ── CAPA 1: Reflejos — ganar o bloquear inmediatamente ─────
    for jugador, ret_score in [(2, 100.0), (1, 50.0)]:
        mov = _buscar_ganadora(board, jugador)
        if mov:
            return mov, ret_score, 1, 0.0, [1.0]

    # ── CAPA 2: Libro de aperturas (early game, ≥ 6 celdas libres) ─
    if empty >= 6 and _estrategia_actual in _APERTURAS_FN:
        apertura = _APERTURAS_FN[_estrategia_actual](board)
        if apertura is not None and _libre(board, apertura):
            print(f'[IA] {_estrategia_actual} → {apertura}')
            return apertura, 30.0, 0, temperature, [1.0]

    # ── CAPA 3: Minimax con temperatura adaptativa ──────────────
    # Mayor profundidad al final (posiciones críticas), más variedad al inicio
    if empty >= 7:
        depth, temp = max(3, base_depth),       temperature + 0.25
    elif empty >= 4:
        depth, temp = max(5, base_depth + 1),   temperature
    else:
        depth, temp = 9,                         max(0.05, temperature - 0.3)

    moves, scores = evaluate_moves_numeric(board, depth)
    if not moves:
        return None, None, depth, temp, []

    noisy = scores + np.random.normal(0, 0.2, size=len(scores))
    probs  = softmax(noisy, temp)
    idx    = int(np.random.choice(len(moves), p=probs))
    return moves[idx], float(noisy[idx]), depth, temp, probs.tolist()


# ──────────────────────────────────────────────
# Servidor Flask
# ──────────────────────────────────────────────

flask_app = Flask(__name__)


def _decodificar_imagen_request():
    data = (request.files['image'].read()
            if 'image' in request.files
            else request.data)
    if not data:
        return None, 'No se recibieron bytes de imagen'
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, 'No se pudo decodificar la imagen'
    return img, None


@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'modelo': ROBOFLOW_MODEL_ID}), 200


@flask_app.route('/procesar', methods=['POST'])
def procesar():
    """
    Recibe una imagen, infiere con Roboflow/YOLO y devuelve el estado del tablero en JSON.

    Respuesta exitosa:
        {
          "tablero":         "{0,0,0;1,0,0;0,2,0}",
          "left":            "{0,1,0,0,0}",
          "right":           "{0,0,2,0,0}",
          "fuera_rojo":      1,
          "fuera_azul":      0,
          "mano_en_tablero": false
        }
    """
    try:
        img, error = _decodificar_imagen_request()
        if error:
            return jsonify({'error': error}), 400

        print('Recibida imagen desde ESP32. Infiriendo...')
        preds  = infer(img)
        result = analyze_board(preds)

        if result is None:
            return jsonify({'error': 'No se detectaron suficientes referencias del tablero (cells/grid/board)'}), 422

        matrix, left, right, n_red_fuera, n_blue_fuera = result
        data = _resultado_a_json(matrix, left, right, n_red_fuera, n_blue_fuera)

        mano_en_tablero = False
        with _sync_landmarker_lock:
            _init_sync_landmarker()
            if _sync_landmarker is not None:
                rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = _sync_landmarker.recognize(mp_image)
                fh, fw   = img.shape[:2]
                mano_en_tablero = hand_over_board(mp_result, preds, fw, fh)

        data['mano_en_tablero'] = mano_en_tablero
        print(f'RESULTADO: tablero={data["tablero"]}  left={data["left"]}  right={data["right"]}  mano={mano_en_tablero}')
        return jsonify(data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@flask_app.route('/movimiento', methods=['POST'])
def movimiento():
    """
    Recibe {"matriz": [[...],[...],[...]]} y devuelve la siguiente jugada de la IA.
    """
    try:
        payload = request.get_json(silent=True)
        if not payload or 'matriz' not in payload:
            return jsonify({'error': 'Debes enviar JSON con la clave "matriz"'}), 400

        matrix = np.array(payload['matriz'])
        if matrix.shape != (3, 3):
            return jsonify({'error': 'La matriz debe ser de tamaño 3x3'}), 400

        move, score, depth, temperature, probabilities = choose_move_softmax(matrix.astype(int))

        if move is None:
            return jsonify({'movimiento': None, 'mensaje': 'Fin'}), 200

        return jsonify({
            'movimiento':    {'fila': int(move[0]), 'columna': int(move[1])},
            'score':         score,
            'depth':         depth,
            'temperature':   temperature,
            'probabilities': probabilities,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# Verificación de estado inicial del tablero
# ──────────────────────────────────────────────

_sync_landmarker      = None
_sync_landmarker_lock = threading.Lock()


def _init_sync_landmarker() -> None:
    """Inicializa (lazy) el GestureRecognizer en modo IMAGE para uso síncrono en Flask."""
    global _sync_landmarker
    if _sync_landmarker is not None or not TIENE_MEDIAPIPE:
        return
    ensure_gesture_model()
    opts = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DETECT,
        min_hand_presence_confidence=0.5,
    )
    _sync_landmarker = mp_vision.GestureRecognizer.create_from_options(opts)
    print('[MediaPipe] GestureRecognizer IMAGE mode inicializado.')


@flask_app.route('/verificar_tablero', methods=['POST'])
def verificar_tablero():
    """
    Recibe una imagen, infiere el estado del tablero y detecta manos.
    Devuelve si el tablero está listo para empezar una partida.

    Respuesta:
        {
          "listo":           true/false,
          "tablero_limpio":  true/false,
          "mano_en_tablero": true/false
        }
    """
    try:
        img, error = _decodificar_imagen_request()
        if error:
            return jsonify({'error': error}), 400

        preds  = infer(img)
        result = analyze_board(preds)

        tablero_limpio = False
        if result is not None:
            matrix, _, _, n_red_fuera, n_blue_fuera = result
            celdas_vacias  = all(matrix[r][c] == 0 for r in range(3) for c in range(3))
            piezas_en_zona = (n_red_fuera == 0 and n_blue_fuera == 0)
            tablero_limpio = celdas_vacias and piezas_en_zona

        mano_en_tablero = False
        with _sync_landmarker_lock:
            _init_sync_landmarker()
            if _sync_landmarker is not None:
                rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = _sync_landmarker.recognize(mp_image)
                fh, fw   = img.shape[:2]
                mano_en_tablero = hand_over_board(mp_result, preds, fw, fh)

        listo = tablero_limpio and not mano_en_tablero
        print(f'[VERIFICAR] listo={listo}  tablero_limpio={tablero_limpio}  mano={mano_en_tablero}')
        return jsonify({
            'listo':           listo,
            'tablero_limpio':  tablero_limpio,
            'mano_en_tablero': mano_en_tablero,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _lanzar_flask_en_hilo(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Arranca Flask en un hilo daemon para que conviva con el bucle live."""
    print('=' * 45)
    print(f'  Flask (ESP32-CAM) → http://{host}:{port}')
    print(f'    /procesar          POST  → análisis de imagen')
    print(f'    /movimiento        POST  → siguiente jugada IA')
    print(f'    /verificar_tablero POST  → estado inicial del tablero')
    print(f'    /health            GET   → estado del servidor')
    print('=' * 45)
    flask_app.run(host=host, port=port, use_reloader=False)


# ──────────────────────────────────────────────
# Estado compartido entre hilos (Roboflow + MediaPipe)
# ──────────────────────────────────────────────

_latest_preds: list = []
_preds_version: int = 0
_preds_lock         = threading.Lock()

_mp_latest_result   = None
_mp_lock            = threading.Lock()


def _mp_callback(result, _output_image, _timestamp_ms) -> None:
    global _mp_latest_result
    with _mp_lock:
        _mp_latest_result = result


def _inference_worker(frame_ref: list, stop_event: threading.Event) -> None:
    """Corre inferencia de Roboflow continuamente sobre el frame más reciente."""
    while not stop_event.is_set():
        with _preds_lock:
            frame = frame_ref[0] if frame_ref else None
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            preds = infer(frame)
            with _preds_lock:
                global _latest_preds, _preds_version
                _latest_preds   = preds
                _preds_version += 1
        except Exception as e:
            print(f'[Roboflow] {e}')


# ──────────────────────────────────────────────
# Descarga del modelo MediaPipe
# ──────────────────────────────────────────────

def ensure_gesture_model() -> None:
    if not Path(GESTURE_MODEL_PATH).exists():
        print('Descargando gesture_recognizer.task (~25 MB)…')
        urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
        print('Descarga completa.')


# ──────────────────────────────────────────────
# Bucle principal (tiempo real)
# ──────────────────────────────────────────────

def main_live(flask_host: str = DEFAULT_HOST,
              flask_port: int = DEFAULT_PORT,
              con_flask: bool = True) -> None:
    """
    Abre la ventana de previsualización en tiempo real.
    Si con_flask=True arranca el servidor Flask en un hilo daemon paralelo.
    """
    if con_flask:
        threading.Thread(
            target=_lanzar_flask_en_hilo,
            args=(flask_host, flask_port),
            daemon=True,
            name='flask-server',
        ).start()

    if TIENE_MEDIAPIPE:
        ensure_gesture_model()
        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            result_callback=_mp_callback,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MP_DETECT,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=MP_TRACK,
        )
        landmarker = mp_vision.GestureRecognizer.create_from_options(options)
    else:
        landmarker = None

    source = ESP32_STREAM_URL if USE_ESP32 else CAMERA_ID
    print(f'Fuente de video: {source}')
    print(f'Modelo Roboflow: {ROBOFLOW_MODEL_ID}\n')

    cap = cv2.VideoCapture(source)
    if not USE_ESP32:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f'ERROR: no se pudo abrir la fuente de video: {source}')
        if landmarker:
            landmarker.close()
        return

    frame_ref  = [None]
    stop_event = threading.Event()
    threading.Thread(
        target=_inference_worker, args=(frame_ref, stop_event), daemon=True
    ).start()

    show_det      = True
    show_mp       = True
    show_conf     = True
    seen_version  = -1
    last_matrix   = None
    hand_covering = False
    prev_gestos: list = []
    start_t       = time.perf_counter()
    prev_t        = start_t

    print("Corriendo — 'q' salir | 'm'/'y' togglear capas | 'c' labels de confianza")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        current_preds = []
        if show_det:
            with _preds_lock:
                frame_ref[0] = frame.copy()
                cur_ver       = _preds_version
                current_preds = list(_latest_preds)

            draw_detections(frame, current_preds, show_conf)

            if cur_ver != seen_version:
                seen_version = cur_ver
                result = analyze_board(current_preds)
                if result is not None:
                    matrix, left, right, n_red_fuera, n_blue_fuera = result
                    if result != last_matrix:
                        last_matrix = result
                        print_board(matrix, left, right, n_red_fuera, n_blue_fuera)

        current_mp = None
        if show_mp and landmarker:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int((time.perf_counter() - start_t) * 1000)
            landmarker.recognize_async(mp_image, ts_ms)
            with _mp_lock:
                current_mp = _mp_latest_result
            if current_mp is not None:
                draw_hand_landmarks(frame, current_mp)

        gestos = get_gestos(current_mp)
        if gestos != prev_gestos:
            prev_gestos = gestos
            for gesto, side, score in gestos:
                print(f'[GESTO] {side}: {gesto} ({score:.2f})')

        fh, fw = frame.shape[:2]
        now_covering = hand_over_board(current_mp, current_preds, fw, fh)
        if now_covering != hand_covering:
            hand_covering = now_covering
            if hand_covering:
                print('[AVISO] Mano detectada sobre el tablero')
            else:
                print('[INFO]  Mano retirada del tablero')

        now    = time.perf_counter()
        fps    = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        draw_fps(frame, fps)
        draw_status(frame, show_det, show_mp)

        cv2.imshow('Roboflow + MediaPipe Hands', frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key == ord('m'):
            show_mp = not show_mp
            print(f'MediaPipe: {"ON" if show_mp else "OFF"}')
        elif key == ord('y'):
            show_det = not show_det
            print(f'Detección: {"ON" if show_det else "OFF"}')
        elif key == ord('c'):
            show_conf = not show_conf

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    if landmarker:
        landmarker.close()


def main_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Arranca solo el servidor Flask (sin ventana)."""
    print('=' * 45)
    print(f'  Flask (ESP32-CAM) → http://{host}:{port}')
    print(f'    /procesar          POST  → análisis de imagen')
    print(f'    /movimiento        POST  → siguiente jugada IA')
    print(f'    /verificar_tablero POST  → estado inicial del tablero')
    print(f'    /health            GET   → estado del servidor')
    print('=' * 45)
    flask_app.run(host=host, port=port)


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tic-Tac-Toe: detección en tiempo real + servidor Flask para ESP32-CAM.'
    )
    parser.add_argument(
        '--modo',
        choices=['live', 'server'],
        default='live',
        help=(
            'live   → ventana tiempo real + Flask en background (default);  '
            'server → solo servidor Flask'
        ),
    )
    parser.add_argument('--host', default=DEFAULT_HOST,
                        help=f'Host del servidor Flask (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Puerto del servidor Flask (default: {DEFAULT_PORT})')
    args = parser.parse_args()

    if args.modo == 'server':
        main_server(host=args.host, port=args.port)
    else:
        main_live(flask_host=args.host, flask_port=args.port, con_flask=True)
