"""
Real-time detector: modelo de Roboflow (inference SDK) + MediaPipe hand landmarks.
Compatible con MediaPipe >= 0.10 (Tasks API).

El modelo se descarga automáticamente desde Roboflow en el primer arranque.
El modelo hand_landmarker.task también se descarga si no existe.

Controls:
  q  — quit
  m  — toggle MediaPipe landmarks on/off
  y  — toggle YOLO / Roboflow detections on/off
  c  — toggle confidence labels
"""

import time
import threading
import urllib.request
from pathlib import Path

import base64

import cv2
import requests
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY = "C6jJ9kSIF8VRiJWR3rFE" 
ROBOFLOW_MODEL_ID = "no-se-7qatn/7"

LANDMARKER_PATH = "hand_landmarker.task"
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

DETECT_CONF   = 0.50
MP_DETECT     = 0.70
MP_TRACK      = 0.55
MAX_HANDS     = 2
INFER_WIDTH   = 640   # tamaño al que se reduce el frame antes de mandarlo a Roboflow
USE_ESP32        = True                        # ← True = ESP32-CAM | False = webcam
ESP32_STREAM_URL = "http://10.191.81.88:81/stream"  # stream MJPEG del ESP32
CAMERA_ID        = 0                           # índice de webcam (0, 1, 2…)
CAM_W, CAM_H       = 1280, 720                 # solo aplica a webcam

# BGR colours — uno por clase YOLO
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "red cross":   (40,  40,  220),
    "blue circle": (220, 120,  30),
    "board":       (0,   210, 210),
    "grid":        (0,   160, 255),
    "cells":       (200, 140, 255),
}
CLASS_TEXT_COLOR: dict[str, tuple[int, int, int]] = {
    "red cross":   (255, 255, 255),
    "blue circle": (255, 255, 255),
    "board":       (0,   0,   0),
    "grid":        (0,   0,   0),
    "cells":       (0,   0,   0),
}
C_FALLBACK_BOX  = (200, 200, 200)
C_FALLBACK_TEXT = (0,   0,   0)
C_FPS       = (0,   255, 255)
C_STATUS    = (180, 180, 180)
C_HAND_SIDE = (255, 255,  0)

# 21 conexiones del esqueleto de la mano
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
# Drawing helpers
# ──────────────────────────────────────────────

def draw_fps(frame: cv2.Mat, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_FPS, 2, cv2.LINE_AA)


def draw_status(frame: cv2.Mat, show_det: bool, show_mp: bool) -> None:
    h = frame.shape[0]
    text = (f"[y] Detección: {'ON ' if show_det else 'OFF'}   "
            f"[m] MediaPipe: {'ON ' if show_mp else 'OFF'}   "
            f"[q] quit")
    cv2.putText(frame, text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_STATUS, 1, cv2.LINE_AA)


def roboflow_infer(frame: cv2.Mat) -> list:
    """Redimensiona el frame y llama a la REST API de Roboflow."""
    h, w = frame.shape[:2]
    if w > INFER_WIDTH:
        scale = INFER_WIDTH / w
        small = cv2.resize(frame, (INFER_WIDTH, int(h * scale)))
    else:
        small = frame

    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64 = base64.b64encode(buf).decode("utf-8")
    resp = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}",
        params={"api_key": ROBOFLOW_API_KEY, "confidence": DETECT_CONF},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=5,
    )
    resp.raise_for_status()
    preds = resp.json().get("predictions", [])

    # Escalar coordenadas de vuelta al tamaño original del frame
    if w > INFER_WIDTH:
        inv = 1.0 / scale
        for p in preds:
            p["x"]      *= inv
            p["y"]      *= inv
            p["width"]  *= inv
            p["height"] *= inv
    return preds


# Estado compartido: Roboflow
_latest_preds: list = []
_preds_version: int  = 0
_preds_lock            = threading.Lock()

# Estado compartido: MediaPipe (LIVE_STREAM callback)
_mp_latest_result = None
_mp_lock          = threading.Lock()

def _mp_callback(result, _output_image, _timestamp_ms) -> None:
    global _mp_latest_result
    with _mp_lock:
        _mp_latest_result = result


def _inference_worker(frame_ref: list, stop_event: threading.Event) -> None:
    """Hilo que corre inferencia continuamente sobre el frame más reciente."""
    while not stop_event.is_set():
        with _preds_lock:
            frame = frame_ref[0] if frame_ref else None
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            preds = roboflow_infer(frame)
            with _preds_lock:
                global _latest_preds, _preds_version
                _latest_preds  = preds
                _preds_version += 1
        except Exception as e:
            print(f"[Roboflow] {e}")


def draw_detections(frame: cv2.Mat, predictions: list, show_conf: bool = True) -> None:
    """Dibuja las predicciones de la REST API de Roboflow.

    Cada prediction es un dict: x, y (centro), width, height, confidence, class
    """
    for pred in predictions:
        cls_name = pred["class"]
        conf     = pred["confidence"]

        if conf < DETECT_CONF:
            continue

        # Roboflow usa formato centro; convertir a esquinas
        x1 = int(pred["x"] - pred["width"]  / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"]  / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        box_color  = CLASS_COLORS.get(cls_name,     C_FALLBACK_BOX)
        text_color = CLASS_TEXT_COLOR.get(cls_name, C_FALLBACK_TEXT)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        label = f"{cls_name} {conf:.2f}" if show_conf else cls_name
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        pad = 4
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - pad * 2),
                      (x1 + tw + pad * 2, y1),
                      box_color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + pad, y1 - baseline - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)


def draw_hand_landmarks(frame: cv2.Mat, mp_result) -> None:
    if not mp_result.hand_landmarks:
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

        if mp_result.handedness:
            side  = mp_result.handedness[hand_idx][0].category_name
            score = mp_result.handedness[hand_idx][0].score
            wrist = pts[0]
            cv2.putText(frame, f"{side} {score:.2f}",
                        (wrist[0] - 35, wrist[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_HAND_SIDE, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Board analysis
# ──────────────────────────────────────────────

def _inside(piece: dict, cell: dict) -> bool:
    """True si el centro de piece cae dentro del bbox de cell."""
    x1 = cell["x"] - cell["width"]  / 2
    y1 = cell["y"] - cell["height"] / 2
    x2 = cell["x"] + cell["width"]  / 2
    y2 = cell["y"] + cell["height"] / 2
    return x1 <= piece["x"] <= x2 and y1 <= piece["y"] <= y2


def _grid_bbox(cells: list):
    xs1 = [c["x"] - c["width"]/2  for c in cells]
    ys1 = [c["y"] - c["height"]/2 for c in cells]
    xs2 = [c["x"] + c["width"]/2  for c in cells]
    ys2 = [c["y"] + c["height"]/2 for c in cells]
    return min(xs1), min(ys1), max(xs2), max(ys2)


def analyze_board(predictions: list) -> tuple | None:
    """
    Devuelve (matrix_3x3, left, right, n_red_fuera, n_blue_fuera) o None si no hay 9 celdas.
      matrix_3x3   : list[list[int]]  — 0=vacío  1=red cross  2=blue circle
      left / right : list[int] de 5 ranuras — 0/1/2, ordenadas de arriba (0) a abajo (4)
      n_red_fuera  : int — red crosses visibles pero fuera de celdas y laterales
      n_blue_fuera : int — blue circles visibles pero fuera de celdas y laterales
    Las ranuras de left/right se calculan por posición Y relativa al bbox del grid,
    sin depender del bbox de board, así las piezas a medias no se pierden.
    """
    cells        = [p for p in predictions if p["class"] == "cells"]
    red_crosses  = [p for p in predictions if p["class"] == "red cross"]
    blue_circles = [p for p in predictions if p["class"] == "blue circle"]

    if len(cells) < 9:
        return None

    # ── Matriz 3×3 ─────────────────────────────────────────────
    by_row = sorted(cells, key=lambda c: c["y"])[:9]
    rows   = [sorted(by_row[i*3:(i+1)*3], key=lambda c: c["x"]) for i in range(3)]

    matrix: list[list[int]] = [[0] * 3 for _ in range(3)]
    used_red: set[int]  = set()
    used_blue: set[int] = set()

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

    # ── Laterales izquierdo/derecho ─────────────────────────────
    # Referencia de altura = grid (no depende del bbox de board)
    gx1, gy1, gx2, gy2 = _grid_bbox(cells)
    gcx    = (gx1 + gx2) / 2
    slot_h = max(gy2 - gy1, 1) / 5

    left  = [0] * 5
    right = [0] * 5

    outside = (
        [(p, 1) for i, p in enumerate(red_crosses)  if i not in used_red] +
        [(p, 2) for i, p in enumerate(blue_circles) if i not in used_blue]
    )

    # Bbox del board expandido un 20 % como margen de tolerancia
    boards = [p for p in predictions if p["class"] == "board"]
    if boards:
        b = max(boards, key=lambda p: p["width"] * p["height"])
        margin_x = b["width"]  * 0.20
        margin_y = b["height"] * 0.20
        ebx1 = b["x"] - b["width"]  / 2 - margin_x
        ebx2 = b["x"] + b["width"]  / 2 + margin_x
        eby1 = b["y"] - b["height"] / 2 - margin_y
        eby2 = b["y"] + b["height"] / 2 + margin_y
        has_board = True
    else:
        has_board = False

    n_red_fuera  = 0
    n_blue_fuera = 0

    for piece, val in outside:
        px, py = piece["x"], piece["y"]
        in_board = (not has_board) or (ebx1 <= px <= ebx2 and eby1 <= py <= eby2)
        if in_board:
            slot = min(4, max(0, int((py - gy1) / slot_h)))
            if px < gcx:
                left[slot] = val
            else:
                right[slot] = val
        else:
            if val == 1:
                n_red_fuera += 1
            else:
                n_blue_fuera += 1

    return matrix, left, right, n_red_fuera, n_blue_fuera


def hand_over_board(mp_result, predictions: list, frame_w: int, frame_h: int) -> bool:
    """True si algún landmark de la mano cae dentro del área del grid (o del board)."""
    if not mp_result or not mp_result.hand_landmarks:
        return False
    cells  = [p for p in predictions if p["class"] == "cells"]
    boards = [p for p in predictions if p["class"] == "board"]
    if len(cells) >= 9:
        ax1, ay1, ax2, ay2 = _grid_bbox(cells)
    elif boards:
        b = max(boards, key=lambda p: p["width"] * p["height"])
        ax1 = b["x"] - b["width"]  / 2
        ay1 = b["y"] - b["height"] / 2
        ax2 = b["x"] + b["width"]  / 2
        ay2 = b["y"] + b["height"] / 2
    else:
        return False
    for hand_landmarks in mp_result.hand_landmarks:
        for lm in hand_landmarks:
            if ax1 <= lm.x * frame_w <= ax2 and ay1 <= lm.y * frame_h <= ay2:
                return True
    return False


def print_board(matrix: list[list[int]],
                left: list[int], right: list[int],
                n_red_fuera: int, n_blue_fuera: int) -> None:
    sym = {1: " 1 ", 2: " 2 ", 0: " 0 "}
    sep_top = "┌───┬───┬───┐"
    sep_mid = "├───┼───┼───┤"
    sep_bot = "└───┴───┴───┘"
    print("left:  " + str(left))
    print("right: " + str(right))
    print(sep_top)
    for i, row in enumerate(matrix):
        print("│" + "│".join(sym[c] for c in row) + "│")
        print(sep_mid if i < 2 else sep_bot)
    if n_red_fuera or n_blue_fuera:
        print("Fuera del tablero → red: {}  blue: {}".format(n_red_fuera, n_blue_fuera))
    print()


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def ensure_landmarker_model() -> None:
    if not Path(LANDMARKER_PATH).exists():
        print("Descargando hand_landmarker.task (~25 MB)…")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("Descarga completa.")


def main() -> None:
    ensure_landmarker_model()

    source = ESP32_STREAM_URL if USE_ESP32 else CAMERA_ID
    print(f"Fuente de video: {source}")
    print(f"Modelo Roboflow: {ROBOFLOW_MODEL_ID}\n")

    # MediaPipe Tasks API — LIVE_STREAM: detect_async no bloquea el loop
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        result_callback=_mp_callback,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DETECT,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=MP_TRACK,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(source)
    if not USE_ESP32:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la fuente de video: {source}")
        landmarker.close()
        return

    # Hilo de inferencia
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
    hand_covering = False          # estado anterior de mano sobre tablero
    start_t       = time.perf_counter()
    prev_t        = start_t

    print("Corriendo — 'q' salir | 'm'/'y' togglear capas | 'c' labels de confianza")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        # ── Roboflow + análisis de tablero ─────
        current_preds = []
        if show_det:
            with _preds_lock:
                frame_ref[0] = frame.copy()
                cur_ver       = _preds_version
                current_preds = list(_latest_preds)

            draw_detections(frame, current_preds, show_conf)

            # Analizar tablero solo cuando hay predicciones nuevas
            if cur_ver != seen_version:
                seen_version = cur_ver
                result = analyze_board(current_preds)
                if result is not None:
                    matrix, left, right, n_red_fuera, n_blue_fuera = result
                    if result != last_matrix:
                        last_matrix = result
                        print_board(matrix, left, right, n_red_fuera, n_blue_fuera)

        # ── MediaPipe Tasks (LIVE_STREAM — no bloquea) ──────────
        current_mp = None
        if show_mp:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int((time.perf_counter() - start_t) * 1000)
            landmarker.detect_async(mp_image, ts_ms)
            with _mp_lock:
                current_mp = _mp_latest_result
            if current_mp is not None:
                draw_hand_landmarks(frame, current_mp)

        # ── Mano sobre tablero ───────────────────
        fh, fw = frame.shape[:2]
        now_covering = hand_over_board(current_mp, current_preds, fw, fh)
        if now_covering != hand_covering:
            hand_covering = now_covering
            if hand_covering:
                print("[AVISO] Mano detectada sobre el tablero")
            else:
                print("[INFO]  Mano retirada del tablero")

        # ── Overlay ────────────────────────────
        now    = time.perf_counter()
        fps    = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        draw_fps(frame, fps)
        draw_status(frame, show_det, show_mp)

        cv2.imshow("Roboflow + MediaPipe Hands", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key == ord('m'):
            show_mp = not show_mp
            print(f"MediaPipe: {'ON' if show_mp else 'OFF'}")
        elif key == ord('y'):
            show_det = not show_det
            print(f"Detección: {'ON' if show_det else 'OFF'}")
        elif key == ord('c'):
            show_conf = not show_conf

    stop_event.set()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
