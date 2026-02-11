"""
Tres en Raya — Visión por Computador
======================================
Detecta el estado del tablero a partir de una foto tomada desde cualquier ángulo.

Estrategia:
  1. Encontrar el panel blanco del tablero con corrección de perspectiva.
  2. Dentro del panel enderezado, localizar el marco negro de la cuadrícula
     usando análisis de contenido (blancura interior + centralidad).
  3. Dividir ese marco en 9 celdas y clasificar cada una por color HSV.
     - Tapa ROJA  → ficha X
     - Tapa AZUL CLARO → ficha O

Uso:
    python tictactoe_vision.py imagen.jpg
    python tictactoe_vision.py imagen.jpg --debug      # guarda imágenes intermedias
    python tictactoe_vision.py imagen.jpg --visualize  # guarda resultado anotado
    python tictactoe_vision.py imagen.jpg --json       # salida JSON

Dependencias:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import json
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# RANGOS DE COLOR HSV
# ──────────────────────────────────────────────────────────────────────────────

# Rojo (fichas X) — el rojo cruza el 0° en HSV, necesita dos rangos
RED_L1 = np.array([0,   70, 55])
RED_U1 = np.array([12, 255, 255])
RED_L2 = np.array([163, 70, 55])
RED_U2 = np.array([180, 255, 255])

# Azul claro / cyan (fichas O)
BLUE_L = np.array([85,  35, 75])
BLUE_U = np.array([118, 255, 255])

# Blanco (interior de las celdas)
WHITE_L = np.array([0,   0,  165])
WHITE_U = np.array([180, 55, 255])

# Threshold mínimo de píxeles de color para detectar una ficha
COLOR_THRESH = 0.055   # fracción del área de la celda (conservador para evitar falsos positivos)


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────

def order_points(pts):
    """Ordena 4 puntos: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    pts = np.array(pts, dtype="float32").reshape(-1, 2)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype="float32")


def red_score(hsv_patch):
    """Fracción de píxeles rojos en un parche HSV."""
    m1 = cv2.inRange(hsv_patch, RED_L1, RED_U1)
    m2 = cv2.inRange(hsv_patch, RED_L2, RED_U2)
    return np.sum(cv2.bitwise_or(m1, m2) > 0) / max(hsv_patch.shape[0] * hsv_patch.shape[1], 1)


def blue_score(hsv_patch):
    """Fracción de píxeles azul-claro en un parche HSV."""
    return np.sum(cv2.inRange(hsv_patch, BLUE_L, BLUE_U) > 0) / max(
        hsv_patch.shape[0] * hsv_patch.shape[1], 1)


def white_fraction(bgr_patch):
    """Fracción de píxeles blancos en un parche BGR."""
    return np.sum(cv2.inRange(cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV),
                              WHITE_L, WHITE_U) > 0) / max(bgr_patch.shape[0] * bgr_patch.shape[1], 1)


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1: ENCONTRAR EL PANEL BLANCO Y APLICAR PERSPECTIVA
# ──────────────────────────────────────────────────────────────────────────────

def warp_panel(img, debug=False):
    """
    Detecta el panel blanco del tablero (fondo blanco con cuadrícula negra)
    y lo endereza con una transformación de perspectiva.

    Retorna (warped, M) o (None, None) si no se encuentra.

    Notas de diseño:
    - Se usa kernel PEQUEÑO (8px) para el cierre morfológico: suficiente para
      rellenar los huecos de las líneas negras de la cuadrícula (~5px de ancho),
      pero sin fusionar el tablero principal con otros elementos blancos
      cercanos (tablero secundario de reserva, reflejos, etc.).
    - Se filtra por distancia al borde: el panel del tablero nunca toca el
      borde de la imagen, a diferencia de fondos o elementos periféricos.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscara de blanco
    white = cv2.inRange(hsv, WHITE_L, WHITE_U)

    # Kernel pequeño: solo rellenar los huecos de las líneas negras (~5-8px)
    # sin fusionar con tableros secundarios u otros elementos blancos cercanos
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    white_closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_close)
    white_clean  = cv2.morphologyEx(white_closed, cv2.MORPH_OPEN, k_open)

    cnts, _ = cv2.findContours(white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el blob que:
    #   1. Es razonablemente cuadrado (ratio > 0.6)
    #   2. No toca el borde de la imagen (border_dist > 20px)
    #   3. Es suficientemente grande (> 4% del área de imagen)
    panel_cnt = None
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < (h * w) * 0.04:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        ratio        = min(bw, bh) / max(bw, bh)
        border_dist  = min(bx, by, w - bx - bw, h - by - bh)
        if ratio > 0.6 and border_dist > 20:
            panel_cnt = cnt
            break

    if panel_cnt is None:
        # Fallback: si ningún contorno cumple los criterios estrictos,
        # relajar el filtro de borde pero mantener el ratio
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < (h * w) * 0.04:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if min(bw, bh) / max(bw, bh) > 0.5:
                panel_cnt = cnt
                break

    if panel_cnt is None:
        return None, None

    # Obtener las 4 esquinas (convex hull → approxPolyDP)
    hull = cv2.convexHull(panel_cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
            break

    if quad is None:
        rect = cv2.minAreaRect(panel_cnt)
        quad = cv2.boxPoints(rect).astype("float32")

    src = order_points(quad)
    side = max(int(np.linalg.norm(src[1] - src[0])),
               int(np.linalg.norm(src[3] - src[0])), 300)
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    M   = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (side, side))

    if debug:
        dbg = img.copy()
        cv2.polylines(dbg, [src.astype(int)], True, (0, 255, 0), 3)
        cv2.imwrite("_debug_panel_corners.jpg", dbg)
        cv2.imwrite("_debug_warped_panel.jpg", warped)

    return warped, M


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2: LOCALIZAR EL MARCO DE LA CUADRÍCULA DENTRO DEL PANEL WARPED
# ──────────────────────────────────────────────────────────────────────────────

def find_grid_bbox(warped, debug=False):
    """
    Dentro del panel warped, localiza el bbox (gx, gy, gw, gh) del marco negro
    de la cuadrícula 3×3.

    Estrategia: threshold 80 (no 70) para el negro del warped + dilatar para
    unir los segmentos de las líneas → el marco de la cuadrícula queda como
    el contorno más grande y cuadrado (ratio ~1.0) con interior blanco.
    """
    side = warped.shape[0]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # threshold 80: suficiente para capturar el negro de las líneas del tablero
    # sin fragmentar el contorno (con 70 la iluminación rompe el contorno en partes)
    _, black = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Dilatar para unir segmentos de las líneas de la cuadrícula
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    black_d = cv2.dilate(black, k_dil)

    if debug:
        cv2.imwrite("_debug_warp_black.jpg", black_d)

    cnts, _ = cv2.findContours(black_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Elegir el contorno más grande que sea cuadrado (ratio > 0.7)
    # y tenga interior predominantemente blanco
    best_box = None
    for cnt in cnts[:8]:
        area = cv2.contourArea(cnt)
        if area < (side ** 2) * 0.05:
            continue
        gx, gy, gw, gh = cv2.boundingRect(cnt)
        ratio = min(gw, gh) / max(gw, gh)
        if ratio < 0.7:
            continue
        interior = warped[gy:gy + gh, gx:gx + gw]
        wf = white_fraction(interior)
        if wf > 0.4:          # al menos 40 % del interior es blanco (las celdas)
            best_box = (gx, gy, gw, gh)
            break             # tomamos el primero válido (mayor área)

    if best_box is None:
        # Fallback: área blanca central del warped
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        wm = cv2.inRange(warped_hsv, WHITE_L, WHITE_U)
        h_idx = np.where(np.sum(wm, axis=1) > side * 0.15)[0]
        v_idx = np.where(np.sum(wm, axis=0) > side * 0.15)[0]
        if len(h_idx) and len(v_idx):
            best_box = (v_idx[0], h_idx[0], v_idx[-1] - v_idx[0], h_idx[-1] - h_idx[0])
        else:
            best_box = (side // 10, side // 10, int(side * 0.8), int(side * 0.8))

    if debug:
        dbg = warped.copy()
        gx, gy, gw, gh = best_box
        cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 3)
        cv2.imwrite("_debug_grid_bbox.jpg", dbg)

    return best_box


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3: CLASIFICAR LAS 9 CELDAS
# ──────────────────────────────────────────────────────────────────────────────

def classify_cells(warped, gx, gy, gw, gh, debug=False):
    """
    Divide el área del tablero en 9 celdas 3×3 y clasifica cada una.

    Retorna:
      board  : lista 3×3 con 'X', 'O' o None
      conf   : lista 3×3 con score de confianza (0–1)
    """
    side = warped.shape[0]
    whsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    board = [[None] * 3 for _ in range(3)]
    conf  = [[0.0]  * 3 for _ in range(3)]

    cell_h = gh // 3
    cell_w = gw // 3
    MARGIN_H = max(int(cell_h * 0.15), 5)
    MARGIN_W = max(int(cell_w * 0.15), 5)

    dbg = warped.copy() if debug else None

    for row in range(3):
        for col in range(3):
            y1 = gy + row * cell_h + MARGIN_H
            y2 = gy + (row + 1) * cell_h - MARGIN_H
            x1 = gx + col * cell_w + MARGIN_W
            x2 = gx + (col + 1) * cell_w - MARGIN_W

            y1, y2 = max(0, y1), min(side, y2)
            x1, x2 = max(0, x1), min(side, x2)

            patch = whsv[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            rs = red_score(patch)
            bs = blue_score(patch)

            piece     = None
            piece_conf = 0.0

            if rs > COLOR_THRESH and rs > bs * 1.3:
                piece      = 'X'
                piece_conf = float(rs)
            elif bs > COLOR_THRESH and bs > rs * 1.3:
                piece      = 'O'
                piece_conf = float(bs)

            board[row][col] = piece
            conf[row][col]  = round(piece_conf, 3)

            if debug:
                color = (0, 0, 200) if piece == 'X' else \
                        (200, 120, 0) if piece == 'O' else (150, 150, 150)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
                label = f"{piece or '?'} {piece_conf:.2f}"
                cv2.putText(dbg, label, (x1 + 3, (y1 + y2) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    if debug:
        cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        cv2.imwrite("_debug_cells.jpg", dbg)

    return board, conf


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4: DETECTAR FICHAS DE RESERVA EN LOS LATERALES
# ──────────────────────────────────────────────────────────────────────────────

def count_reserve(img, board_corners):
    """
    Cuenta las fichas de reserva en las zonas laterales al tablero.
    Zona izquierda → fichas X (rojas)
    Zona derecha   → fichas O (azul claro)

    Retorna {'X': n, 'O': n}
    """
    h, w = img.shape[:2]
    ordered = order_points(board_corners)
    tl, tr, br, bl = ordered

    def count_zone(x1, y1, x2, y2, is_red):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        zone = img[y1:y2, x1:x2]
        if zone.size == 0:
            return 0
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        if is_red:
            m1 = cv2.inRange(hsv, RED_L1, RED_U1)
            m2 = cv2.inRange(hsv, RED_L2, RED_U2)
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, BLUE_L, BLUE_U)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in cnts if cv2.contourArea(c) > 80)

    # Zona izquierda
    lx1 = max(0, min(tl[0], bl[0]) * 0.1)
    lx2 = min(tl[0], bl[0])
    ly1 = min(tl[1], bl[1])
    ly2 = max(tl[1], bl[1])

    # Zona derecha
    rx1 = max(tr[0], br[0])
    rx2 = min(w, rx1 + (w - rx1) * 0.9)
    ry1 = min(tr[1], br[1])
    ry2 = max(tr[1], br[1])

    return {
        'X': count_zone(lx1, ly1, lx2, ly2, True),
        'O': count_zone(rx1, ry1, rx2, ry2, False),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def analyze(image_path, debug=False, visualize=False):
    """
    Análisis completo de una imagen.

    Retorna dict:
      board_found : bool
      board       : lista 3×3 ('X', 'O', None)
      confidence  : lista 3×3 de floats
      reserve     : {'X': int, 'O': int}
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se puede cargar: {image_path}")

    # Redimensionar si es muy grande
    max_dim = 1200
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_dim:
        s = max_dim / max(h0, w0)
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)))

    result = {
        'board_found': False,
        'board':       [[None] * 3 for _ in range(3)],
        'confidence':  [[0.0]  * 3 for _ in range(3)],
        'reserve':     {'X': 0, 'O': 0},
    }

    # ── Paso 1: warp del panel blanco ──────────────────────────────────
    warped, M = warp_panel(img, debug=debug)
    if warped is None:
        print("[WARN] No se encontró el panel blanco del tablero.")
        return result

    result['board_found'] = True

    # ── Paso 2: localizar el marco de la cuadrícula ────────────────────
    gx, gy, gw, gh = find_grid_bbox(warped, debug=debug)

    # ── Paso 3: clasificar las 9 celdas ───────────────────────────────
    board, conf = classify_cells(warped, gx, gy, gw, gh, debug=debug)
    result['board']      = board
    result['confidence'] = conf

    # ── Paso 4: reserva (en imagen original) ──────────────────────────
    h, w = img.shape[:2]
    panel_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    result['reserve'] = count_reserve(img, panel_corners)

    # ── Visualización ─────────────────────────────────────────────────
    if visualize:
        vis = warped.copy()
        cell_h = gh // 3
        cell_w = gw // 3
        MARG_H = max(int(cell_h * 0.15), 5)
        MARG_W = max(int(cell_w * 0.15), 5)
        side = warped.shape[0]

        for row in range(3):
            for col in range(3):
                y1 = max(0, gy + row * cell_h + MARG_H)
                y2 = min(side, gy + (row + 1) * cell_h - MARG_H)
                x1 = max(0, gx + col * cell_w + MARG_W)
                x2 = min(side, gx + (col + 1) * cell_w - MARG_W)
                piece = board[row][col]
                c_val = conf[row][col]
                color = (0, 0, 200) if piece == 'X' else \
                        (200, 120, 0) if piece == 'O' else (160, 160, 160)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{piece or '?'} {c_val:.2f}"
                cv2.putText(vis, label, (x1 + 4, (y1 + y2) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 220, 0), 2)
        out_path = Path(image_path).stem + "_result.jpg"
        cv2.imwrite(out_path, vis)
        print(f"[INFO] Resultado guardado en: {out_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# SALIDA
# ──────────────────────────────────────────────────────────────────────────────

def print_board(result):
    syms = {None: '.', 'X': 'X', 'O': 'O'}
    print()
    print("┌───┬───┬───┐")
    for ri, row in enumerate(result['board']):
        cells = " │ ".join(syms[c] for c in row)
        print(f"│ {cells} │")
        if ri < 2:
            print("├───┼───┼───┤")
    print("└───┴───┴───┘")
    print("\nConfianza:")
    for row in result['confidence']:
        print("  " + "  ".join(f"{v:.3f}" for v in row))
    r = result['reserve']
    print(f"\nFichas de reserva → X: {r['X']}  O: {r['O']}")
    print(f"Tablero detectado:    {result['board_found']}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detecta el estado del tablero de tres en raya desde una foto.")
    parser.add_argument("image",      help="Ruta a la imagen de entrada")
    parser.add_argument("--debug",    action="store_true",
                        help="Guardar imágenes de depuración (_debug_*.jpg)")
    parser.add_argument("--visualize", action="store_true",
                        help="Guardar imagen anotada con el resultado (_result.jpg)")
    parser.add_argument("--json",     action="store_true",
                        help="Imprimir resultado como JSON")
    args = parser.parse_args()

    try:
        result = analyze(args.image, debug=args.debug, visualize=args.visualize)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Serializar numpy floats
        def to_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return obj
        clean = {
            'board_found': result['board_found'],
            'board':       result['board'],
            'confidence':  [[to_serializable(v) for v in row]
                            for row in result['confidence']],
            'reserve':     result['reserve'],
        }
        print(json.dumps(clean, indent=2))
    else:
        print_board(result)


if __name__ == "__main__":
    main()
