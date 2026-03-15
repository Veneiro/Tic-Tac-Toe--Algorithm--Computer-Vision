import argparse
import os
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

try:
    import gradio as gr
    TIENE_GRADIO = True
except ImportError:
    gr = None
    TIENE_GRADIO = False

# ==========================================
# CONFIGURACIÓN
# ==========================================
RUTA_MODELO = './pt/best (3).pt'
RUTA_IMAGEN = 'original_image1.png'

NOMBRE_ROJA = 'red cross'
NOMBRE_AZUL = 'blue circle'
NOMBRE_CELDA = 'cells'

CONF_MIN_ROJA = 0.45
CONF_MIN_AZUL = 0.36
CONF_MIN_AZUL_PALIDA = 0.30
DIST_DUPLICADO_FALLBACK = 40
MARGEN_INTERIOR_NORM = 0.03
RADIO_CENTRO_CELDA_NORM = 0.58
SHRINK_FALLBACK = 0.12
ANCLA_Y_FICHA_NORM = 0.82
PESO_ANCLA_FILA = 0.85
PESO_ANCLA_COLUMNA = 0.35

UMBRAL_BLUR_LAPLACIAN = 85.0
UMBRAL_SATURACION_MEDIA = 42.0
FACTOR_CONTRASTE_PALIDO = 1.18
FACTOR_SATURACION_PALIDO = 1.55

EXPANSION_AREA_BASE = 0.04
EXPANSION_AREA_POR_CELDA_FALTANTE = 0.015
EXPANSION_AREA_MAX = 0.18
TOL_NIVELES_EJE = 0.16

T_SOFTMAX = 0.5
SEARCH_DEPTH = 2
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 5000
# ==========================================

app = Flask(__name__)
_MODEL = None


def _obtener_modelo():
    global _MODEL
    if _MODEL is None:
        print('Cargando modelo...')
        _MODEL = YOLO(RUTA_MODELO)
    return _MODEL


def _ordenar_puntos(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _expandir_poligono(src, factor):
    centro = np.mean(src, axis=0)
    return centro + (src - centro) * factor


def _contar_niveles_eje(valores, tolerancia=TOL_NIVELES_EJE):
    if len(valores) == 0:
        return 0, 0.5

    valores = np.sort(np.array(valores, dtype=np.float32))
    grupos = [[float(valores[0])]]

    for valor in valores[1:]:
        media_grupo = float(np.mean(grupos[-1]))
        if abs(float(valor) - media_grupo) <= tolerancia:
            grupos[-1].append(float(valor))
        else:
            grupos.append([float(valor)])

    medias = [float(np.mean(g)) for g in grupos]
    return len(medias), float(np.mean(medias))


def _calcular_limites_eje(num_niveles, media_niveles, expansion_base):
    if num_niveles >= 3:
        span = 1.0 + expansion_base
    elif num_niveles == 2:
        span = 1.50
    elif num_niveles == 1:
        span = 2.00
    else:
        span = 1.0 + expansion_base

    centro = 0.5 + (media_niveles - 0.5) * 0.6
    min_lim = centro - span / 2.0
    max_lim = centro + span / 2.0
    return float(min_lim), float(max_lim)


def _estimar_homografia_desde_celdas(celdas_centros, celdas_boxes):
    if len(celdas_boxes) >= 2:
        pts = []
        for x1, y1, x2, y2 in celdas_boxes:
            pts.extend([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
        pts = np.array(pts, dtype=np.float32)
    elif len(celdas_centros) >= 2:
        pts = np.array(celdas_centros, dtype=np.float32)
    else:
        return None, None

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    src = _ordenar_puntos(box)

    dst_unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H_pre = cv2.getPerspectiveTransform(src.astype(np.float32), dst_unit)

    if len(celdas_centros) > 0:
        centros = np.array(celdas_centros, dtype=np.float32).reshape(-1, 1, 2)
        uv = cv2.perspectiveTransform(centros, H_pre).reshape(-1, 2)
    else:
        uv = np.empty((0, 2), dtype=np.float32)

    n_cols, media_cols = _contar_niveles_eje(uv[:, 0] if len(uv) else [])
    n_filas, media_filas = _contar_niveles_eje(uv[:, 1] if len(uv) else [])

    faltantes = max(0, 9 - len(celdas_centros))
    expansion = EXPANSION_AREA_BASE + EXPANSION_AREA_POR_CELDA_FALTANTE * faltantes
    expansion = float(np.clip(expansion, EXPANSION_AREA_BASE, EXPANSION_AREA_MAX))

    umin, umax = _calcular_limites_eje(n_cols, media_cols, expansion)
    vmin, vmax = _calcular_limites_eje(n_filas, media_filas, expansion)

    tl, tr, br, bl = src
    eje_x = tr - tl
    eje_y = bl - tl

    src = np.array([
        tl + umin * eje_x + vmin * eje_y,
        tl + umax * eje_x + vmin * eje_y,
        tl + umax * eje_x + vmax * eje_y,
        tl + umin * eje_x + vmax * eje_y,
    ], dtype=np.float32)

    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H, src


def _estimar_anclas_rejilla(H, celdas_centros):
    if H is None or len(celdas_centros) < 4:
        return None, None

    pts = np.array(celdas_centros, dtype=np.float32).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    mask = (
        (uv[:, 0] >= -0.25) & (uv[:, 0] <= 1.25) &
        (uv[:, 1] >= -0.25) & (uv[:, 1] <= 1.25)
    )
    uv = uv[mask]

    if len(uv) < 4:
        return None, None

    cols = np.percentile(uv[:, 0], [16.67, 50.0, 83.33]).astype(np.float32)
    filas = np.percentile(uv[:, 1], [16.67, 50.0, 83.33]).astype(np.float32)

    cols = np.clip(cols, 0.05, 0.95)
    filas = np.clip(filas, 0.05, 0.95)
    return cols, filas


def _calcular_ancla_ficha(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    h = max(y2 - y1, 1.0)
    ay = y1 + ANCLA_Y_FICHA_NORM * h
    return cx, ay


def _metrica_calidad(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_media = float(np.mean(hsv[:, :, 1]))
    return blur_score, sat_media


def _aplicar_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _aplicar_realce_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * FACTOR_SATURACION_PALIDO, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * FACTOR_CONTRASTE_PALIDO, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _aplicar_nitidez(img_bgr):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img_bgr, -1, kernel)


def _score_resultado(resultados, id_roja, id_azul, id_celda):
    score = 0.0
    for box in resultados.boxes:
        clase = int(box.cls[0])
        conf = float(box.conf[0])
        if clase in [id_roja, id_azul]:
            score += conf * 2.2
        elif clase == id_celda:
            score += conf * 0.7
    return score


def _inferir_mejor_resultado(model, img, id_roja, id_azul, id_celda):
    blur_score, sat_media = _metrica_calidad(img)
    imagenes = [('original', img)]

    imagen_blur = blur_score < UMBRAL_BLUR_LAPLACIAN
    imagen_palida = sat_media < UMBRAL_SATURACION_MEDIA

    if imagen_palida:
        imagenes.append(('clahe', _aplicar_clahe(img)))
        imagenes.append(('realce_color', _aplicar_realce_color(img)))

    if imagen_blur:
        imagenes.append(('nitidez', _aplicar_nitidez(img)))

    if imagen_palida and imagen_blur:
        combinada = _aplicar_nitidez(_aplicar_realce_color(_aplicar_clahe(img)))
        imagenes.append(('clahe_color_nitidez', combinada))

    mejor = None
    mejor_score = -1.0

    for nombre, img_variante in imagenes:
        resultados = model(img_variante, conf=0.25, iou=0.5, imgsz=640)[0]
        score = _score_resultado(resultados, id_roja, id_azul, id_celda)
        if score > mejor_score:
            mejor_score = score
            mejor = (nombre, resultados, blur_score, sat_media)

    return mejor


def _procesar_imagen_modelo(img, model=None):
    if img is None:
        raise ValueError('La imagen recibida es None')

    if model is None:
        model = _obtener_modelo()

    print('Analizando imagen...')

    clases_modelo = model.names if hasattr(model, 'names') else {}
    id_roja = next((k for k, v in clases_modelo.items() if v == NOMBRE_ROJA), -1)
    id_azul = next((k for k, v in clases_modelo.items() if v == NOMBRE_AZUL), -1)
    id_celda = next((k for k, v in clases_modelo.items() if v == NOMBRE_CELDA), -1)

    variante_usada, resultados, blur_score, sat_media = _inferir_mejor_resultado(
        model, img, id_roja, id_azul, id_celda
    )

    conf_min_azul_actual = CONF_MIN_AZUL_PALIDA if sat_media < UMBRAL_SATURACION_MEDIA else CONF_MIN_AZUL

    celdas_centros = []
    celdas_boxes = []
    detecciones_fichas = []

    for box in resultados.boxes:
        clase = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if clase == id_celda:
            celdas_centros.append([cx, cy])
            celdas_boxes.append([x1, y1, x2, y2])
        elif clase in [id_roja, id_azul]:
            conf_min_clase = CONF_MIN_ROJA if clase == id_roja else conf_min_azul_actual
            if conf < conf_min_clase:
                continue
            valor = 1 if clase == id_roja else 2
            ax, ay = _calcular_ancla_ficha(x1, y1, x2, y2)
            detecciones_fichas.append({
                'tipo': valor,
                'centro': (cx, cy),
                'ancla': (ax, ay),
                'conf': conf
            })

    fichas_finales = []
    dist_duplicado = DIST_DUPLICADO_FALLBACK

    if len(celdas_boxes) >= 2:
        celdas_boxes_arr = np.array(celdas_boxes)
        anchos_celdas = celdas_boxes_arr[:, 2] - celdas_boxes_arr[:, 0]
        altos_celdas = celdas_boxes_arr[:, 3] - celdas_boxes_arr[:, 1]
        lado_ref = float(np.median(np.maximum(anchos_celdas, altos_celdas))) if len(anchos_celdas) else 1.0
        dist_duplicado = float(np.clip(0.35 * lado_ref, 18, 45))

    detecciones_fichas.sort(key=lambda x: x['conf'], reverse=True)
    for f in detecciones_fichas:
        duplicada = False
        for ff in fichas_finales:
            dist = np.linalg.norm(np.array(f['centro']) - np.array(ff['centro']))
            if dist < dist_duplicado:
                duplicada = True
                break
        if not duplicada:
            fichas_finales.append(f)

    matriz = np.zeros((3, 3), dtype=int)
    H, tablero_poly = _estimar_homografia_desde_celdas(celdas_centros, celdas_boxes)
    cols_anchor, filas_anchor = _estimar_anclas_rejilla(H, celdas_centros)
    modo_zona = 'homografia' if H is not None else 'fallback'
    min_x = min_y = max_x = max_y = None

    if H is not None and cols_anchor is not None and filas_anchor is not None:
        modo_zona = 'homografia_anclada'

    if H is not None:
        conf_celda = np.full((3, 3), -1.0, dtype=float)

        for f in fichas_finales:
            fx, fy = f['centro']
            ax, ay = f['ancla']

            p_centro = np.array([[[fx, fy]]], dtype=np.float32)
            p_ancla = np.array([[[ax, ay]]], dtype=np.float32)
            u_c, v_c = cv2.perspectiveTransform(p_centro, H)[0][0]
            u_a, v_a = cv2.perspectiveTransform(p_ancla, H)[0][0]

            u = (1.0 - PESO_ANCLA_COLUMNA) * u_c + PESO_ANCLA_COLUMNA * u_a
            v = (1.0 - PESO_ANCLA_FILA) * v_c + PESO_ANCLA_FILA * v_a

            if not (MARGEN_INTERIOR_NORM <= u <= 1 - MARGEN_INTERIOR_NORM and
                    MARGEN_INTERIOR_NORM <= v <= 1 - MARGEN_INTERIOR_NORM):
                continue

            if cols_anchor is not None and filas_anchor is not None:
                col = int(np.argmin(np.abs(cols_anchor - u)))
                fila = int(np.argmin(np.abs(filas_anchor - v)))

                du_cell = abs((u - cols_anchor[col]) * 3.0)
                dv_cell = abs((v - filas_anchor[fila]) * 3.0)
                dist_centro = float(np.hypot(du_cell, dv_cell))
            else:
                gx = float(np.clip(u * 3.0, 0, 2.9999))
                gy = float(np.clip(v * 3.0, 0, 2.9999))
                col = int(gx)
                fila = int(gy)
                dist_centro = float(np.hypot(gx - (col + 0.5), gy - (fila + 0.5)))

            if dist_centro > RADIO_CENTRO_CELDA_NORM:
                continue

            if f['conf'] > conf_celda[fila][col]:
                conf_celda[fila][col] = f['conf']
                matriz[fila][col] = f['tipo']

        filas_str = [','.join(map(str, f)) for f in matriz]
        resultado_str = 'tablero={' + ';'.join(filas_str) + '}'

    elif len(celdas_centros) >= 2:
        celdas_boxes_arr = np.array(celdas_boxes)
        min_x = float(np.min(celdas_boxes_arr[:, 0]))
        min_y = float(np.min(celdas_boxes_arr[:, 1]))
        max_x = float(np.max(celdas_boxes_arr[:, 2]))
        max_y = float(np.max(celdas_boxes_arr[:, 3]))

        anchos_celdas = celdas_boxes_arr[:, 2] - celdas_boxes_arr[:, 0]
        altos_celdas = celdas_boxes_arr[:, 3] - celdas_boxes_arr[:, 1]
        lado_ref = float(np.median(np.maximum(anchos_celdas, altos_celdas))) if len(anchos_celdas) else 1.0

        shrink = SHRINK_FALLBACK * lado_ref
        min_x_val = min_x + shrink
        min_y_val = min_y + shrink
        max_x_val = max_x - shrink
        max_y_val = max_y - shrink

        if max_x_val <= min_x_val or max_y_val <= min_y_val:
            resultado_str = 'tablero={Error: Zona de juego invalida}'
            min_x_val, min_y_val, max_x_val, max_y_val = min_x, min_y, max_x, max_y
        else:
            min_x, min_y, max_x, max_y = min_x_val, min_y_val, max_x_val, max_y_val

            ancho_t = max_x - min_x
            alto_t = max_y - min_y
            ancho_celda = max(ancho_t / 3.0, 1.0)
            alto_celda = max(alto_t / 3.0, 1.0)
            max_dist_centro_celda = 0.45 * np.hypot(ancho_celda, alto_celda)

            conf_celda = np.full((3, 3), -1.0, dtype=float)

            for f in fichas_finales:
                fx, fy = f['centro']
                ax, ay = f['ancla']

                fx_eff = (1.0 - PESO_ANCLA_COLUMNA) * fx + PESO_ANCLA_COLUMNA * ax
                fy_eff = (1.0 - PESO_ANCLA_FILA) * fy + PESO_ANCLA_FILA * ay

                if not (min_x <= fx_eff <= max_x and min_y <= fy_eff <= max_y):
                    continue

                rel_x = (fx_eff - min_x) / (ancho_t if ancho_t > 0 else 1)
                rel_y = (fy_eff - min_y) / (alto_t if alto_t > 0 else 1)
                col = int(np.clip(rel_x * 3, 0, 2))
                fila = int(np.clip(rel_y * 3, 0, 2))

                cx_celda = min_x + (col + 0.5) * ancho_celda
                cy_celda = min_y + (fila + 0.5) * alto_celda
                dist_centro = float(np.hypot(fx_eff - cx_celda, fy_eff - cy_celda))
                if dist_centro > max_dist_centro_celda:
                    continue

                if f['conf'] > conf_celda[fila][col]:
                    conf_celda[fila][col] = f['conf']
                    matriz[fila][col] = f['tipo']

            filas_str = [','.join(map(str, f)) for f in matriz]
            resultado_str = 'tablero={' + ';'.join(filas_str) + '}'

    else:
        resultado_str = 'tablero={Error: Pocas celdas detectadas}'

    print('\n' + '=' * 30)
    print('MATRIZ DETECTADA:')
    print(resultado_str)
    print(
        f'Variante imagen: {variante_usada} | blur={blur_score:.1f} | sat_media={sat_media:.1f}'
    )
    print(
        f'Umbral roja={CONF_MIN_ROJA:.2f} | Umbral azul={conf_min_azul_actual:.2f}'
    )
    print(
        f'Modo zona: {modo_zona} | Celdas detectadas: {len(celdas_centros)} | '
        f'Fichas filtradas: {len(fichas_finales)}'
    )
    print('=' * 30 + '\n')

    img_anotada = resultados.plot()

    if tablero_poly is not None:
        poly = tablero_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_anotada, [poly], True, (0, 255, 0), 2)
        pt = tuple(tablero_poly[0].astype(int))
        cv2.putText(
            img_anotada,
            'Area Tablero (warp)',
            (pt[0], pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    elif len(celdas_centros) >= 2 and None not in (min_x, min_y, max_x, max_y):
        cv2.rectangle(img_anotada, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
        cv2.putText(
            img_anotada,
            'Area Tablero',
            (int(min_x), int(min_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    alto, ancho = img_anotada.shape[:2]
    if alto > 800 or ancho > 800:
        escala = 800 / max(alto, ancho)
        img_anotada = cv2.resize(img_anotada, (0, 0), fx=escala, fy=escala)

    info = {
        'variante_usada': variante_usada,
        'blur_score': blur_score,
        'sat_media': sat_media,
        'modo_zona': modo_zona,
        'celdas_detectadas': len(celdas_centros),
        'fichas_filtradas': len(fichas_finales),
    }
    return resultado_str, img_anotada, info


def probar_modelo_optimizado(ruta_img):
    img = cv2.imread(ruta_img)
    if img is None:
        raise ValueError(f'No se pudo cargar la imagen: {ruta_img}')

    resultado_str, img_anotada, _ = _procesar_imagen_modelo(img)
    return resultado_str, img_anotada


# ==========================================
# LÓGICA DEL JUEGO (desde main_modelov2)
# ==========================================
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
        for i, j in moves:
            b = board.copy()
            b[i, j] = 2
            best = max(best, minimax_numeric(b, depth - 1, False))
        return best

    best = np.inf
    for i, j in moves:
        b = board.copy()
        b[i, j] = 1
        best = min(best, minimax_numeric(b, depth - 1, True))
    return best


def evaluate_moves_numeric(board, depth):
    moves = possible_moves_numeric(board)
    scores = []
    for i, j in moves:
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
# UTILIDADES ESP32 / FLASK
# ==========================================
def _decodificar_imagen_request():
    if 'image' in request.files:
        file = request.files['image']
        data = file.read()
    else:
        data = request.data

    if not data:
        return None, 'No se recibieron bytes de imagen'

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 'No se pudo decodificar la imagen'

    return img, None


def analizar_tablero_esp32(img):
    resultado_str, _, _ = _procesar_imagen_modelo(img)
    return resultado_str


def lanzar_interfaz_web():
    if not TIENE_GRADIO:
        raise RuntimeError('Gradio no está instalado. Instala con: pip install gradio')

    def ejecutar_deteccion_web(ruta_img):
        if not ruta_img:
            return 'Selecciona o arrastra una imagen.', None
        if not os.path.isfile(ruta_img):
            return f'Archivo no encontrado: {ruta_img}', None

        try:
            resultado, img_anotada = probar_modelo_optimizado(ruta_img)
            img_rgb = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)
            return resultado, img_rgb
        except Exception as e:
            return f'Error procesando imagen: {e}', None

    with gr.Blocks(title='Detector Tic-Tac-Toe + ESP32') as demo:
        gr.Markdown('## Detector Tic-Tac-Toe')
        gr.Markdown('Arrastra/suelta una imagen o selecciónala, y pulsa **Detectar**.')

        entrada = gr.Image(type='filepath', label='Imagen de entrada')
        boton = gr.Button('Detectar', variant='primary')
        salida_texto = gr.Textbox(label='Resultado tablero')
        salida_img = gr.Image(label='Imagen anotada')

        boton.click(
            fn=ejecutar_deteccion_web,
            inputs=[entrada],
            outputs=[salida_texto, salida_img]
        )

    demo.launch(inbrowser=True, share=False)


# ==========================================
# SERVIDOR WEB (FLASK)
# ==========================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'modelo': os.path.basename(RUTA_MODELO)}), 200


@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        img, error = _decodificar_imagen_request()
        if error:
            return error, 400

        print('Recibida imagen desde ESP32/cliente. Analizando con modelDetector...')
        resultado = analizar_tablero_esp32(img)
        print(f'RESULTADO: {resultado}')
        return resultado, 200

    except Exception as e:
        print(f'Error en /procesar: {e}')
        traceback.print_exc()
        return 'Error Server', 500


@app.route('/movimiento', methods=['POST'])
def movimiento():
    try:
        payload = request.get_json(silent=True)
        if not payload or 'matriz' not in payload:
            return jsonify({'error': 'Debes enviar JSON con la clave "matriz"'}), 400

        matrix = np.array(payload['matriz'])
        if matrix.shape != (3, 3):
            return jsonify({'error': 'La matriz debe ser de tamaño 3x3'}), 400

        matrix = matrix.astype(int)
        move, score, depth, temperature, probabilities = choose_move_softmax(matrix)

        if move is None:
            return jsonify({'movimiento': None, 'mensaje': 'Fin'}), 200

        return jsonify({
            'movimiento': {'fila': int(move[0]), 'columna': int(move[1])},
            'score': score,
            'depth': depth,
            'temperature': temperature,
            'probabilities': probabilities,
        }), 200

    except Exception as e:
        print(f'Error en /movimiento: {e}')
        traceback.print_exc()
        return jsonify({'error': 'Error Server'}), 500


def lanzar_servidor_flask(host=DEFAULT_HOST, port=DEFAULT_PORT):
    _obtener_modelo()
    print(f'Servidor arrancando en http://{host}:{port}')
    print(f'Endpoint ESP32-CAM: http://{host}:{port}/procesar')
    print(f'Endpoint IA movimiento: http://{host}:{port}/movimiento')
    app.run(host=host, port=port)


def _modo_imagen(ruta_img):
    resultado, img_anotada = probar_modelo_optimizado(ruta_img)
    salida = 'modelDetector_esp32_resultado.jpg'
    cv2.imwrite(salida, img_anotada)
    print(f'Resultado: {resultado}')
    print(f'Imagen anotada guardada en: {os.path.abspath(salida)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Versión combinada de modelDetector con Flask y comunicación para ESP32-CAM.'
    )
    parser.add_argument('--modo', choices=['server', 'web', 'imagen'], default='server')
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--imagen', default=RUTA_IMAGEN)
    args = parser.parse_args()

    if args.modo == 'web':
        lanzar_interfaz_web()
    elif args.modo == 'imagen':
        _modo_imagen(args.imagen)
    else:
        lanzar_servidor_flask(host=args.host, port=args.port)
