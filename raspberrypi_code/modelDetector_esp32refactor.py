"""
modelDetector_esp32.py
Detector de tablero Tic-Tac-Toe con YOLO + servidor Flask para ESP32-CAM.
"""

import argparse
import os
import threading
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

# Rutas
RUTA_MODELO = './pt/best (3).pt'
RUTA_IMAGEN = 'original_image1.png'

# Nombres de clases
NOMBRE_ROJA  = 'red cross'
NOMBRE_AZUL  = 'blue circle'
NOMBRE_CELDA = 'cells'

# Umbrales de confianza
CONF_MIN_ROJA       = 0.45
CONF_MIN_AZUL       = 0.36
CONF_MIN_AZUL_PALIDA = 0.30

# Geometría del tablero
DIST_DUPLICADO_FALLBACK  = 40
MARGEN_INTERIOR_NORM     = 0.03
RADIO_CENTRO_CELDA_NORM  = 0.58
SHRINK_FALLBACK          = 0.12
ANCLA_Y_FICHA_NORM       = 0.82
PESO_ANCLA_FILA          = 0.85
PESO_ANCLA_COLUMNA       = 0.35

# Calidad de imagen
UMBRAL_BLUR_LAPLACIAN    = 85.0
UMBRAL_SATURACION_MEDIA  = 42.0
FACTOR_CONTRASTE_PALIDO  = 1.18
FACTOR_SATURACION_PALIDO = 1.55

# Expansión del área de búsqueda
EXPANSION_AREA_BASE             = 0.04
EXPANSION_AREA_POR_CELDA_FALTANTE = 0.015
EXPANSION_AREA_MAX              = 0.18
TOL_NIVELES_EJE                 = 0.16

# IA
T_SOFTMAX           = 0.5
SEARCH_DEPTH        = 2

# Servidor
DEFAULT_HOST        = 'localhost'
DEFAULT_PORT        = 5000
DEFAULT_GRADIO_PORT = 7860


# ==========================================
# MODELO (singleton)
# ==========================================

app = Flask(__name__)
_MODEL = None


def _obtener_modelo() -> YOLO:
    global _MODEL
    if _MODEL is None:
        print('Cargando modelo...')
        _MODEL = YOLO(RUTA_MODELO)
    return _MODEL


# ==========================================
# VISIÓN / GEOMETRÍA
# ==========================================

def _ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos en: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ], dtype=np.float32)


def _expandir_poligono(src: np.ndarray, factor: float) -> np.ndarray:
    """Expande (o contrae) un polígono respecto a su centro."""
    centro = np.mean(src, axis=0)
    return centro + (src - centro) * factor


def _contar_niveles_eje(valores: list, tolerancia: float = TOL_NIVELES_EJE):
    """Agrupa valores en niveles y devuelve (num_niveles, media_global)."""
    if not valores:
        return 0, 0.5

    ordenados = np.sort(np.array(valores, dtype=np.float32))
    grupos = [[float(ordenados[0])]]

    for v in ordenados[1:]:
        if abs(float(v) - float(np.mean(grupos[-1]))) <= tolerancia:
            grupos[-1].append(float(v))
        else:
            grupos.append([float(v)])

    medias = [float(np.mean(g)) for g in grupos]
    return len(medias), float(np.mean(medias))


def _calcular_limites_eje(n_niveles: int, media_niveles: float, expansion_base: float):
    """Devuelve (min_lim, max_lim) para un eje del tablero."""
    span_por_niveles = {3: 1.0 + expansion_base, 2: 1.50, 1: 2.00}
    span = span_por_niveles.get(n_niveles, 1.0 + expansion_base)

    centro  = 0.5 + (media_niveles - 0.5) * 0.6
    min_lim = centro - span / 2.0
    max_lim = centro + span / 2.0
    return float(min_lim), float(max_lim)


def _estimar_homografia_desde_celdas(celdas_centros: list, celdas_boxes: list):
    """Calcula la homografía que mapea el tablero a coordenadas [0,1]²."""
    if len(celdas_boxes) >= 2:
        pts = []
        for x1, y1, x2, y2 in celdas_boxes:
            pts.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        pts = np.array(pts, dtype=np.float32)
    elif len(celdas_centros) >= 2:
        pts = np.array(celdas_centros, dtype=np.float32)
    else:
        return None, None

    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect)
    src  = _ordenar_puntos(box)

    dst_unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H_pre    = cv2.getPerspectiveTransform(src.astype(np.float32), dst_unit)

    if celdas_centros:
        centros = np.array(celdas_centros, dtype=np.float32).reshape(-1, 1, 2)
        uv = cv2.perspectiveTransform(centros, H_pre).reshape(-1, 2)
    else:
        uv = np.empty((0, 2), dtype=np.float32)

    col_vals  = uv[:, 0].tolist() if len(uv) else []
    fila_vals = uv[:, 1].tolist() if len(uv) else []
    n_cols,  media_cols  = _contar_niveles_eje(col_vals)
    n_filas, media_filas = _contar_niveles_eje(fila_vals)

    faltantes = max(0, 9 - len(celdas_centros))
    expansion = float(np.clip(
        EXPANSION_AREA_BASE + EXPANSION_AREA_POR_CELDA_FALTANTE * faltantes,
        EXPANSION_AREA_BASE,
        EXPANSION_AREA_MAX,
    ))

    umin, umax = _calcular_limites_eje(n_cols,  media_cols,  expansion)
    vmin, vmax = _calcular_limites_eje(n_filas, media_filas, expansion)

    tl, tr, _br, bl = src
    eje_x = tr - tl
    eje_y = bl - tl

    src_final = np.array([
        tl + umin * eje_x + vmin * eje_y,
        tl + umax * eje_x + vmin * eje_y,
        tl + umax * eje_x + vmax * eje_y,
        tl + umin * eje_x + vmax * eje_y,
    ], dtype=np.float32)

    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H   = cv2.getPerspectiveTransform(src_final, dst)
    return H, src_final


def _estimar_anclas_rejilla(H: np.ndarray, celdas_centros: list):
    """Estima los centros de columnas y filas de la rejilla 3×3."""
    if H is None or len(celdas_centros) < 4:
        return None, None

    pts = np.array(celdas_centros, dtype=np.float32).reshape(-1, 1, 2)
    uv  = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    mask = (
        (uv[:, 0] >= -0.25) & (uv[:, 0] <= 1.25) &
        (uv[:, 1] >= -0.25) & (uv[:, 1] <= 1.25)
    )
    uv = uv[mask]

    if len(uv) < 4:
        return None, None

    percentiles = [16.67, 50.0, 83.33]
    cols  = np.clip(np.percentile(uv[:, 0], percentiles).astype(np.float32), 0.05, 0.95)
    filas = np.clip(np.percentile(uv[:, 1], percentiles).astype(np.float32), 0.05, 0.95)
    return cols, filas


def _calcular_ancla_ficha(x1: float, y1: float, x2: float, y2: float):
    """Devuelve el punto de anclaje inferior de una ficha."""
    cx = (x1 + x2) / 2.0
    ay = y1 + ANCLA_Y_FICHA_NORM * max(y2 - y1, 1.0)
    return cx, ay


# ==========================================
# PREPROCESADO DE IMAGEN
# ==========================================

def _metrica_calidad(img_bgr: np.ndarray):
    """Devuelve (blur_score, sat_media) de la imagen."""
    gris      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    hsv        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_media  = float(np.mean(hsv[:, :, 1]))
    return blur_score, sat_media


def _aplicar_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _aplicar_realce_color(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * FACTOR_SATURACION_PALIDO, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * FACTOR_CONTRASTE_PALIDO,  0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _aplicar_nitidez(img_bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img_bgr, -1, kernel)


def _generar_variantes(img: np.ndarray, imagen_palida: bool, imagen_blur: bool) -> list:
    """Genera las variantes de imagen a evaluar según la calidad."""
    variantes = [('original', img)]
    if imagen_palida:
        variantes.append(('clahe',        _aplicar_clahe(img)))
        variantes.append(('realce_color', _aplicar_realce_color(img)))
    if imagen_blur:
        variantes.append(('nitidez',      _aplicar_nitidez(img)))
    if imagen_palida and imagen_blur:
        variantes.append(('clahe_color_nitidez',
                          _aplicar_nitidez(_aplicar_realce_color(_aplicar_clahe(img)))))
    return variantes


# ==========================================
# INFERENCIA
# ==========================================

def _score_resultado(resultados, id_roja: int, id_azul: int, id_celda: int) -> float:
    """Puntúa un resultado YOLO priorizando fichas sobre celdas."""
    score = 0.0
    for box in resultados.boxes:
        clase = int(box.cls[0])
        conf  = float(box.conf[0])
        if clase in (id_roja, id_azul):
            score += conf * 2.2
        elif clase == id_celda:
            score += conf * 0.7
    return score


def _inferir_mejor_resultado(model, img: np.ndarray, id_roja: int, id_azul: int, id_celda: int):
    """Infiere con la variante de imagen que mayor score obtiene."""
    blur_score, sat_media = _metrica_calidad(img)
    variantes = _generar_variantes(
        img,
        imagen_palida=sat_media < UMBRAL_SATURACION_MEDIA,
        imagen_blur=blur_score  < UMBRAL_BLUR_LAPLACIAN,
    )

    mejor_nombre, mejor_resultado, mejor_score = None, None, -1.0
    for nombre, img_variante in variantes:
        resultado = model(img_variante, conf=0.25, iou=0.5, imgsz=640)[0]
        score = _score_resultado(resultado, id_roja, id_azul, id_celda)
        if score > mejor_score:
            mejor_score    = score
            mejor_nombre   = nombre
            mejor_resultado = resultado

    return mejor_nombre, mejor_resultado, blur_score, sat_media


# ==========================================
# PROCESADO DEL TABLERO
# ==========================================

def _extraer_detecciones(resultados, id_roja: int, id_azul: int, id_celda: int,
                          conf_min_azul_actual: float):
    """Separa las detecciones en celdas y fichas, filtrando por confianza."""
    celdas_centros  = []
    celdas_boxes    = []
    fichas_candidatas = []

    for box in resultados.boxes:
        clase = int(box.cls[0])
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if clase == id_celda:
            celdas_centros.append([cx, cy])
            celdas_boxes.append([x1, y1, x2, y2])
        elif clase in (id_roja, id_azul):
            conf_min = CONF_MIN_ROJA if clase == id_roja else conf_min_azul_actual
            if conf < conf_min:
                continue
            fichas_candidatas.append({
                'tipo':  1 if clase == id_roja else 2,
                'centro': (cx, cy),
                'ancla':  _calcular_ancla_ficha(x1, y1, x2, y2),
                'conf':   conf,
            })

    return celdas_centros, celdas_boxes, fichas_candidatas


def _eliminar_duplicados(fichas: list, dist_min: float) -> list:
    """Elimina fichas duplicadas priorizando las de mayor confianza."""
    fichas_finales = []
    fichas.sort(key=lambda f: f['conf'], reverse=True)

    for ficha in fichas:
        es_duplicada = any(
            np.linalg.norm(np.array(ficha['centro']) - np.array(ff['centro'])) < dist_min
            for ff in fichas_finales
        )
        if not es_duplicada:
            fichas_finales.append(ficha)

    return fichas_finales


def _calcular_dist_duplicado(celdas_boxes: list) -> float:
    if len(celdas_boxes) < 2:
        return float(DIST_DUPLICADO_FALLBACK)
    arr = np.array(celdas_boxes)
    lados = np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])
    return float(np.clip(0.35 * np.median(lados), 18, 45))


def _asignar_con_homografia(fichas: list, H: np.ndarray,
                             cols_anchor, filas_anchor) -> np.ndarray:
    """Asigna fichas a celdas usando la homografía estimada."""
    matriz    = np.zeros((3, 3), dtype=int)
    conf_celda = np.full((3, 3), -1.0, dtype=float)

    for f in fichas:
        fx, fy = f['centro']
        ax, ay = f['ancla']

        u_c, v_c = cv2.perspectiveTransform(
            np.array([[[fx, fy]]], dtype=np.float32), H)[0][0]
        u_a, v_a = cv2.perspectiveTransform(
            np.array([[[ax, ay]]], dtype=np.float32), H)[0][0]

        u = (1.0 - PESO_ANCLA_COLUMNA) * u_c + PESO_ANCLA_COLUMNA * u_a
        v = (1.0 - PESO_ANCLA_FILA)    * v_c + PESO_ANCLA_FILA    * v_a

        margen = MARGEN_INTERIOR_NORM
        if not (margen <= u <= 1 - margen and margen <= v <= 1 - margen):
            continue

        if cols_anchor is not None and filas_anchor is not None:
            col  = int(np.argmin(np.abs(cols_anchor  - u)))
            fila = int(np.argmin(np.abs(filas_anchor - v)))
            du   = abs((u - cols_anchor[col])  * 3.0)
            dv   = abs((v - filas_anchor[fila]) * 3.0)
        else:
            gx  = float(np.clip(u * 3.0, 0, 2.9999))
            gy  = float(np.clip(v * 3.0, 0, 2.9999))
            col, fila = int(gx), int(gy)
            du  = gx - (col + 0.5)
            dv  = gy - (fila + 0.5)

        if float(np.hypot(du, dv)) > RADIO_CENTRO_CELDA_NORM:
            continue

        if f['conf'] > conf_celda[fila][col]:
            conf_celda[fila][col] = f['conf']
            matriz[fila][col]     = f['tipo']

    return matriz


def _asignar_con_fallback(fichas: list, celdas_boxes: list) -> tuple:
    """Asigna fichas a celdas usando bounding-box como fallback."""
    arr   = np.array(celdas_boxes)
    min_x = float(np.min(arr[:, 0]))
    min_y = float(np.min(arr[:, 1]))
    max_x = float(np.max(arr[:, 2]))
    max_y = float(np.max(arr[:, 3]))

    lados   = np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])
    lado_ref = float(np.median(lados)) if len(lados) else 1.0
    shrink   = SHRINK_FALLBACK * lado_ref

    min_xv, min_yv = min_x + shrink, min_y + shrink
    max_xv, max_yv = max_x - shrink, max_y - shrink

    if max_xv <= min_xv or max_yv <= min_yv:
        return None, (min_x, min_y, max_x, max_y)

    min_x, min_y, max_x, max_y = min_xv, min_yv, max_xv, max_yv
    ancho_t  = max_x - min_x
    alto_t   = max_y - min_y
    ancho_c  = max(ancho_t / 3.0, 1.0)
    alto_c   = max(alto_t  / 3.0, 1.0)
    max_dist = 0.45 * np.hypot(ancho_c, alto_c)

    matriz    = np.zeros((3, 3), dtype=int)
    conf_celda = np.full((3, 3), -1.0, dtype=float)

    for f in fichas:
        fx, fy = f['centro']
        ax, ay = f['ancla']
        fx_eff = (1.0 - PESO_ANCLA_COLUMNA) * fx + PESO_ANCLA_COLUMNA * ax
        fy_eff = (1.0 - PESO_ANCLA_FILA)    * fy + PESO_ANCLA_FILA    * ay

        if not (min_x <= fx_eff <= max_x and min_y <= fy_eff <= max_y):
            continue

        rel_x  = (fx_eff - min_x) / (ancho_t if ancho_t > 0 else 1)
        rel_y  = (fy_eff - min_y) / (alto_t  if alto_t  > 0 else 1)
        col    = int(np.clip(rel_x * 3, 0, 2))
        fila   = int(np.clip(rel_y * 3, 0, 2))

        cx_c  = min_x + (col  + 0.5) * ancho_c
        cy_c  = min_y + (fila + 0.5) * alto_c
        if float(np.hypot(fx_eff - cx_c, fy_eff - cy_c)) > max_dist:
            continue

        if f['conf'] > conf_celda[fila][col]:
            conf_celda[fila][col] = f['conf']
            matriz[fila][col]     = f['tipo']

    return matriz, (min_x, min_y, max_x, max_y)


def _matriz_a_str(matriz: np.ndarray) -> str:
    return 'tablero={' + ';'.join(','.join(map(str, fila)) for fila in matriz) + '}'


def _anotar_imagen(img_anotada: np.ndarray, tablero_poly,
                   celdas_centros: list, bbox) -> np.ndarray:
    """Dibuja el área del tablero sobre la imagen anotada."""
    VERDE = (0, 255, 0)

    if tablero_poly is not None:
        poly = tablero_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_anotada, [poly], True, VERDE, 2)
        pt = tuple(tablero_poly[0].astype(int))
        cv2.putText(img_anotada, 'Area Tablero (warp)',
                    (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERDE, 1)

    elif len(celdas_centros) >= 2 and bbox is not None and None not in bbox:
        min_x, min_y, max_x, max_y = bbox
        cv2.rectangle(img_anotada, (int(min_x), int(min_y)), (int(max_x), int(max_y)), VERDE, 2)
        cv2.putText(img_anotada, 'Area Tablero',
                    (int(min_x), int(min_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, VERDE, 1)

    alto, ancho = img_anotada.shape[:2]
    if alto > 800 or ancho > 800:
        escala = 800 / max(alto, ancho)
        img_anotada = cv2.resize(img_anotada, (0, 0), fx=escala, fy=escala)

    return img_anotada


def _procesar_imagen_modelo(img: np.ndarray, model=None):
    """Pipeline completo: inferencia → asignación de celdas → anotación."""
    if img is None:
        raise ValueError('La imagen recibida es None')

    if model is None:
        model = _obtener_modelo()

    print('Analizando imagen...')

    # IDs de clase
    clases = model.names if hasattr(model, 'names') else {}
    id_roja  = next((k for k, v in clases.items() if v == NOMBRE_ROJA),  -1)
    id_azul  = next((k for k, v in clases.items() if v == NOMBRE_AZUL),  -1)
    id_celda = next((k for k, v in clases.items() if v == NOMBRE_CELDA), -1)

    # Inferencia con la mejor variante de imagen
    variante_usada, resultados, blur_score, sat_media = _inferir_mejor_resultado(
        model, img, id_roja, id_azul, id_celda
    )

    conf_min_azul_actual = (
        CONF_MIN_AZUL_PALIDA if sat_media < UMBRAL_SATURACION_MEDIA else CONF_MIN_AZUL
    )

    celdas_centros, celdas_boxes, fichas_candidatas = _extraer_detecciones(
        resultados, id_roja, id_azul, id_celda, conf_min_azul_actual
    )

    dist_duplicado = _calcular_dist_duplicado(celdas_boxes)
    fichas_finales  = _eliminar_duplicados(fichas_candidatas, dist_duplicado)

    H, tablero_poly = _estimar_homografia_desde_celdas(celdas_centros, celdas_boxes)
    cols_anchor, filas_anchor = _estimar_anclas_rejilla(H, celdas_centros)

    # Determinar modo de asignación
    bbox = (None, None, None, None)

    if H is not None:
        modo_zona = 'homografia_anclada' if cols_anchor is not None else 'homografia'
        matriz    = _asignar_con_homografia(fichas_finales, H, cols_anchor, filas_anchor)
        resultado_str = _matriz_a_str(matriz)
    elif len(celdas_centros) >= 2:
        modo_zona = 'fallback'
        matriz_fb, bbox = _asignar_con_fallback(fichas_finales, celdas_boxes)
        if matriz_fb is None:
            resultado_str = 'tablero={Error: Zona de juego invalida}'
        else:
            resultado_str = _matriz_a_str(matriz_fb)
    else:
        modo_zona     = 'fallback'
        resultado_str = 'tablero={Error: Pocas celdas detectadas}'

    # Log de diagnóstico
    separador = '=' * 30
    print(f'\n{separador}')
    print('MATRIZ DETECTADA:')
    print(resultado_str)
    print(f'Variante imagen: {variante_usada} | blur={blur_score:.1f} | sat_media={sat_media:.1f}')
    print(f'Umbral roja={CONF_MIN_ROJA:.2f} | Umbral azul={conf_min_azul_actual:.2f}')
    print(f'Modo zona: {modo_zona} | Celdas detectadas: {len(celdas_centros)} | '
          f'Fichas filtradas: {len(fichas_finales)}')
    print(f'{separador}\n')

    img_anotada = _anotar_imagen(resultados.plot(), tablero_poly, celdas_centros, bbox)

    info = {
        'variante_usada':    variante_usada,
        'blur_score':        blur_score,
        'sat_media':         sat_media,
        'modo_zona':         modo_zona,
        'celdas_detectadas': len(celdas_centros),
        'fichas_filtradas':  len(fichas_finales),
    }
    return resultado_str, img_anotada, info


def probar_modelo_optimizado(ruta_img: str):
    img = cv2.imread(ruta_img)
    if img is None:
        raise ValueError(f'No se pudo cargar la imagen: {ruta_img}')
    resultado_str, img_anotada, _ = _procesar_imagen_modelo(img)
    return resultado_str, img_anotada


# ==========================================
# LÓGICA DEL JUEGO (minimax + softmax)
# ==========================================

def possible_moves_numeric(board: np.ndarray) -> list:
    return [(i, j) for i in range(board.shape[0])
                   for j in range(board.shape[1])
                   if board[i, j] == 0]


def evaluate_numeric(board: np.ndarray) -> int:
    lineas = (
        list(board)
        + list(board.T)
        + [np.diag(board), np.diag(np.fliplr(board))]
    )

    score = 0
    for linea in lineas:
        if np.all(linea == 2):
            return 100
        if np.all(linea == 1):
            return -100
        if np.count_nonzero(linea == 2) == 2 and np.count_nonzero(linea == 0) == 1:
            score += 10
        if np.count_nonzero(linea == 1) == 2 and np.count_nonzero(linea == 0) == 1:
            score -= 8

    return score


def minimax_numeric(board: np.ndarray, depth: int, maximizing: bool) -> float:
    score = evaluate_numeric(board)
    if abs(score) == 100 or depth == 0:
        return score

    moves = possible_moves_numeric(board)
    if not moves:
        return score

    if maximizing:
        best = -np.inf
        for i, j in moves:
            b = board.copy(); b[i, j] = 2
            best = max(best, minimax_numeric(b, depth - 1, False))
    else:
        best = np.inf
        for i, j in moves:
            b = board.copy(); b[i, j] = 1
            best = min(best, minimax_numeric(b, depth - 1, True))

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


def choose_move_softmax(board: np.ndarray,
                         temperature: float = T_SOFTMAX,
                         base_depth: int = SEARCH_DEPTH):
    empty = int(np.count_nonzero(board == 0))

    if empty >= 7:
        depth, temp = max(1, base_depth - 1), temperature + 0.3
    elif empty >= 4:
        depth, temp = base_depth, temperature
    else:
        depth, temp = base_depth + 2, max(0.05, temperature - 0.3)

    moves, scores = evaluate_moves_numeric(board, depth)
    if not moves:
        return None, None, depth, temp, []

    noisy  = scores + np.random.normal(0, 0.3, size=len(scores))
    probs  = softmax(noisy, temp)
    idx    = int(np.random.choice(len(moves), p=probs))
    return moves[idx], float(noisy[idx]), depth, temp, probs.tolist()


# ==========================================
# UTILIDADES FLASK / ESP32
# ==========================================

def _decodificar_imagen_request():
    """Lee bytes de la petición y los decodifica como imagen OpenCV."""
    data = (request.files['image'].read()
            if 'image' in request.files
            else request.data)

    if not data:
        return None, 'No se recibieron bytes de imagen'

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, 'No se pudo decodificar la imagen'

    return img, None


def analizar_tablero_esp32(img: np.ndarray) -> str:
    resultado_str, _, _ = _procesar_imagen_modelo(img)
    return resultado_str


# ==========================================
# INTERFAZ WEB (Gradio)
# ==========================================

def _construir_app_gradio():
    """
    Construye y devuelve el objeto `demo` de Gradio sin lanzarlo.
    Así puede reutilizarse tanto en modo 'web' (standalone) como en modo 'full'
    (arrancado en un hilo junto a Flask).
    El modelo se asume ya cargado por quien llame a esta función.
    """
    def ejecutar_deteccion_web(imagen_entrada):
        """
        Callback de Gradio. `imagen_entrada` puede ser:
          - None        → el usuario no ha subido nada todavía.
          - str         → ruta temporal (Gradio ≤3.x con type='filepath').
          - dict        → {"path": str, ...} (Gradio ≥4.x).
          - np.ndarray  → array RGB si type='numpy' (no es nuestro caso,
                          pero lo manejamos por robustez).
        """
        if imagen_entrada is None:
            return 'Selecciona o arrastra una imagen.', None

        if isinstance(imagen_entrada, dict):
            ruta = imagen_entrada.get('path') or imagen_entrada.get('name')
            if not ruta:
                return 'No se pudo obtener la ruta de la imagen subida.', None
            img_bgr = cv2.imread(ruta)
        elif isinstance(imagen_entrada, str):
            img_bgr = cv2.imread(imagen_entrada)
        elif isinstance(imagen_entrada, np.ndarray):
            img_bgr = cv2.cvtColor(imagen_entrada, cv2.COLOR_RGB2BGR)
        else:
            return f'Tipo de entrada inesperado: {type(imagen_entrada)}', None

        if img_bgr is None:
            return 'No se pudo decodificar la imagen.', None

        try:
            resultado_str, img_anotada, _ = _procesar_imagen_modelo(img_bgr)
            return resultado_str, cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)
        except Exception as e:
            traceback.print_exc()
            return f'Error procesando imagen: {e}', None

    with gr.Blocks(title='Detector Tic-Tac-Toe + ESP32') as demo:
        gr.Markdown('## Detector Tic-Tac-Toe')
        gr.Markdown('Arrastra/suelta una imagen o selecciónala, y pulsa **Detectar**.')

        entrada      = gr.Image(type='filepath', label='Imagen de entrada')
        boton        = gr.Button('Detectar', variant='primary')
        salida_texto = gr.Textbox(label='Resultado tablero')
        salida_img   = gr.Image(label='Imagen anotada')

        boton.click(
            fn=ejecutar_deteccion_web,
            inputs=[entrada],
            outputs=[salida_texto, salida_img],
        )

    return demo


def lanzar_interfaz_web(host: str = DEFAULT_HOST, port: int = DEFAULT_GRADIO_PORT):
    """Modo 'web': solo Gradio, bloquea el proceso principal."""
    if not TIENE_GRADIO:
        raise RuntimeError('Gradio no está instalado. Instala con: pip install gradio')

    _obtener_modelo()
    demo = _construir_app_gradio()
    print(f'Interfaz web disponible en http://{host}:{port}')
    demo.launch(server_name=host, server_port=port, inbrowser=True, share=False)


def lanzar_modo_full(host: str = DEFAULT_HOST,
                     flask_port: int = DEFAULT_PORT,
                     gradio_port: int = DEFAULT_GRADIO_PORT):
    """
    Modo 'full': arranca Flask y Gradio en el mismo proceso.
    - Flask  corre en el hilo principal (bloqueante).
    - Gradio corre en un hilo daemon: se cierra solo cuando Flask termina.
    """
    if not TIENE_GRADIO:
        raise RuntimeError('Gradio no está instalado. Instala con: pip install gradio')

    # Carga el modelo una sola vez antes de lanzar ningún servidor.
    _obtener_modelo()

    # --- Hilo Gradio (daemon) ---
    demo = _construir_app_gradio()

    def _hilo_gradio():
        # prevent_thread_lock=True es imprescindible: sin él demo.launch()
        # bloquearía este hilo indefinidamente con su propio bucle de eventos.
        demo.launch(
            server_name=host,
            server_port=gradio_port,
            inbrowser=False,   # No abrir navegador automáticamente en modo full
            share=False,
            prevent_thread_lock=True,
        )

    hilo = threading.Thread(target=_hilo_gradio, daemon=True, name='gradio-server')
    hilo.start()

    print('=' * 45)
    print(f'  Flask  (ESP32-CAM) → http://{host}:{flask_port}')
    print(f'    /procesar   POST  → análisis de imagen')
    print(f'    /movimiento POST  → siguiente jugada IA')
    print(f'    /health     GET   → estado del servidor')
    print(f'  Gradio (web UI)    → http://{host}:{gradio_port}')
    print('=' * 45)

    # --- Flask en el hilo principal (bloqueante) ---
    app.run(host=host, port=flask_port)


# ==========================================
# SERVIDOR FLASK
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
        print(f'Error en /movimiento: {e}')
        traceback.print_exc()
        return jsonify({'error': 'Error Server'}), 500


def lanzar_servidor_flask(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Modo 'server': solo Flask, bloquea el proceso principal."""
    _obtener_modelo()
    print('=' * 45)
    print(f'  Flask  (ESP32-CAM) → http://{host}:{port}')
    print(f'    /procesar   POST  → análisis de imagen')
    print(f'    /movimiento POST  → siguiente jugada IA')
    print(f'    /health     GET   → estado del servidor')
    print('=' * 45)
    app.run(host=host, port=port)


# ==========================================
# PUNTO DE ENTRADA
# ==========================================

def _modo_imagen(ruta_img: str):
    resultado, img_anotada = probar_modelo_optimizado(ruta_img)
    salida = 'modelDetector_esp32_resultado.jpg'
    cv2.imwrite(salida, img_anotada)
    print(f'Resultado: {resultado}')
    print(f'Imagen anotada guardada en: {os.path.abspath(salida)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detector Tic-Tac-Toe con Flask y comunicación para ESP32-CAM.'
    )
    parser.add_argument(
        '--modo',
        choices=['server', 'web', 'full', 'imagen'],
        default='server',
        help=(
            'server  → solo Flask (ESP32-CAM);  '
            'web     → solo Gradio (interfaz visual);  '
            'full    → Flask + Gradio simultáneamente;  '
            'imagen  → procesa un fichero local y sale'
        ),
    )
    parser.add_argument('--host',        default=DEFAULT_HOST)
    parser.add_argument('--port',        type=int, default=DEFAULT_PORT,
                        help='Puerto Flask (default: 5000)')
    parser.add_argument('--gradio-port', type=int, default=DEFAULT_GRADIO_PORT,
                        dest='gradio_port',
                        help='Puerto Gradio (default: 7860)')
    parser.add_argument('--imagen',      default=RUTA_IMAGEN)
    args = parser.parse_args()

    if args.modo == 'web':
        lanzar_interfaz_web(host=args.host, port=args.gradio_port)
    elif args.modo == 'full':
        lanzar_modo_full(host=args.host,
                         flask_port=args.port,
                         gradio_port=args.gradio_port)
    elif args.modo == 'imagen':
        _modo_imagen(args.imagen)
    else:
        lanzar_servidor_flask(host=args.host, port=args.port)