import cv2
import numpy as np
from ultralytics import YOLO
import os

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
NOMBRE_TABLERO = 'board'

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

EXPANSION_TABLERO_BASE = 0.06
EXPANSION_TABLERO_MAX = 0.24
CONF_MIN_TABLERO = 0.35
# ==========================================

def _ordenar_puntos(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def _box_a_poligono(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def _expandir_poligono_tablero(src, n_celdas, board_box=None):
    centro = np.mean(src, axis=0)
    faltantes = max(0, 9 - n_celdas)
    extra = EXPANSION_TABLERO_BASE + 0.02 * faltantes
    extra = float(np.clip(extra, EXPANSION_TABLERO_BASE, EXPANSION_TABLERO_MAX))
    src_expand = centro + (src - centro) * (1.0 + extra)

    if board_box is not None:
        board_poly = _box_a_poligono(board_box)
        board_centro = np.mean(board_poly, axis=0)
        board_expand = board_centro + (board_poly - board_centro) * 1.02

        merged = np.vstack([src_expand, board_expand]).astype(np.float32)
        hull = cv2.convexHull(merged).reshape(-1, 2)

        if len(hull) >= 4:
            rect = cv2.minAreaRect(hull)
            src_expand = _ordenar_puntos(cv2.boxPoints(rect))

    return src_expand

def _estimar_homografia_desde_celdas(celdas_centros, celdas_boxes, board_box=None):
    if len(celdas_boxes) >= 2:
        pts = []
        for x1, y1, x2, y2 in celdas_boxes:
            pts.extend([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
        pts = np.array(pts, dtype=np.float32)

        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        src_extremos = np.array([
            pts[np.argmin(s)],  # top-left
            pts[np.argmax(d)],  # top-right
            pts[np.argmax(s)],  # bottom-right
            pts[np.argmin(d)],  # bottom-left
        ], dtype=np.float32)
        src = _ordenar_puntos(src_extremos)

        # Validación mínima para evitar homografías degeneradas
        area_src = abs(cv2.contourArea(src.astype(np.float32)))
        if area_src < 100:
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            src = _ordenar_puntos(box)
    elif len(celdas_centros) >= 4:
        pts = np.array(celdas_centros, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        src = _ordenar_puntos(box)
    else:
        return None, None

    src = _expandir_poligono_tablero(src, len(celdas_centros), board_box=board_box)

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
    imagenes = [("original", img)]

    imagen_blur = blur_score < UMBRAL_BLUR_LAPLACIAN
    imagen_palida = sat_media < UMBRAL_SATURACION_MEDIA

    if imagen_palida:
        imagenes.append(("clahe", _aplicar_clahe(img)))
        imagenes.append(("realce_color", _aplicar_realce_color(img)))

    if imagen_blur:
        imagenes.append(("nitidez", _aplicar_nitidez(img)))

    if imagen_palida and imagen_blur:
        combinada = _aplicar_nitidez(_aplicar_realce_color(_aplicar_clahe(img)))
        imagenes.append(("clahe_color_nitidez", combinada))

    mejor = None
    mejor_score = -1.0

    for nombre, img_variante in imagenes:
        resultados = model(img_variante, conf=0.25, iou=0.5, imgsz=640)[0]
        score = _score_resultado(resultados, id_roja, id_azul, id_celda)
        if score > mejor_score:
            mejor_score = score
            mejor = (nombre, resultados, blur_score, sat_media)

    return mejor

def probar_modelo_optimizado(ruta_img):
    print("Cargando modelo...")
    model = YOLO(RUTA_MODELO)
    
    img = cv2.imread(ruta_img)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_img}")

    print("Analizando imagen...")

    # Leemos nombres de clase desde metadatos del modelo
    clases_modelo = model.names if hasattr(model, 'names') else {}
    id_roja = next((k for k, v in clases_modelo.items() if v == NOMBRE_ROJA), -1)
    id_azul = next((k for k, v in clases_modelo.items() if v == NOMBRE_AZUL), -1)
    id_celda = next((k for k, v in clases_modelo.items() if v == NOMBRE_CELDA), -1)
    id_tablero = next((k for k, v in clases_modelo.items() if v == NOMBRE_TABLERO), -1)

    # conf=0.25 para no perder esquinas, iou=0.5 para reducir duplicados desde el modelo
    variante_usada, resultados, blur_score, sat_media = _inferir_mejor_resultado(
        model, img, id_roja, id_azul, id_celda
    )

    conf_min_azul_actual = CONF_MIN_AZUL_PALIDA if sat_media < UMBRAL_SATURACION_MEDIA else CONF_MIN_AZUL

    clases_modelo = resultados.names

    celdas_centros = []
    celdas_boxes = []
    detecciones_fichas = []
    board_box = None
    board_conf = -1.0

    # 1. Extraer coordenadas de todas las detecciones
    for box in resultados.boxes:
        clase = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if clase == id_celda:
            celdas_centros.append([cx, cy])
            celdas_boxes.append([x1, y1, x2, y2])
        elif clase == id_tablero and conf > board_conf and conf >= CONF_MIN_TABLERO:
            board_conf = conf
            board_box = [x1, y1, x2, y2]
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

    # 2. FILTRO ANTI-FANTASMAS (NMS Manual)
    # Evita que si YOLO detecta dos fichas en el mismo sitio, se cuenten ambas
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
            if dist < dist_duplicado: # Si están muy cerca, es la misma ficha
                duplicada = True
                break
        if not duplicada:
            fichas_finales.append(f)

    # 3. LÓGICA DE ASIGNACIÓN POR REJILLA (homografía si hay inclinación)
    matriz = np.zeros((3, 3), dtype=int)
    H, tablero_poly = _estimar_homografia_desde_celdas(celdas_centros, celdas_boxes, board_box=board_box)
    cols_anchor, filas_anchor = _estimar_anclas_rejilla(H, celdas_centros)
    modo_zona = "homografia" if H is not None else "fallback"
    if H is not None and cols_anchor is not None and filas_anchor is not None:
        modo_zona = "homografia_anclada"
    
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

            # Zona interior estricta: evita coger fichas tocando solo borde
                margen_interior = MARGEN_INTERIOR_NORM if len(celdas_centros) >= 7 else 0.015
                if not (margen_interior <= u <= 1 - margen_interior and
                    margen_interior <= v <= 1 - margen_interior):
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

                # Penaliza detecciones muy pegadas a líneas de celda
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

        if len(celdas_centros) >= 7:
            shrink = SHRINK_FALLBACK * lado_ref
        elif len(celdas_centros) >= 5:
            shrink = 0.04 * lado_ref
        else:
            shrink = -0.08 * lado_ref

        min_x_val = min_x + shrink
        min_y_val = min_y + shrink
        max_x_val = max_x - shrink
        max_y_val = max_y - shrink

        if board_box is not None:
            bx1, by1, bx2, by2 = board_box
            min_x_val = min(min_x_val, bx1)
            min_y_val = min(min_y_val, by1)
            max_x_val = max(max_x_val, bx2)
            max_y_val = max(max_y_val, by2)

        if max_x_val <= min_x_val or max_y_val <= min_y_val:
            resultado_str = "tablero={Error: Zona de juego invalida}"
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
        resultado_str = "tablero={Error: Pocas celdas detectadas}"

    # 4. FORMATEO Y SALIDA POR CONSOLA
    print("\n" + "="*30)
    print("MATRIZ DETECTADA:")
    print(resultado_str)
    print(
        f"Variante imagen: {variante_usada} | blur={blur_score:.1f} | sat_media={sat_media:.1f}"
    )
    print(
        f"Umbral roja={CONF_MIN_ROJA:.2f} | Umbral azul={conf_min_azul_actual:.2f}"
    )
    board_info = f" | board_conf={board_conf:.2f}" if board_box is not None else ""
    print(f"Modo zona: {modo_zona} | Celdas detectadas: {len(celdas_centros)} | Fichas filtradas: {len(fichas_finales)}{board_info}")
    print("="*30 + "\n")

    # 5. VISUALIZACIÓN (RESTAURADA Y MEJORADA)
    img_anotada = resultados.plot() # Dibuja las cajas de YOLO
    
    # Dibujar área de juego estimada para debug
    if tablero_poly is not None:
        poly = tablero_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_anotada, [poly], True, (0, 255, 0), 2)
        pt = tuple(tablero_poly[0].astype(int))
        cv2.putText(img_anotada, "Area Tablero (warp)", (pt[0], pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif len(celdas_centros) >= 2:
        cv2.rectangle(img_anotada, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
        cv2.putText(img_anotada, "Area Tablero", (int(min_x), int(min_y)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Reescalar si la imagen es muy grande
    alto, ancho = img_anotada.shape[:2]
    if alto > 800 or ancho > 800:
        escala = 800 / max(alto, ancho)
        img_anotada = cv2.resize(img_anotada, (0,0), fx=escala, fy=escala)

    return resultado_str, img_anotada

def lanzar_interfaz_web():
    if not TIENE_GRADIO:
        raise RuntimeError("Gradio no está instalado. Instala con: pip install gradio")

    def ejecutar_deteccion_web(ruta_img):
        if not ruta_img:
            return "Selecciona o arrastra una imagen.", None
        if not os.path.isfile(ruta_img):
            return f"Archivo no encontrado: {ruta_img}", None

        try:
            resultado, img_anotada = probar_modelo_optimizado(ruta_img)
            img_rgb = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)
            return resultado, img_rgb
        except Exception as e:
            return f"Error procesando imagen: {e}", None

    with gr.Blocks(title="Detector Tic-Tac-Toe") as demo:
        gr.Markdown("## Detector Tic-Tac-Toe")
        gr.Markdown("Arrastra/suelta una imagen o selecciónala, y pulsa **Detectar**.")

        entrada = gr.Image(type="filepath", label="Imagen de entrada")
        boton = gr.Button("Detectar", variant="primary")
        salida_texto = gr.Textbox(label="Resultado tablero")
        salida_img = gr.Image(label="Imagen anotada")

        boton.click(
            fn=ejecutar_deteccion_web,
            inputs=[entrada],
            outputs=[salida_texto, salida_img]
        )

    demo.launch(inbrowser=True, share=False)

if __name__ == "__main__":
    lanzar_interfaz_web()