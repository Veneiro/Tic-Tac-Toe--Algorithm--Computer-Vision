"""
capturador.py
-------------
Servidor Flask mínimo: recibe una imagen de la Raspberry, la guarda,
detecta el estado del tablero y guarda el resultado.  Nada más.

Directorio de salida: capturas/run_YYYYMMDD_HHMMSS_µs/
  original_image.png      <- imagen tal como llega
  estado_tablero.txt      <- estado detectado (formato tablero={...})

Endpoint:
  POST /procesar          <- cuerpo = bytes JPEG/PNG crudos (mismo que main.py)
    Respuesta 200: "tablero={0,0,0;0,0,0;0,0,0}" (o error)

Uso:
  python capturador.py
  python capturador.py --port 5001 --out capturas
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request

# Reutilizamos el pipeline CV completo de main.py
from main import (
    analyze_precapture,
    apply_adaptive_enhancement,
    warp_panel,
    find_grid_bbox,
    classify_cells,
    board_to_string,
)

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
OUTPUT_DIR = "capturas"   # sobreescribible con --out

app = Flask(__name__)

EXTREME_BLUR_MIN = 6.0


def _save_run(img: np.ndarray, estado: str) -> str:
    """Guarda imagen y estado en una carpeta nueva; devuelve la ruta."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    cv2.imwrite(os.path.join(run_dir, "original_image.png"), img)

    with open(os.path.join(run_dir, "estado_tablero.txt"), "w") as f:
        f.write(estado + "\n")

    return run_dir


# ──────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────
@app.route("/procesar", methods=["POST"])
def capturar():
    try:
        print("\n[capturador] ── Nueva petición recibida ──────────────────────")

        # 1. Decodificar imagen
        data = request.data
        print(f"[capturador] Bytes recibidos: {len(data)}")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print("[capturador] ERROR: no se pudo decodificar la imagen (¿formato incorrecto?)")
            return "Error: imagen no decodificable", 400
        print(f"[capturador] Imagen decodificada: {img.shape[1]}x{img.shape[0]} px")

        # 2. Filtro de desenfoque extremo (solo basura total)
        blur_var = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        print(f"[capturador] Nitidez (Laplacian var): {blur_var:.1f}")
        if blur_var < EXTREME_BLUR_MIN:
            print(f"[capturador] ✗ Imagen descartada por desenfoque EXTREMO (< {EXTREME_BLUR_MIN})")
            return f"Descartada: desenfoque extremo (var={blur_var:.1f})", 200

        # 3. Preprocesar
        print("[capturador] Analizando condiciones de captura...")
        capture_info = analyze_precapture(img)
        issues = capture_info.get("issues", [])
        print(f"[capturador] Precaptura ok={capture_info.get('ok')}  issues={issues if issues else 'ninguno'}")
        preprocessed = apply_adaptive_enhancement(img, capture_info)

        # 4. Detectar tablero
        print("[capturador] Buscando tablero...")
        warped, _ = warp_panel(preprocessed)
        if warped is None:
            print("[capturador] ✗ No se encontró el tablero — se guarda la imagen igualmente")
            estado = "tablero={Error: No Panel}"
        else:
            print(f"[capturador] ✓ Tablero encontrado ({warped.shape[1]}x{warped.shape[0]} px)")
            gx, gy, gw, gh = find_grid_bbox(warped)
            print(f"[capturador] Grid bbox: x={gx} y={gy} w={gw} h={gh}")
            board = classify_cells(warped, gx, gy, gw, gh)
            estado = board_to_string(board)
            print(f"[capturador] Celdas clasificadas: {estado}")

        # 5. Guardar
        run_dir = _save_run(img, estado)
        print(f"[capturador] ✓ Guardado en: {run_dir}")
        print(f"[capturador] ── Fin ─────────────────────────────────────────\n")
        return estado, 200

    except Exception as exc:
        import traceback
        print(f"[capturador] ERROR INESPERADO: {exc}")
        traceback.print_exc()
        return "Error Server", 500


# ──────────────────────────────────────────
# Arranque
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--out", default="capturas",
                        help="Directorio donde guardar las capturas")
    parser.add_argument("--extreme-blur-min", type=float, default=6.0,
                        help="Umbral mínimo de nitidez para descartar SOLO fotos extremadamente movidas")
    args = parser.parse_args()

    EXTREME_BLUR_MIN = float(args.extreme_blur_min)
    OUTPUT_DIR = args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Capturador arrancando en http://0.0.0.0:{args.port}")
    print(f"Guardando en: {os.path.abspath(OUTPUT_DIR)}")
    app.run(host="0.0.0.0", port=args.port)
