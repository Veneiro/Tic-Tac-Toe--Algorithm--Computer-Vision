"""
Descarga el modelo Roboflow en formato TFjs y lo convierte a ONNX.

El modelo 'misovos/7' fue entrenado con el framework yolo26 de Roboflow,
que genera modelos TensorFlow.js (no PyTorch). Este script descarga los
pesos TFjs y los convierte a ONNX para usarlos con ultralytics/onnxruntime.

Requisitos:
    pip install tensorflowjs onnx tf2onnx tensorflow

Uso:
    python descargar_modelo.py
"""

import json
import pathlib
import zipfile
import requests

API_KEY   = 'C6jJ9kSIF8VRiJWR3rFE'
WORKSPACE = 'ttt-vofgg'
PROJECT   = 'misovos'
VERSION   = 7
BASE_API  = 'https://api.roboflow.com'
BASE_SL   = 'https://serverless.roboflow.com'

DEST_DIR  = pathlib.Path('modelo_tfjs')
DEST_ONNX = pathlib.Path('modelo_roboflow.onnx')


def download_stream(url, dest: pathlib.Path, headers=None):
    with requests.get(url, stream=True, timeout=120, headers=headers or {}) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        done  = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {done / total * 100:.1f}%  ({done // 1024} KB)', end='', flush=True)
    print()


def try_get_json(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def main():
    print(f'=== Descarga modelo Roboflow (TFjs → ONNX) ===\n')

    # ── 1. Intentar endpoints de descarga de pesos ─────────────
    print('── Buscando endpoint de descarga de pesos...')
    candidates = [
        # Endpoint serverless con API key — a veces expone los archivos del modelo
        f'{BASE_SL}/{PROJECT}/{VERSION}?api_key={API_KEY}',
        # Intentar endpoint de TFjs directo
        f'{BASE_SL}/{PROJECT}/{VERSION}/tfjs?api_key={API_KEY}',
        # Endpoint de OAK (OpenCV AI Kit)
        f'{BASE_SL}/{PROJECT}/{VERSION}/oak?api_key={API_KEY}',
        # Endpoint alternativo en api.roboflow.com
        f'{BASE_API}/{WORKSPACE}/{PROJECT}/{VERSION}/model?api_key={API_KEY}',
    ]

    weights_zip_url = None
    for url in candidates:
        data = try_get_json(url)
        if data:
            print(f'  OK: {url}')
            # Buscar cualquier URL de descarga en la respuesta
            text = json.dumps(data)
            for line in text.split('"'):
                if line.startswith('https://') and any(
                    x in line.lower() for x in ('model', 'weight', '.zip', '.bin', '.json', 'tfjs')
                ):
                    print(f'       → candidata: {line}')
                    if not weights_zip_url:
                        weights_zip_url = line
            print(f'  Respuesta: {json.dumps(data, indent=2)[:400]}\n')
        else:
            print(f'  No disponible: {url}')

    # ── 2. Si no se encontró nada descargable, mostrar instrucciones ─
    if not weights_zip_url:
        print('\n[!] Roboflow no expone los pesos del modelo como archivo descargable.')
        print('    El modelo yolo26 es propietario de Roboflow y solo está')
        print('    disponible a través de su API de inferencia.\n')
        print('══ OPCIONES ══════════════════════════════════════════════')
        print()
        print('1. USAR LA API REST (ya funciona, sin instalar nada):')
        print('   En vision_esp32.py → INFERENCE_MODE = "roboflow"')
        print()
        print('2. REENTRENAR CON ULTRALYTICS (crea un .pt estándar):')
        print('   Tienes el dataset descargado anteriormente (best(roboflow).pt')
        print('   es en realidad un ZIP con las imágenes y etiquetas).')
        print()
        print('   Pasos:')
        print('   a) Renombra best(roboflow).pt → dataset.zip y descomprímelo')
        print('   b) pip install ultralytics')
        print('   c) yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640')
        print()
        print('   Con 94% mAP en Roboflow, reentrenando con esos datos obtendrás')
        print('   resultados similares o mejores en un .pt estándar.')
        print()
        print('3. Descarga manual desde la web:')
        print('   app.roboflow.com → tu proyecto → Versions → Deploy')
        print('   → "Download ZIP" y busca si hay opción "PyTorch" o "ONNX"')
        return

    # ── 3. Descargar el modelo ─────────────────────────────────
    print(f'\n── Descargando modelo desde: {weights_zip_url}')
    tmp = pathlib.Path('modelo_tmp.zip')
    download_stream(weights_zip_url, tmp)

    # Descomprimir si es ZIP
    if zipfile.is_zipfile(tmp):
        DEST_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(DEST_DIR)
        tmp.unlink()
        print(f'Modelo extraído en: {DEST_DIR}/')
        print('Archivos:')
        for f in sorted(DEST_DIR.rglob('*')):
            if f.is_file():
                print(f'  {f}')
    else:
        tmp.rename('modelo_descargado.bin')
        print('Archivo descargado como modelo_descargado.bin')

    # ── 4. Intentar conversión a ONNX ─────────────────────────
    model_json = next(DEST_DIR.rglob('model.json'), None) if DEST_DIR.exists() else None
    if model_json:
        print(f'\n── Intentando conversión TFjs → ONNX...')
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, '-m', 'tf2onnx.convert',
                 '--tfjs', str(model_json),
                 '--output', str(DEST_ONNX)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f'Conversión OK → {DEST_ONNX}')
                print('Para usar con ultralytics:')
                print(f'  from ultralytics import YOLO')
                print(f'  model = YOLO("{DEST_ONNX}")')
            else:
                print(f'Error en conversión:\n{result.stderr}')
                print('\nInstala con: pip install tf2onnx tensorflow tensorflowjs')
        except FileNotFoundError:
            print('tf2onnx no instalado. Ejecuta:')
            print('  pip install tf2onnx tensorflow tensorflowjs')
            print(f'  python -m tf2onnx.convert --tfjs {model_json} --output {DEST_ONNX}')


if __name__ == '__main__':
    main()
