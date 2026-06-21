"""
Prepara el dataset de Roboflow y lanza el entrenamiento con ultralytics.

Uso:
    python preparar_entrenamiento.py
"""

import zipfile
import pathlib
import shutil
import sys
import yaml

DATASET_ZIP  = pathlib.Path('misovos.yolo26.zip')
DATASET_DIR  = pathlib.Path('dataset_ttt')

# ── Configuración de entrenamiento ──────────────────────────────
# Sin GPU: cambia a 'yolov8n.pt' y EPOCHS=50 para una prueba rápida en CPU (~2-3h)
MODEL_BASE   = 'yolo26s.pt'
EPOCHS       = 300
IMG_SIZE     = 640
WORKERS      = 4        # en Windows >4 da problemas con multiprocessing
PATIENCE     = 150
BATCH        = -1       # auto-batch según VRAM disponible
LR0          = 0.001
COS_LR       = True
CACHE        = True
HALF         = True     # FP16 en GPU RTX → ~2× más rápido con Tensor Cores

# Aumentaciones — replicando la config exacta de Roboflow:
#   Las 90° rotations ya las aplica YOLO con degrees=90 discreto vía el dataset x2 de
#   Roboflow, pero como el ZIP solo trae los 494 originales dejamos degrees=15 para
#   la rotación continua ±15° y añadimos rotate90 vía fliplr+flipud combinados.
#   Roboflow: flip H, 90°rotate, ±15°rotate, shear±10°, brightness±15%,
#             noise≤1%, saturation±25%, hue±15°, camera gain σ=0.05
AUG = dict(
    fliplr       = 0.5,    # flip horizontal (Roboflow: Flip Horizontal)
    flipud       = 0.0,    # sin flip vertical
    degrees      = 90.0,   # ±90° cubre las rotaciones discretas de 90° + los ±15° de Roboflow
    shear        = 10.0,   # ±10° H y V (Roboflow: Shear ±10°)
    hsv_h        = 0.042,  # hue ±15° → 15/360 ≈ 0.042 (Roboflow: Hue ±15°)
    hsv_s        = 0.25,   # saturación ±25% (Roboflow: Saturation ±25%)
    hsv_v        = 0.15,   # brillo ±15% + camera gain σ=0.05 (Roboflow: Brightness ±15%)
    perspective  = 0.0005, # distorsión de cámara mínima
    mosaic       = 1.0,    # mosaico 4 imágenes (útil con dataset pequeño de 494 imgs)
    erasing      = 0.0,    # Roboflow usa noise ≤1% de pixel, no random erasing
)


def extraer_dataset():
    if not DATASET_ZIP.exists():
        print(f'[!] No se encuentra {DATASET_ZIP}')
        print('    Asegúrate de que el archivo descargado de Roboflow está en esta carpeta.')
        sys.exit(1)

    if not zipfile.is_zipfile(DATASET_ZIP):
        print('[!] El archivo no es un ZIP válido.')
        sys.exit(1)

    # Mostrar estructura interna antes de extraer
    with zipfile.ZipFile(DATASET_ZIP) as z:
        nombres = z.namelist()
        yamls = [n for n in nombres if n.endswith('.yaml') or n.endswith('.yml')]
        print(f'ZIP contiene {len(nombres)} archivos, {len(yamls)} yamls:')
        for y in yamls:
            print(f'  {y}')
        print()

        if DATASET_DIR.exists():
            print(f'Dataset ya extraído en {DATASET_DIR}/\n')
            # Mostrar yamls actuales para diagnóstico
            for y in sorted(DATASET_DIR.rglob('*.yaml')):
                print(f'  yaml existente: {y}')
            print()
            return

        # Extraer preservando estructura de carpetas (NO aplana como Windows Explorer)
        print(f'Extrayendo {DATASET_ZIP} → {DATASET_DIR}/')
        DATASET_DIR.mkdir(exist_ok=True)
        for member in z.infolist():
            # Saltar entradas que salgan del directorio destino (seguridad)
            dest = DATASET_DIR / member.filename
            if not str(dest.resolve()).startswith(str(DATASET_DIR.resolve())):
                continue
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, 'wb') as dst:
                    dst.write(src.read())

    print('Extracción completada.\n')


# ── Arreglar dataset ────────────────────────────────────────────

def arreglar_yaml() -> pathlib.Path:
    """Encuentra el yaml maestro y corrige sus rutas si apuntan a sitios que no existen."""
    candidates = sorted(DATASET_DIR.rglob('data.yaml')) or sorted(DATASET_DIR.rglob('*.yaml'))
    print(f'Yamls encontrados ({len(candidates)}):')
    for c in candidates:
        print(f'  {c}')

    master = None
    for c in candidates:
        try:
            with open(c, encoding='utf-8') as f:
                d = yaml.safe_load(f)
            if 'names' in d and any(k in d for k in ('train', 'val', 'valid')):
                master = c
                break
        except Exception:
            pass
    if master is None:
        master = min(candidates, key=lambda p: len(p.parts))
    print(f'Usando: {master}\n')

    with open(master, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    changed = False
    for key in ('train', 'val', 'valid', 'test'):
        if key not in data:
            continue
        raw = data[key]

        # Comprobar si la ruta resuelta relativa al yaml existe
        candidate = (master.parent / raw).resolve()
        if candidate.exists():
            # Existe pero puede estar duplicada — guardamos como ruta relativa limpia
            rel = candidate.relative_to(master.parent.resolve())
            clean = str(rel).replace('\\', '/')
            if clean != raw:
                data[key] = clean
                changed = True
                print(f'  Ruta simplificada [{key}]: {raw} → {clean}')
            continue

        # No existe — buscar la carpeta de imágenes por split dentro del dataset
        split = 'valid' if key == 'val' else key
        found = [f for f in DATASET_DIR.rglob('images')
                 if f.is_dir() and split in str(f)
                 and (list(f.glob('*.jpg')) or list(f.glob('*.png')))]
        best = found[0] if found else None

        if best:
            # Guardar como ruta relativa respecto al yaml
            try:
                rel = best.resolve().relative_to(master.parent.resolve())
                data[key] = str(rel).replace('\\', '/')
            except ValueError:
                data[key] = str(best.resolve())
            changed = True
            print(f'  Ruta corregida [{key}]: {raw} → {data[key]}')
        else:
            print(f'  [!] No se encontró carpeta de imágenes para [{key}]')

    if changed:
        with open(master, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        print('data.yaml actualizado.\n')
    else:
        print('Rutas del data.yaml OK.\n')

    return master


def arreglar_labels():
    """Recorre todos los .txt de labels y clipea coordenadas fuera de [0,1]."""
    total_archivos = 0
    total_corregidos = 0
    for lbl in DATASET_DIR.rglob('labels/**/*.txt'):
        lines = lbl.read_text(encoding='utf-8').strip().splitlines()
        nuevas = []
        corregido = False
        for linea in lines:
            partes = linea.strip().split()
            if len(partes) != 5:
                continue
            try:
                cls = int(partes[0])
                cx, cy, w, h = map(float, partes[1:])
            except ValueError:
                continue
            if any(v < 0 or v > 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.001, min(1.0, w))
                h  = max(0.001, min(1.0, h))
                corregido = True
            nuevas.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        if corregido:
            lbl.write_text('\n'.join(nuevas) + '\n', encoding='utf-8')
            total_corregidos += 1
        total_archivos += 1
    print(f'Labels revisados: {total_archivos} archivos, {total_corregidos} corregidos.\n')


def verificar_estructura(yaml_path: pathlib.Path):
    print('── Contenido del data.yaml ──')
    print(yaml_path.read_text(encoding='utf-8'))
    print('────────────────────────────\n')


def contar_imagenes():
    for split in ('train', 'valid', 'test'):
        imgs = list((DATASET_DIR / split / 'images').glob('*.jpg')) + \
               list((DATASET_DIR / split / 'images').glob('*.png'))
        if imgs:
            print(f'  {split}: {len(imgs)} imágenes')


def entrenar(yaml_path: pathlib.Path, device: int = 0):
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[!] ultralytics no instalado. Ejecuta: pip install ultralytics')
        sys.exit(1)

    model = YOLO(MODEL_BASE)

    print(f'\n── Iniciando entrenamiento ──')
    print(f'   Modelo    : {MODEL_BASE}')
    print(f'   Épocas    : {EPOCHS}  (patience={PATIENCE})')
    print(f'   Imagen    : {IMG_SIZE}px  batch={BATCH}')
    print(f'   Device    : cuda:{device}  workers={WORKERS}')
    print(f'   FP16      : {HALF}  (Tensor Cores activos)')
    print(f'   LR        : {LR0}  cos_lr={COS_LR}')
    print(f'   Dataset   : {yaml_path}\n')

    model.train(
        data     = str(yaml_path),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        device   = device,
        workers  = WORKERS,
        patience = PATIENCE,
        batch    = BATCH,
        lr0      = LR0,
        cos_lr   = COS_LR,
        cache    = CACHE,
        half     = HALF,
        rect     = False,   # imágenes cuadradas 640×640 (igual que stretch de Roboflow)
        save     = True,
        project  = 'runs_ttt',
        name     = 'train',
        **AUG,
    )

    best = pathlib.Path('runs_ttt/train/weights/best.pt')
    if best.exists():
        shutil.copy(best, 'best_local.pt')
        print(f'\n══ Entrenamiento completado ══')
        print(f'   Modelo guardado: best_local.pt')
        print(f'\n   Para activarlo en vision_esp32.py:')
        print(f'     INFERENCE_MODE   = "yolo"')
        print(f'     LOCAL_MODEL_PATH = "best_local.pt"')
    else:
        print('\n[!] No se encontró best.pt tras el entrenamiento.')


def verificar_gpu():
    """Comprueba que CUDA esté disponible y muestra info del dispositivo."""
    try:
        import torch
    except ImportError:
        print('[!] PyTorch no está instalado. Ejecuta:')
        print('    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
        sys.exit(1)

    if not torch.cuda.is_available():
        print('[!] CUDA no disponible. PyTorch detectó solo CPU.')
        print('    Posible causa: tienes la versión CPU-only de PyTorch.')
        print()
        print('    Tu driver soporta CUDA 13.2 — los wheels de PyTorch van por')
        print('    CUDA 12.x, pero el driver es retrocompatible y funciona igual.')
        print()
        print('    Solución — reinstala PyTorch con CUDA 12.4:')
        print('    pip uninstall torch torchvision torchaudio -y')
        print('    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124')
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'   GPU detectada : {gpu}')
    print(f'   VRAM          : {vram:.1f} GB')
    print(f'   CUDA          : {torch.version.cuda}')
    print(f'   PyTorch       : {torch.__version__}\n')
    return 0  # device index


def main():
    print('=== Preparación y entrenamiento ===\n')

    print('── 0. Verificando GPU ──')
    device = verificar_gpu()

    print('── 1. Extrayendo dataset ──')
    extraer_dataset()

    print('── 2. Arreglando data.yaml ──')
    yaml_path = arreglar_yaml()

    print('── 3. Reparando labels ──')
    arreglar_labels()

    print('── 4. Verificando estructura ──')
    verificar_estructura(yaml_path)
    contar_imagenes()

    print('── 5. Entrenando ──')
    entrenar(yaml_path, device)


if __name__ == '__main__':
    main()
