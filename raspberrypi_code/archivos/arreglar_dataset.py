"""
Diagnostica y repara el dataset exportado de Roboflow.

Problemas que detecta y corrige:
  1. Rutas en data.yaml que no existen (problema más común con exportaciones de Roboflow)
  2. Labels con coordenadas fuera del rango [0, 1] (bounding boxes que ultralytics ignora)
  3. Labels vacíos o corruptos
  4. Imágenes sin label correspondiente

Uso:
    python arreglar_dataset.py
"""

import pathlib
import yaml
import shutil

DATASET_DIR = pathlib.Path('dataset_ttt')


# ── Helpers ────────────────────────────────────────────────────

def cargar_yaml(path: pathlib.Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def guardar_yaml(data: dict, path: pathlib.Path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def encontrar_yaml() -> pathlib.Path:
    candidates = sorted(DATASET_DIR.rglob('data.yaml'))
    if not candidates:
        candidates = sorted(DATASET_DIR.rglob('*.yaml'))
    if not candidates:
        raise FileNotFoundError(f'No se encontró data.yaml en {DATASET_DIR}')

    print(f'data.yaml encontrados ({len(candidates)}):')
    for c in candidates:
        print(f'  {c}')

    # El correcto es el que tiene 'names' y al menos train o val
    for c in candidates:
        try:
            data = cargar_yaml(c)
            if 'names' in data and ('train' in data or 'val' in data or 'valid' in data):
                print(f'\nUsando: {c}\n')
                return c
        except Exception:
            pass

    # Si ninguno tiene names+splits, usar el de nivel más alto (menos profundo)
    best = min(candidates, key=lambda p: len(p.parts))
    print(f'\nUsando (por nivel): {best}\n')
    return best


# ── 1. Arreglar rutas en data.yaml ─────────────────────────────

def arreglar_rutas(yaml_path: pathlib.Path) -> pathlib.Path:
    data    = cargar_yaml(yaml_path)
    base    = yaml_path.parent
    changed = False

    print('── data.yaml original ──')
    for key in ('train', 'val', 'valid', 'test'):
        if key in data:
            print(f'  {key}: {data[key]}')

    for key in ('train', 'val', 'valid', 'test'):
        if key not in data:
            continue
        raw = data[key]
        # Intentar la ruta tal cual (relativa al yaml)
        candidate = (base / raw).resolve()
        if candidate.exists():
            continue  # ya funciona

        # Buscar la carpeta por nombre dentro del dataset
        folder_name = pathlib.Path(raw).name  # "images", "train/images", etc.
        found = list(DATASET_DIR.rglob(folder_name))
        # Filtrar: tiene que contener imágenes
        found = [f for f in found if f.is_dir() and
                 (list(f.glob('*.jpg')) or list(f.glob('*.png')))]

        # Elegir la que tenga el split correcto en el path
        split = 'valid' if key == 'val' else key
        best  = next((f for f in found if split in str(f)), None) or (found[0] if found else None)

        if best:
            # Guardar como ruta absoluta para evitar ambigüedades
            data[key] = str(best)
            changed = True
            print(f'  ✓ {key}: {raw} → {best}')
        else:
            print(f'  ✗ {key}: no se encontró carpeta para "{raw}"')

    if changed:
        guardar_yaml(data, yaml_path)
        print(f'\ndata.yaml actualizado: {yaml_path}\n')
    else:
        print('\nRutas del data.yaml OK\n')

    return yaml_path


# ── 2. Validar y reparar labels ────────────────────────────────

def clip_coord(v: float) -> float:
    return max(0.0, min(1.0, v))


def validar_label(label_path: pathlib.Path, fix: bool = True) -> dict:
    lines        = label_path.read_text(encoding='utf-8').strip().splitlines()
    errores      = 0
    lineas_ok    = []
    lineas_malas = []

    for linea in lines:
        partes = linea.strip().split()
        if len(partes) != 5:
            lineas_malas.append(linea)
            errores += 1
            continue
        try:
            cls, cx, cy, w, h = int(partes[0]), float(partes[1]), float(partes[2]), \
                                 float(partes[3]), float(partes[4])
        except ValueError:
            lineas_malas.append(linea)
            errores += 1
            continue

        fuera = any(v < 0 or v > 1 for v in (cx, cy, w, h))
        invalido = w <= 0 or h <= 0

        if fuera or invalido:
            errores += 1
            if fix and not invalido:
                # Clip a [0,1]
                cx, cy = clip_coord(cx), clip_coord(cy)
                w  = min(w,  1 - cx) if cx + w / 2 > 1 else w
                h  = min(h,  1 - cy) if cy + h / 2 > 1 else h
                w, h = max(0.001, w), max(0.001, h)
                lineas_ok.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        else:
            lineas_ok.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

    if fix and errores and lineas_ok:
        label_path.write_text('\n'.join(lineas_ok) + '\n', encoding='utf-8')

    return {'errores': errores, 'ok': len(lineas_ok), 'malos': len(lineas_malas)}


def validar_split(images_dir: pathlib.Path):
    labels_dir = images_dir.parent.parent / 'labels' / images_dir.name
    if not labels_dir.exists():
        # Estructura alternativa: mismo nivel, carpeta labels junto a images
        labels_dir = images_dir.parent / 'labels'

    imagenes   = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    total_err  = 0
    sin_label  = 0

    for img in imagenes:
        lbl = labels_dir / (img.stem + '.txt')
        if not lbl.exists():
            sin_label += 1
            continue
        stats = validar_label(lbl, fix=True)
        total_err += stats['errores']

    return len(imagenes), total_err, sin_label


# ── Main ───────────────────────────────────────────────────────

def main():
    if not DATASET_DIR.exists():
        print(f'[!] No se encuentra {DATASET_DIR}/')
        print('    Ejecuta primero: python preparar_entrenamiento.py')
        return

    print('=== Diagnóstico y reparación del dataset ===\n')

    # 1. Arreglar rutas
    yaml_path = encontrar_yaml()
    print(f'data.yaml: {yaml_path}\n')
    yaml_path = arreglar_rutas(yaml_path)

    # 2. Validar labels por split
    data = cargar_yaml(yaml_path)
    print('── Validando labels ──')
    for key in ('train', 'val', 'valid', 'test'):
        raw = data.get(key)
        if not raw:
            continue
        images_dir = pathlib.Path(raw)
        if not images_dir.exists():
            print(f'  {key}: carpeta no encontrada ({images_dir})')
            continue
        n_imgs, n_err, sin_lbl = validar_split(images_dir)
        estado = '✓' if n_err == 0 and sin_lbl == 0 else '⚠'
        print(f'  {estado} {key}: {n_imgs} imágenes | {n_err} coords corregidas | {sin_lbl} sin label')

    # 3. Mostrar clases
    nombres = data.get('names', [])
    print(f'\nClases ({data.get("nc", "?")}): {nombres}')
    print('\nClases que espera vision_esp32.py:')
    print('  ["blue circle", "board", "cells", "grid", "red cross"]')

    esperadas = ['blue circle', 'board', 'cells', 'grid', 'red cross']
    if sorted(nombres) == sorted(esperadas):
        print('  ✓ Coinciden perfectamente')
    else:
        print('  ⚠ Diferencia detectada — revisa que los nombres coincidan')
        faltan  = set(esperadas) - set(nombres)
        sobran  = set(nombres) - set(esperadas)
        if faltan:  print(f'    Faltan en dataset  : {faltan}')
        if sobran:  print(f'    Sobran en dataset  : {sobran}')

    print('\n── Dataset listo para entrenar ──')
    print(f'Ejecuta: python preparar_entrenamiento.py')


if __name__ == '__main__':
    main()
