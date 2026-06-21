import argparse
import ast
import csv
import glob
import os
import re

import cv2
import numpy as np

from main import (
    analyze_precapture,
    apply_adaptive_enhancement,
    warp_panel,
    find_grid_bbox,
    infer_grid_edges,
)


LABEL_MAP = {None: 0, 'X': 1, 'O': 2}
CSV_LABEL_MAP = {'0': 0, '1': 1, '2': 2}


def parse_board(board_path):
    with open(board_path, 'r', encoding='utf-8') as file:
        raw = file.read().strip()
    board = ast.literal_eval(raw)
    if not isinstance(board, list) or len(board) != 3:
        raise ValueError(f'Tablero inválido en {board_path}')
    return board


def parse_estado_tablero(estado_path):
    with open(estado_path, 'r', encoding='utf-8') as file:
        raw = file.read().strip()

    tablero_match = re.search(r'tablero=\{[^\n\r]*\}', raw)
    if tablero_match:
        token = tablero_match.group(0)
        inside = token[len('tablero={'):-1]
        rows_raw = [part.strip() for part in inside.split(';') if part.strip()]
        if len(rows_raw) != 3:
            raise ValueError(f'Tablero inválido en {estado_path}')

        board_int = []
        for row_raw in rows_raw:
            cols = [c.strip() for c in row_raw.split(',')]
            if len(cols) != 3:
                raise ValueError(f'Tablero inválido en {estado_path}')
            row_int = []
            for value in cols:
                if value not in ('0', '1', '2'):
                    raise ValueError(f'Valor inválido "{value}" en {estado_path}')
                row_int.append(int(value))
            board_int.append(row_int)
        return board_int

    if raw.startswith('tablero={') and raw.endswith('}'):
        inside = raw[len('tablero={'):-1]
        rows_raw = [part.strip() for part in inside.split(';') if part.strip()]
        if len(rows_raw) != 3:
            raise ValueError(f'Tablero inválido en {estado_path}')

        board_int = []
        for row_raw in rows_raw:
            cols = [c.strip() for c in row_raw.split(',')]
            if len(cols) != 3:
                raise ValueError(f'Tablero inválido en {estado_path}')
            row_int = []
            for value in cols:
                if value not in ('0', '1', '2'):
                    raise ValueError(f'Valor inválido "{value}" en {estado_path}')
                row_int.append(int(value))
            board_int.append(row_int)
        return board_int

    board = ast.literal_eval(raw)
    if not isinstance(board, list) or len(board) != 3:
        raise ValueError(f'Tablero inválido en {estado_path}')

    board_int = [[0] * 3 for _ in range(3)]
    for row in range(3):
        if not isinstance(board[row], list) or len(board[row]) != 3:
            raise ValueError(f'Tablero inválido en {estado_path}')
        for col in range(3):
            value = board[row][col]
            if value in (0, 1, 2):
                board_int[row][col] = int(value)
            else:
                board_int[row][col] = LABEL_MAP.get(value, 0)
    return board_int


def load_manual_labels(labels_csv):
    labels_by_run = {}
    with open(labels_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        required = [f'c{row}{col}' for row in range(3) for col in range(3)]
        for field in ['run'] + required:
            if field not in reader.fieldnames:
                raise ValueError(f'Falta columna "{field}" en {labels_csv}')

        for row in reader:
            run_name = (row.get('run') or '').strip()
            if not run_name:
                continue

            board = [[None] * 3 for _ in range(3)]
            complete = True
            for r in range(3):
                for c in range(3):
                    value = (row.get(f'c{r}{c}') or '').strip()
                    if value not in CSV_LABEL_MAP:
                        complete = False
                        break
                    board[r][c] = int(value)
                if not complete:
                    break

            if complete:
                labels_by_run[run_name] = board

    return labels_by_run


def extract_features(image):
    """Extrae 625 features: gris(576) + H-hist(24) + S-hist(8) + V-hist(8) + color_explicito(9).
    Las features de color explícito son las más discriminativas para X (rojo) vs O (azul).
    """
    resized = cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # H con 24 bins (doble resolución) para mejor discriminación X(rojo) vs O(azul)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    h_hist = h_hist / max(np.sum(h_hist), 1e-6)
    s_hist = s_hist / max(np.sum(s_hist), 1e-6)
    v_hist = v_hist / max(np.sum(v_hist), 1e-6)

    # ── Features de color explícitas (muy discriminativas) ──────────────────────
    h_ch = hsv[:, :, 0].astype(np.float32)   # 0-180
    s_ch = hsv[:, :, 1].astype(np.float32)   # 0-255
    v_ch = hsv[:, :, 2].astype(np.float32)   # 0-255
    total = float(24 * 24)

    sat_mask = s_ch > 45  # píxeles con color significativo

    # Rojo: hue cerca de 0 O cerca de 180 (wraparound en HSV)
    red_mask  = ((h_ch < 18) | (h_ch > 162)) & sat_mask
    # Azul/cyan: hue 90-130
    blue_mask = (h_ch > 88) & (h_ch < 132) & sat_mask
    # Verde: hue 40-85 (puede aparecer en el tablero)
    green_mask = (h_ch > 38) & (h_ch < 86) & sat_mask

    red_ratio   = float(np.sum(red_mask))  / total
    blue_ratio  = float(np.sum(blue_mask)) / total
    green_ratio = float(np.sum(green_mask)) / total

    # Blanco/vacío: V alto, S bajo
    white_mask  = (v_ch > 175) & (s_ch < 55)
    white_ratio = float(np.sum(white_mask)) / total

    # Saturación media global (tokens tienen S alta, celdas vacías S baja)
    mean_sat = float(np.mean(s_ch)) / 255.0

    # ── Centro 8×8: el token siempre está centrado ───────────────────────────
    center = resized[8:16, 8:16]
    c_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    c_h = c_hsv[:, :, 0].astype(np.float32)
    c_s = c_hsv[:, :, 1].astype(np.float32)
    c_total  = float(8 * 8)
    c_sat    = c_s > 45
    c_red    = ((c_h < 18) | (c_h > 162)) & c_sat
    c_blue   = (c_h > 88) & (c_h < 132) & c_sat
    c_red_ratio  = float(np.sum(c_red))  / c_total
    c_blue_ratio = float(np.sum(c_blue)) / c_total
    c_mean_s     = float(np.mean(c_s)) / 255.0
    c_mean_v     = float(np.mean(c_hsv[:, :, 2])) / 255.0

    color_feats = np.array([
        red_ratio, blue_ratio, green_ratio, white_ratio, mean_sat,
        c_red_ratio, c_blue_ratio, c_mean_s, c_mean_v,
    ], dtype=np.float32)

    gray_flat = gray.flatten()
    return np.concatenate([gray_flat, h_hist, s_hist, v_hist, color_feats], axis=0)


def find_original_image(run_dir):
    preferred = [
        os.path.join(run_dir, 'original_image.png'),
        os.path.join(run_dir, '001_01_original_image.png'),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    candidates = sorted(glob.glob(os.path.join(run_dir, '*original_image.png')))
    return candidates[0] if candidates else None


def extract_patches_from_original(run_dir):
    image_path = find_original_image(run_dir)
    if not image_path:
        return []

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    capture_info = analyze_precapture(img)
    preprocessed = apply_adaptive_enhancement(img, capture_info)
    warped, _ = warp_panel(preprocessed)
    if warped is None:
        return []

    gx, gy, gw, gh = find_grid_bbox(warped)
    x_edges, y_edges, _ = infer_grid_edges(warped, gx, gy, gw, gh)

    cell_h = max(1, min(y_edges[1] - y_edges[0], y_edges[2] - y_edges[1], y_edges[3] - y_edges[2]))
    cell_w = max(1, min(x_edges[1] - x_edges[0], x_edges[2] - x_edges[1], x_edges[3] - x_edges[2]))
    margin_h = max(int(cell_h * 0.09), 3)
    margin_w = max(int(cell_w * 0.09), 3)

    side_h, side_w = warped.shape[:2]
    patches = []
    for row in range(3):
        for col in range(3):
            y1 = max(0, y_edges[row] + margin_h)
            y2 = min(side_h, y_edges[row + 1] - margin_h)
            x1 = max(0, x_edges[col] + margin_w)
            x2 = min(side_w, x_edges[col + 1] - margin_w)
            if y2 <= y1 or x2 <= x1:
                continue
            patch = warped[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            source = f'{image_path}#cell_{row}_{col}'
            patches.append((row, col, patch, source))

    return patches


def resolve_board_labels(run_dir):
    preferred = [
        os.path.join(run_dir, 'estado_real_tablero.txt'),
        os.path.join(run_dir, 'estado_tablero.txt'),
    ]
    for path in preferred:
        if os.path.exists(path):
            return parse_estado_tablero(path)

    board_files = glob.glob(os.path.join(run_dir, '*classification_board.txt'))
    if board_files:
        board = parse_board(board_files[0])
        board_int = [[0] * 3 for _ in range(3)]
        for row in range(3):
            for col in range(3):
                board_int[row][col] = LABEL_MAP.get(board[row][col], 0)
        return board_int

    return None


def build_dataset(debug_dir, out_npz, min_tokens=1, labels_csv=None):
    run_dirs = sorted(glob.glob(os.path.join(debug_dir, 'run_*')))
    if not run_dirs:
        raise RuntimeError(f'No hay runs en {debug_dir}')

    pattern = re.compile(r'cell_(\d)_(\d)_patch\.png$')
    labels_by_run = load_manual_labels(labels_csv) if labels_csv else None

    features = []
    labels = []
    sources = []

    used_runs = 0
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        if labels_by_run is not None:
            if run_name not in labels_by_run:
                continue
            board_int = labels_by_run[run_name]
            token_count = sum(1 for row in board_int for value in row if value in (1, 2))
            if token_count < min_tokens:
                continue
        else:
            try:
                board_int = resolve_board_labels(run_dir)
            except Exception:
                continue

            if board_int is None:
                continue

            token_count = sum(1 for row in board_int for value in row if value in (1, 2))
            if token_count < min_tokens:
                continue

        patch_paths = sorted(glob.glob(os.path.join(run_dir, '*cell_*_patch.png')))
        generated_patches = []
        if not patch_paths:
            generated_patches = extract_patches_from_original(run_dir)
            if not generated_patches:
                continue

        run_added = 0
        if patch_paths:
            for patch_path in patch_paths:
                match = pattern.search(patch_path)
                if not match:
                    continue

                row = int(match.group(1))
                col = int(match.group(2))
                if row > 2 or col > 2:
                    continue

                image = cv2.imread(patch_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                label = int(board_int[row][col])
                feat = extract_features(image)
                features.append(feat.astype(np.float32))
                labels.append(int(label))
                sources.append(patch_path)
                run_added += 1
        else:
            for row, col, image, source in generated_patches:
                if row > 2 or col > 2:
                    continue
                label = int(board_int[row][col])
                feat = extract_features(image)
                features.append(feat.astype(np.float32))
                labels.append(int(label))
                sources.append(source)
                run_added += 1

        if run_added > 0:
            used_runs += 1

    if not features:
        raise RuntimeError(f'No se pudieron extraer muestras de {debug_dir}')

    x = np.stack(features, axis=0)
    y = np.array(labels, dtype=np.int64)
    src = np.array(sources, dtype=object)

    os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
    np.savez_compressed(out_npz, x=x, y=y, src=src)

    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f'Runs usadas: {used_runs}')
    print(f'Muestras: {len(y)}')
    print(f'Distribución labels (0=vacía,1=X,2=O): {counts}')
    print(f'Dataset guardado en: {out_npz}')


def main():
    parser = argparse.ArgumentParser(description='Construye dataset de celdas desde runs (capturas o debug_steps)')
    parser.add_argument('--debug-dir', default='capturas', help='Carpeta con runs (ej: capturas o debug_steps)')
    parser.add_argument('--out', default='ml_data/cell_dataset_capturas.npz', help='Ruta de salida .npz')
    parser.add_argument('--min-tokens', type=int, default=1, help='Mínimo de fichas en tablero para usar un run')
    parser.add_argument('--labels-csv', default=None, help='CSV de etiquetas manuales por run (columnas: run,c00..c22)')
    args = parser.parse_args()

    build_dataset(args.debug_dir, args.out, min_tokens=args.min_tokens, labels_csv=args.labels_csv)


if __name__ == '__main__':
    main()
